import math
import torch
from typing import NamedTuple
from einops import rearrange, reduce, repeat

from benchmarks.backends import torch_ops
from benchmarks.utils import data
from benchmarks.utils import models
from benchmarks.utils.data import Sampler, Shape
from coda.core.ops.rope_utils import make_rope_frequencies


class Normal(NamedTuple):
    # a Gaussian's mean and std
    mean: float
    std: float


class Output(NamedTuple):
    # a block output; its cotangent is
    # (B, T, dim_expansion * dim)
    name: str
    dim_expansion: int


class Block(NamedTuple):
    # the outputs that take a cotangent;
    # a final block's one is the loss
    outputs: tuple[Output, ...]
    leaves: tuple[str, ...]
    # the model's last block, which ends
    # with the LM head as its w3
    is_final: bool
    # rope on the whole [q | k | v], v at
    # frequency 0, not on q and k alone
    rope_on_qkv: bool
    # XSA on the previous attention's
    # output, reading its v in place
    has_xsa: bool


# Llama-3's rope theta
ROPE_THETA = 500000.0

# std of a block's input activations;
# v0's is 1
ACTIVATION_STD = 0.5
# max over min of an rmsnorm input's
# rstd across tokens
RSTD_SPREAD = 2.0

# max over min of qk-norm's head mean
# square across tokens and across heads
HEAD_MEAN_SQ_TOKEN_SPREAD = 2.0
HEAD_MEAN_SQ_HEAD_SPREAD = 4.0
# qk-norm's weights: one for q, one for
# k, each head_dim wide
NUM_QK_NORM_WEIGHTS = 2
# largest cosine, either sign, between
# qk-norm's cotangent and a normed head
QK_COTANGENT_MAX_COSINE = 0.6

# range of the cosine between XSA's x
# and v, per head
XSA_COSINE_LOW = -0.2
XSA_COSINE_HIGH = 0.8
# x's norm as a share of v's
XSA_X_NORM_RATIO = 0.3

# the LM head's logits, fit to
# Llama-3.2-1B and Qwen3-0.6B: each
# token's candidates sit at a height
# above its background logits, each
# candidate offset from that height
CANDIDATE_HEIGHT = Normal(
    mean=17.0,
    std=1.1,
)
TARGET_OFFSET = Normal(
    mean=1.8,
    std=4.4,
)
# the rival is the strongest wrong
# candidate
RIVAL_OFFSET = Normal(
    mean=-1.3,
    std=2.7,
)
OTHER_CANDIDATE_OFFSET = Normal(
    mean=0.0,
    std=1.0,
)
# candidates per token: the target,
# the rival and the others
NUM_OTHER_CANDIDATES = 8
NUM_CANDIDATES = 2 + NUM_OTHER_CANDIDATES
# range of the background logits' std,
# and the least variance they keep
BACKGROUND_STD_LOW = 2.0
BACKGROUND_STD_HIGH = 3.2
BACKGROUND_VARIANCE_MIN = 1.0
# keeps the candidates' Gram matrix
# solvable when a candidate repeats
GRAM_RIDGE = 1e-3
# tokens per chunk, to bound memory
LM_HEAD_INPUT_CHUNK_SIZE = 4096
DLOGITS_CHUNK_SIZE = 1024

# epsilon in Muon's input normalization
MUON_EPS = 1e-7

# each shifted block's outputs and
# leaves: block_pre starts at the
# embedding, block sits between
# attentions, and block_post ends at
# the loss
BLOCKS = {
    "transformer.block_pre": Block(
        outputs=(
            Output(name="residual", dim_expansion=1),
            Output(name="qkv", dim_expansion=3),
            Output(name="v", dim_expansion=1),
        ),
        leaves=("x", "w", "wn", "wn2"),
        is_final=False,
        rope_on_qkv=False,
        has_xsa=False,
    ),
    "transformer.block": Block(
        outputs=(
            Output(name="x2", dim_expansion=1),
            Output(name="qkv", dim_expansion=3),
            Output(name="v", dim_expansion=1),
        ),
        leaves=("x0", "o0", "v0", "w0", "w1", "w2", "w3", "wn0", "wn1", "wn2"),
        is_final=False,
        rope_on_qkv=False,
        has_xsa=True,
    ),
    "transformer.block_post": Block(
        outputs=(),
        leaves=("x0", "o0", "v0", "w0", "w1", "w2", "w3", "wn0", "wn1"),
        is_final=True,
        rope_on_qkv=False,
        has_xsa=True,
    ),
    "llama3.block_pre": Block(
        outputs=(
            Output(name="residual", dim_expansion=1),
            Output(name="qkv", dim_expansion=3),
        ),
        leaves=("x", "w", "wn"),
        is_final=False,
        rope_on_qkv=True,
        has_xsa=False,
    ),
    "llama3.block": Block(
        outputs=(
            Output(name="x2", dim_expansion=1),
            Output(name="qkv", dim_expansion=3),
        ),
        leaves=("x0", "y0", "w0", "w1", "w2", "w3", "wn0", "wn1"),
        is_final=False,
        rope_on_qkv=True,
        has_xsa=False,
    ),
    "llama3.block_post": Block(
        outputs=(),
        leaves=("x0", "y0", "w0", "w1", "w2", "w3", "wn0", "wn1"),
        is_final=True,
        rope_on_qkv=True,
        has_xsa=False,
    ),
}


def sequence_positions(shape: Shape) -> torch.Tensor:
    # rope positions for one sequence, starting at the shape's offset
    end = shape.position_offset + models.SEQUENCE_LENGTH
    return torch.arange(
        start=shape.position_offset,
        end=end,
        dtype=torch.int32,
        device="cuda",
    )


def token_positions(shape: Shape) -> torch.Tensor:
    # every token's rope position, (B * T,); each sequence starts again
    # at the shape's offset
    sequence = sequence_positions(shape=shape)
    positions = repeat(
        tensor=sequence,
        pattern="t -> (b t)",
        b=models.batch_size(shape=shape),
    )
    assert positions.is_contiguous()
    return positions


def inv_freq(shape: Shape) -> torch.Tensor:
    # rope's inverse frequency for each pair j of a head,
    # base^(-2j / head_dim), in fp32 as Llama-3 computes it
    even_indices = torch.arange(
        start=0,
        end=shape.head_dim,
        step=2,
        dtype=torch.float32,
        device="cuda",
    )
    exponents = even_indices / shape.head_dim
    powers = ROPE_THETA**exponents
    pair_frequencies = 1.0 / powers
    assert pair_frequencies.is_contiguous()
    return pair_frequencies


def batched_shape(
    shape: Shape,
    width: int,
) -> tuple[int, int, int]:
    # (B, T, width): the step's tokens as a batch of sequences, as a block
    # takes them
    return (
        models.batch_size(shape=shape),
        models.SEQUENCE_LENGTH,
        width,
    )


def split_layout_heads(
    x: torch.Tensor,
    num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    # each head along the last axis, from the interleaved layout, pairs
    # (2j, 2j + 1), to the split layout, pairs (j, j + head_dim / 2)
    assert x.shape[-1] == num_heads * head_dim
    assert head_dim % 2 == 0
    half = head_dim // 2
    return rearrange(
        tensor=x,
        pattern="... (h p two) -> ... (h two p)",
        h=num_heads,
        two=2,
        p=half,
    ).contiguous()


def split_layout_qkv(
    x: torch.Tensor,
    size_qk: int,
    head_dim: int,
) -> torch.Tensor:
    # a [q | k | v] activation in the split layout; v has no rope pairs
    assert x.ndim == 2
    assert 0 < size_qk < x.shape[1]
    assert size_qk % head_dim == 0
    num_heads = size_qk // head_dim
    qk = x[:, :size_qk]
    v = x[:, size_qk:]
    split_qk = split_layout_heads(
        x=qk,
        num_heads=num_heads,
        head_dim=head_dim,
    )
    split = torch.cat(
        tensors=(split_qk, v),
        dim=1,
    )
    assert split.is_contiguous()
    return split


def split_layout_rope_weight(
    B: torch.Tensor,
    size_qk: int,
    head_dim: int,
) -> torch.Tensor:
    # B is the transposed weight: its q and k rows move to the split
    # layout, any v rows stay, and it is transposed back to keep B's strides
    weight = B.mT
    assert weight.ndim == 2
    assert weight.is_contiguous()
    assert 0 < size_qk <= weight.shape[0]
    assert head_dim % 2 == 0
    weight_qk = weight[:size_qk]
    weight_v = weight[size_qk:]
    half = head_dim // 2
    weight_split_qk = rearrange(
        tensor=weight_qk,
        pattern="(h p two) k -> (h two p) k",
        two=2,
        p=half,
    )
    weight_split = torch.cat(
        tensors=(weight_split_qk, weight_v),
        dim=0,
    )
    assert weight_split.is_contiguous()
    return weight_split.mT


def split_layout_swiglu_weight(B: torch.Tensor) -> torch.Tensor:
    # B is the transposed weight: its gate and up rows are regrouped into
    # torch's split layout, all gates then all ups, and transposed back to
    # keep B's strides
    split = rearrange(
        tensor=B.mT,
        pattern="(pair two) k -> (two pair) k",
        two=2,
    ).contiguous()
    return split.mT


def cache_rope_cos_sin(
    positions: torch.Tensor,
    frequencies: torch.Tensor,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # fp32 cos and sin tables of one head's rope angles; the angles are
    # computed in fp64, so the tables rotate by the oracle's angles
    assert frequencies.ndim == 1
    assert frequencies.shape[0] >= head_dim
    assert frequencies.shape[0] % head_dim == 0
    head_frequencies = frequencies[:head_dim]
    head_frequencies_fp64 = head_frequencies.to(dtype=torch.float64)
    positions_fp64 = positions.to(dtype=torch.float64)
    cos, sin = torch_ops.rope_tables(
        positions=positions_fp64,
        frequencies=head_frequencies_fp64,
    )
    return cos.to(dtype=torch.float32), sin.to(dtype=torch.float32)


def measure_head_norms(
    x: torch.Tensor,
    head_dim: int,
) -> torch.Tensor:
    # each token's norm per head, (tokens, heads), in fp32
    x_fp32 = x.to(dtype=torch.float32)
    heads = rearrange(
        tensor=x_fp32,
        pattern="m (h d) -> m h d",
        d=head_dim,
    )
    return heads.norm(dim=-1)


def count_non_ignored_targets(target: torch.Tensor) -> int:
    # the number of targets not set to IGNORE_INDEX
    non_ignored = (target != data.IGNORE_INDEX)
    return non_ignored.sum().item()


def calibrate_cosines(
    x: torch.Tensor,
    direction: torch.Tensor,
    cosine: torch.Tensor,
    norm: torch.Tensor,
    head_dim: int,
) -> torch.Tensor:
    # x aligned with direction, per head, to the given cosine and norm, in
    # fp32; the rest of it points along x's part perpendicular to direction
    direction_fp32 = direction.to(dtype=torch.float32)
    direction_heads = rearrange(
        tensor=direction_fp32,
        pattern="m (h d) -> m h d",
        d=head_dim,
    )
    x_fp32 = x.to(dtype=torch.float32)
    x_heads = rearrange(
        tensor=x_fp32,
        pattern="m (h d) -> m h d",
        d=head_dim,
    )

    # x's part along direction, and its part perpendicular to it
    direction_norms = direction_heads.norm(
        dim=-1,
        keepdim=True,
    )
    direction_unit = direction_heads / direction_norms
    dot_terms = x_heads * direction_unit
    x_dot_direction = reduce(
        tensor=dot_terms,
        pattern="m h d -> m h 1",
        reduction="sum",
    )
    x_parallel = x_dot_direction * direction_unit
    x_perpendicular = x_heads - x_parallel
    x_perpendicular_norms = x_perpendicular.norm(
        dim=-1,
        keepdim=True,
    )
    perpendicular_unit = x_perpendicular / x_perpendicular_norms

    # the unit vector at the given cosine to direction, on x's side
    cosine_fp32 = cosine.to(dtype=torch.float32)
    cosine_heads = rearrange(
        tensor=cosine_fp32,
        pattern="m h -> m h 1",
    )
    cosine_squared = cosine_heads.square()
    sine_squared = 1.0 - cosine_squared
    sine_heads = torch.sqrt(sine_squared)
    parallel_part = cosine_heads * direction_unit
    perpendicular_part = sine_heads * perpendicular_unit
    aligned_unit = parallel_part + perpendicular_part

    # scaled to the given norm
    norm_fp32 = norm.to(dtype=torch.float32)
    norm_heads = rearrange(
        tensor=norm_fp32,
        pattern="m h -> m h 1",
    )
    aligned_heads = aligned_unit * norm_heads
    return rearrange(
        tensor=aligned_heads,
        pattern="m h d -> m (h d)",
    )


def calibrate_logits(
    x: torch.Tensor,
    candidate_weights: torch.Tensor,
    logits: torch.Tensor,
) -> torch.Tensor:
    # x (num_tokens, dim) moved by the smallest step that makes its logits on
    # each token's candidate weights (num_tokens, num_logits, dim) equal
    # logits (num_tokens, num_logits); the step lies within those weights, so
    # x's other logits barely change
    assert x.ndim == 2
    num_tokens, dim = x.shape
    assert candidate_weights.ndim == 3
    num_logits = candidate_weights.shape[1]
    assert candidate_weights.shape == (num_tokens, num_logits, dim)
    assert logits.shape == (num_tokens, num_logits)

    # x's current logits on its candidates, and their gap to the targets
    x_column = rearrange(
        tensor=x,
        pattern="m d -> m d 1",
    )
    current_logits = torch.bmm(
        input=candidate_weights,
        mat2=x_column,
    )
    current_logits = rearrange(
        tensor=current_logits,
        pattern="m n 1 -> m n",
    )
    logit_gap = logits - current_logits

    # how much of each candidate weight the step adds: solved from the
    # weights' dot products, plus a small ridge that keeps the system
    # solvable when a token drew the same candidate twice
    candidate_dots = torch.bmm(
        input=candidate_weights,
        mat2=candidate_weights.mT,
    )
    identity = torch.eye(
        n=num_logits,
        dtype=torch.float32,
        device="cuda",
    )
    ridged_dots = candidate_dots + GRAM_RIDGE * identity
    step_coefficients = torch.linalg.solve(
        A=ridged_dots,
        B=logit_gap,
    )

    # the step: step_coefficients times the candidate weights
    step_coefficients = rearrange(
        tensor=step_coefficients,
        pattern="m n -> m 1 n",
    )
    step = torch.bmm(
        input=step_coefficients,
        mat2=candidate_weights,
    )
    step = rearrange(
        tensor=step,
        pattern="m 1 d -> m d",
    )
    return x + step


def gemm_operands(
    shape: Shape,
    out_features: int,
    sampler: Sampler,
    dtype: torch.dtype,
) -> dict:
    # a linear layer's inputs: A (tokens, dim), and B, the transposed
    # weight, k-major as nn.Linear stores it
    weight = sampler.weight(
        shape=(out_features, shape.dim),
        dtype=dtype,
    )
    A = sampler.randn(
        shape=(shape.tokens, shape.dim),
        dtype=dtype,
    )
    return {
        "A": A,
        "B": weight.mT,
    }


def rmsnorm_operands(
    shape: Shape,
    x: torch.Tensor,
    rstd_scale: float,
    sampler: Sampler,
    dtype: torch.dtype,
) -> dict:
    # an rmsnorm input A and its rstd: x rescaled per token so its rstd lies
    # near rstd_scale, a value measured on real models, which keeps A's
    # scale realistic
    row_multipliers = sampler.multipliers(
        size=shape.tokens,
        ratio=RSTD_SPREAD,
        dtype=torch.float32,
    )
    x_fp32 = x.to(dtype=torch.float32)
    x_square = x_fp32.square()
    mean_square = reduce(
        tensor=x_square,
        pattern="m n -> m",
        reduction="mean",
    )
    rms = torch.sqrt(mean_square)
    scaled_rms = rstd_scale * rms
    row_scale = row_multipliers / scaled_rms
    row_scale = rearrange(
        tensor=row_scale,
        pattern="m -> m 1",
    )
    A_fp32 = x_fp32 * row_scale
    A = A_fp32.to(dtype=dtype)

    # rstd of A after its rounding to dtype, which is what the GEMM reads
    rstd = torch_ops.rmsnorm_rstd(
        x=A.to(dtype=torch.float32),
        eps=data.NORM_EPS,
    )
    return {
        "A": A,
        "rstd": rstd,
    }


def rope_operands(shape: Shape) -> dict:
    # rope's positions, its per-column frequencies for [q | k | v] and for
    # [q | k], and the widths of q, k and v
    size_q = shape.heads * shape.head_dim
    size_k = shape.kv_heads * shape.head_dim
    size_v = shape.kv_heads * shape.head_dim
    pair_frequencies = inv_freq(shape=shape)
    qk_heads = shape.heads + shape.kv_heads
    frequencies_qk = make_rope_frequencies(
        inv_freq=pair_frequencies,
        num_heads=qk_heads,
    )

    # v's columns get frequency 0, which leaves them unrotated
    frequencies_v = torch.zeros(
        size=(size_v,),
        dtype=frequencies_qk.dtype,
        device=frequencies_qk.device,
    )
    frequencies_qkv = torch.cat(
        tensors=(frequencies_qk, frequencies_v),
        dim=0,
    )
    positions = token_positions(shape=shape)
    return {
        "positions": positions,
        "frequencies_qkv": frequencies_qkv,
        "frequencies_qk": frequencies_qk,
        "size_q": size_q,
        "size_k": size_k,
        "size_v": size_v,
    }


def muon_iterate(
    weight: torch.Tensor,
    steps: int,
    coefficients: tuple[float, float, float],
) -> torch.Tensor:
    # Muon's Newton-Schulz iterate after this many steps, computed in fp32:
    # the weight divided by its norm, then X = a X + (b G + c G G) X with
    # G = X X^T at each step
    a, b, c = coefficients
    weight_fp32 = weight.to(dtype=torch.float32)
    norm = torch.linalg.vector_norm(weight_fp32)
    X = weight_fp32 / (norm + MUON_EPS)

    for _ in range(steps):
        gram = torch.mm(
            input=X,
            mat2=X.mT,
        )
        gram_squared = torch.mm(
            input=gram,
            mat2=gram,
        )
        polynomial = b * gram + c * gram_squared
        update = torch.mm(
            input=polynomial,
            mat2=X,
        )
        X = a * X + update

    return X


def calibrate_qk_cotangent(
    shape: Shape,
    random_cotangent: torch.Tensor,
    qk_preact: torch.Tensor,
    gamma: torch.Tensor,
    sampler: Sampler,
    dtype: torch.dtype,
) -> torch.Tensor:
    # a realistic cotangent on the rotated q and k heads, made from a random
    # one; qk_preact holds q and k before qk-norm, and gamma the norm's
    # weight, one q head's then one k head's

    # every head's norm weight: q's for the q heads, k's for the k heads
    num_heads = shape.heads + shape.kv_heads
    gamma_fp32 = gamma.to(dtype=torch.float32)
    gamma_q, gamma_k = torch.split(
        tensor=gamma_fp32,
        split_size_or_sections=(shape.head_dim, shape.head_dim),
    )
    gamma_per_q_head = repeat(
        tensor=gamma_q,
        pattern="d -> h d",
        h=shape.heads,
    )
    gamma_per_k_head = repeat(
        tensor=gamma_k,
        pattern="d -> h d",
        h=shape.kv_heads,
    )
    gamma_per_head = torch.cat(tensors=(gamma_per_q_head, gamma_per_k_head))

    # q and k after qk-norm, up to each head's scale: the norm divides a
    # head by one scalar, its rstd, before multiplying by gamma
    qk_fp32 = qk_preact.to(dtype=torch.float32)
    qk_per_head = rearrange(
        tensor=qk_fp32,
        pattern="m (h d) -> m h d",
        d=shape.head_dim,
    )
    normed_qk_per_head = qk_per_head * gamma_per_head
    normed_qk = rearrange(
        tensor=normed_qk_per_head,
        pattern="m h d -> m (h d)",
    )

    # each head of the random cotangent turned to a cosine with its normed
    # head drawn within QK_COTANGENT_MAX_COSINE, keeping its length; left
    # random, the cosine sits near 0.1 and shrinks qk-norm's projection term
    # to a tenth
    cosine_per_head = sampler.uniform(
        shape=(shape.tokens, num_heads),
        low=-QK_COTANGENT_MAX_COSINE,
        high=QK_COTANGENT_MAX_COSINE,
        dtype=torch.float32,
    )
    norm_per_head = measure_head_norms(
        x=random_cotangent,
        head_dim=shape.head_dim,
    )
    cotangent_before_rope = calibrate_cosines(
        x=random_cotangent,
        direction=normed_qk,
        cosine=cosine_per_head,
        norm=norm_per_head,
        head_dim=shape.head_dim,
    )

    # then rotated by the forward's rope angles, since the cotangent arrives
    # on the rotated heads; rope takes (tokens, heads, head_dim)
    samples_rope = rope_operands(shape=shape)
    frequencies = samples_rope["frequencies_qk"]
    assert frequencies.shape[0] == num_heads * shape.head_dim
    cos, sin = cache_rope_cos_sin(
        positions=samples_rope["positions"],
        frequencies=frequencies,
        head_dim=shape.head_dim,
    )
    cotangent_before_rope_per_head = rearrange(
        tensor=cotangent_before_rope,
        pattern="m (h d) -> m h d",
        d=shape.head_dim,
    )
    cotangent_per_head = torch_ops.rope(
        x=cotangent_before_rope_per_head,
        cos=cos,
        sin=sin,
        layout=torch_ops.PairLayout.interleaved,
    )
    cotangent = rearrange(
        tensor=cotangent_per_head,
        pattern="m h d -> m (h d)",
    )
    return cotangent.to(dtype=dtype)


def calibrate_lm_head_input(
    shape: Shape,
    random_input: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    sampler: Sampler,
) -> torch.Tensor:
    # the LM head's input A (tokens, dim), the final norm's output, in fp32:
    # random_input calibrated so the logits A @ weight^T follow the fit at
    # the top, each position's candidates at drawn logits over a random
    # background
    assert random_input.shape == (shape.tokens, shape.dim)
    assert weight.shape == (shape.vocab_size, shape.dim)
    assert target.shape == (shape.tokens,)

    # each position's candidates: its target first, or a random token where
    # the target is ignored, then a rival and the others
    candidate_tokens = sampler.tokens(
        shape=(shape.tokens, NUM_CANDIDATES),
        vocab_size=shape.vocab_size,
    )
    non_ignored = (target != data.IGNORE_INDEX)
    candidate_tokens[:, 0] = torch.where(
        condition=non_ignored,
        input=target,
        other=candidate_tokens[:, 0],
    )

    # their logits: each token's height above its background, plus each
    # candidate's offset from it
    height_noise = sampler.randn(
        shape=(shape.tokens, 1),
        dtype=torch.float32,
    )
    height_offset = CANDIDATE_HEIGHT.std * height_noise
    candidate_height = CANDIDATE_HEIGHT.mean + height_offset
    offset_noise = sampler.randn(
        shape=(shape.tokens, NUM_CANDIDATES),
        dtype=torch.float32,
    )
    background_std = sampler.log_uniform(
        shape=(shape.tokens,),
        low=BACKGROUND_STD_LOW,
        high=BACKGROUND_STD_HIGH,
        dtype=torch.float32,
    )
    other_means = [OTHER_CANDIDATE_OFFSET.mean] * NUM_OTHER_CANDIDATES
    other_stds = [OTHER_CANDIDATE_OFFSET.std] * NUM_OTHER_CANDIDATES
    offset_means = torch.tensor(
        data=[TARGET_OFFSET.mean, RIVAL_OFFSET.mean] + other_means,
        dtype=torch.float32,
        device="cuda",
    )
    offset_stds = torch.tensor(
        data=[TARGET_OFFSET.std, RIVAL_OFFSET.std] + other_stds,
        dtype=torch.float32,
        device="cuda",
    )
    mean_logits = candidate_height + offset_means
    candidate_logits = mean_logits + offset_stds * offset_noise

    # the background: the random input, scaled so each position's other
    # logits have std background_std; each candidate's weight also adds its
    # logit times a unit-variance draw to every other logit, so the
    # background keeps only the variance left, at least
    # BACKGROUND_VARIANCE_MIN
    candidate_logits_squared = candidate_logits.square()
    candidate_sum_squares = reduce(
        tensor=candidate_logits_squared,
        pattern="m n -> m",
        reduction="sum",
    )
    candidate_variance = candidate_sum_squares / shape.dim
    variance_left = background_std.square() - candidate_variance
    background_variance = torch.clamp(
        input=variance_left,
        min=BACKGROUND_VARIANCE_MIN,
    )
    background_scale = torch.sqrt(background_variance)
    background_scale = rearrange(
        tensor=background_scale,
        pattern="m -> m 1",
    )
    random_input_fp32 = random_input.to(dtype=torch.float32)
    background = random_input_fp32 * background_scale

    # the background moved so the candidates' logits are the drawn ones,
    # LM_HEAD_INPUT_CHUNK_SIZE positions at a time to bound the candidate
    # weights' memory
    A = torch.empty_like(background)
    assert shape.tokens % LM_HEAD_INPUT_CHUNK_SIZE == 0
    chunk_shape = (LM_HEAD_INPUT_CHUNK_SIZE, shape.dim)
    candidates_shape = (LM_HEAD_INPUT_CHUNK_SIZE, NUM_CANDIDATES)
    for start in range(0, shape.tokens, LM_HEAD_INPUT_CHUNK_SIZE):
        stop = start + LM_HEAD_INPUT_CHUNK_SIZE
        background_chunk = background[start:stop, ...]
        candidate_tokens_chunk = candidate_tokens[start:stop, ...]
        candidate_logits_chunk = candidate_logits[start:stop, ...]
        assert background_chunk.shape == chunk_shape
        assert candidate_tokens_chunk.shape == candidates_shape
        assert candidate_logits_chunk.shape == candidates_shape
        candidate_weights = weight[candidate_tokens_chunk]
        candidate_weights_fp32 = candidate_weights.to(dtype=torch.float32)
        A_chunk = calibrate_logits(
            x=background_chunk,
            candidate_weights=candidate_weights_fp32,
            logits=candidate_logits_chunk,
        )
        assert A_chunk.shape == chunk_shape
        A[start:stop, ...] = A_chunk

    return A


def lm_head_dlogits(
    A: torch.Tensor,
    B: torch.Tensor,
    target: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    # the cross-entropy gradient on the LM head's logits: each token's sums
    # to zero, with most of its weight on the target
    assert A.ndim == 2
    num_tokens, dim = A.shape
    assert B.ndim == 2
    assert B.shape[0] == dim
    assert target.shape == (num_tokens,)
    assert num_tokens % DLOGITS_CHUNK_SIZE == 0
    dlogits_chunks = []

    for start in range(0, num_tokens, DLOGITS_CHUNK_SIZE):
        stop = start + DLOGITS_CHUNK_SIZE
        A_chunk = A[start:stop]
        target_chunk = target[start:stop]
        assert A_chunk.shape == (DLOGITS_CHUNK_SIZE, dim)
        assert target_chunk.shape == (DLOGITS_CHUNK_SIZE,)
        logits = torch.mm(
            input=A_chunk,
            mat2=B,
            out_dtype=torch.float32,
        )
        _, dlogits = torch_ops.cross_entropy(
            logits=logits,
            target=target_chunk,
            ignore_index=data.IGNORE_INDEX,
            cross_entropy_numerics=torch_ops.CrossEntropyNumerics.precise,
        )
        dlogits_chunks.append(dlogits.to(dtype=dtype))

    return torch.cat(tensors=dlogits_chunks)


def block_input(
    shape: Shape,
    name: str,
    is_final: bool,
    sampler: Sampler,
    dtype: torch.dtype,
) -> torch.Tensor:
    # one of a block's inputs, by name, at the usual init scale; w and w3
    # project to [q | k | v] and w1 to the MLP's gate and up; an activation
    # is (B, T, dim)
    weight_shapes = {
        "w": (3 * shape.dim, shape.dim),
        "w0": (shape.dim, shape.dim),
        "w1": (2 * shape.ffn, shape.dim),
        "w2": (shape.dim, shape.ffn),
        "w3": (3 * shape.dim, shape.dim),
    }

    # the final block takes the LM head as w3
    if is_final:
        weight_shapes["w3"] = (shape.vocab_size, shape.dim)

    # the norm weights' sizes; wn2 is q's and k's per-head norm weights, one
    # head each
    norm_sizes = {
        "wn": shape.dim,
        "wn0": shape.dim,
        "wn1": shape.dim,
        "wn2": NUM_QK_NORM_WEIGHTS * shape.head_dim,
    }
    # the activations' stds: ACTIVATION_STD, and 1 for v0, the previous
    # block's v
    activation_stds = {
        "x": ACTIVATION_STD,
        "x0": ACTIVATION_STD,
        "o0": ACTIVATION_STD,
        "y0": ACTIVATION_STD,
        "v0": 1.0,
    }
    # each name is in exactly one table
    tables = (weight_shapes, norm_sizes, activation_stds)
    assert sum(name in table for table in tables) == 1

    if name in weight_shapes:
        return sampler.weight(
            shape=weight_shapes[name],
            dtype=dtype,
        )
    elif name in norm_sizes:
        return sampler.gamma(
            size=norm_sizes[name],
            dtype=dtype,
        )
    else:
        values = sampler.randn(
            shape=batched_shape(
                shape=shape,
                width=shape.dim,
            ),
            dtype=dtype,
        )
        return activation_stds[name] * values


def gemm_xsa_bwd_operands(
    shape: Shape,
    sampler: Sampler,
    dtype: torch.dtype,
) -> dict:
    # attention's output x and its value v, and the output projection's
    # cotangent A and k-major weight B
    v = sampler.randn(
        shape=(shape.tokens, shape.dim),
        dtype=dtype,
    )
    random_x = sampler.randn(
        shape=(shape.tokens, shape.dim),
        dtype=dtype,
    )
    A = sampler.randn(
        shape=(shape.tokens, shape.dim),
        dtype=dtype,
    )
    weight = sampler.weight(
        shape=(shape.dim, shape.dim),
        dtype=dtype,
    )

    # per head, x is XSA_X_NORM_RATIO of v's length, at a drawn cosine to v
    assert shape.dim % shape.head_dim == 0
    num_heads = shape.dim // shape.head_dim
    cosines_to_v = sampler.uniform(
        shape=(shape.tokens, num_heads),
        low=XSA_COSINE_LOW,
        high=XSA_COSINE_HIGH,
        dtype=torch.float32,
    )
    v_norms = measure_head_norms(
        x=v,
        head_dim=shape.head_dim,
    )
    x_norms = XSA_X_NORM_RATIO * v_norms
    x_fp32 = calibrate_cosines(
        x=random_x,
        direction=v,
        cosine=cosines_to_v,
        norm=x_norms,
        head_dim=shape.head_dim,
    )
    x = x_fp32.to(dtype=dtype)

    # under causal attention, each sequence's first token sees only
    # itself, so its attention output x is its value v
    assert shape.tokens % models.SEQUENCE_LENGTH == 0
    x[:: models.SEQUENCE_LENGTH, ...] = v[:: models.SEQUENCE_LENGTH, ...]

    # XSA reads v in place from a [Q | K | V] buffer; the rest is NaN, so
    # any read outside v shows up
    buffer = torch.full(
        size=(shape.tokens, 3 * shape.dim),
        fill_value=math.nan,
        dtype=v.dtype,
        device=v.device,
    )
    v_start = 2 * shape.dim
    v_packed = buffer[:, v_start:]
    v_packed.copy_(v)
    B = weight.mT
    return {
        "x": x,
        "v": v_packed,
        "head_dim": shape.head_dim,
        "A": A,
        "B": B,
    }


def qknorm_rope_bwd_operands(
    shape: Shape,
    rmsnorm: bool,
    sampler: Sampler,
    dtype: torch.dtype,
) -> dict:
    # what qk-norm rope's forward saves for its backward: gamma (one q
    # head's, then one k head's), preact, head_mean_sq, and rstd with rmsnorm
    heads = shape.heads + shape.kv_heads

    # z is what qk-norm normalizes, at the shape's head mean square: the
    # GEMM's fp32 accumulator, or with rmsnorm rstd times it
    noise = sampler.randn(
        shape=(shape.tokens, heads * shape.head_dim),
        dtype=torch.float32,
    )
    row_scales = sampler.multipliers(
        size=shape.tokens,
        ratio=HEAD_MEAN_SQ_TOKEN_SPREAD,
        dtype=torch.float32,
    )
    head_scales = sampler.multipliers(
        size=heads,
        ratio=HEAD_MEAN_SQ_HEAD_SPREAD,
        dtype=torch.float32,
    )
    row_scales = rearrange(
        tensor=row_scales,
        pattern="m -> m 1",
    )
    row_mean_square = shape.statistics.head_mean_sq * row_scales
    mean_square = row_mean_square * head_scales
    rms = torch.sqrt(mean_square)
    column_rms = repeat(
        tensor=rms,
        pattern="m h -> m (h d)",
        d=shape.head_dim,
    )
    z = noise * column_rms
    gamma = sampler.gamma(
        size=NUM_QK_NORM_WEIGHTS * shape.head_dim,
        dtype=dtype,
    )
    inputs = {"gamma": gamma}

    if rmsnorm:
        # rstd at the input norm's level, varied across tokens; preact is
        # the accumulator before rstd scales it
        multipliers = sampler.multipliers(
            size=shape.tokens,
            ratio=HEAD_MEAN_SQ_TOKEN_SPREAD,
            dtype=torch.float32,
        )
        rstd = shape.statistics.rstd * multipliers
        rstd_column = rearrange(
            tensor=rstd,
            pattern="m -> m 1",
        )
        accumulator = z / rstd_column
        preact = accumulator.to(dtype=dtype)
        preact_fp32 = preact.to(dtype=torch.float32)
        qknorm_input = preact_fp32 * rstd_column
        inputs["rstd"] = rstd
    else:
        preact = z.to(dtype=dtype)
        qknorm_input = preact.to(dtype=torch.float32)

    inputs["preact"] = preact

    # head_mean_sq as the backward forms it: from qk-norm's input, rebuilt
    # from the stored preact
    inputs["head_mean_sq"] = torch_ops.head_mean_square(
        x=qknorm_input,
        head_dim=shape.head_dim,
    )
    return inputs


def qknorm_rope_bwd_zdz_operands(
    shape: Shape,
    preact: torch.Tensor,
    gamma: torch.Tensor,
    size_v: int,
    sampler: Sampler,
    dtype: torch.dtype,
) -> dict:
    # what the zdz backward reads besides the forward's saved tensors: dy,
    # the cotangent of [q | k | v] (from calibrate_qk_cotangent on q and k,
    # so zdz is a well-conditioned sum, and random on v); v; and dv, v's
    # gradient from attention, which the backward adds to dy on v's columns
    random = sampler.randn(
        shape=preact.shape,
        dtype=dtype,
    )
    cotangent_qk = calibrate_qk_cotangent(
        shape=shape,
        random_cotangent=random,
        qk_preact=preact,
        gamma=gamma,
        sampler=sampler,
        dtype=dtype,
    )
    cotangent_v = sampler.randn(
        shape=(shape.tokens, size_v),
        dtype=dtype,
    )
    dy = torch.cat(
        tensors=(cotangent_qk, cotangent_v),
        dim=1,
    )
    v = sampler.randn(
        shape=(shape.tokens, size_v),
        dtype=dtype,
    )
    dv = sampler.randn(
        shape=(shape.tokens, size_v),
        dtype=dtype,
    )
    return {
        "dy": dy,
        "v": v,
        "dv": dv,
    }


def lm_head_operands(
    shape: Shape,
    rmsnorm: bool,
    sampler: Sampler,
    dtype: torch.dtype,
) -> dict:
    # the LM head's inputs: the final norm's output A, the weight
    # transposed as B, and targets with some ignored
    samples_gemm = gemm_operands(
        shape=shape,
        out_features=shape.vocab_size,
        sampler=sampler,
        dtype=dtype,
    )
    weight = samples_gemm["B"].mT
    target = sampler.targets(
        size=shape.tokens,
        vocab_size=shape.vocab_size,
    )

    A_fp32 = calibrate_lm_head_input(
        shape=shape,
        random_input=samples_gemm["A"],
        weight=weight,
        target=target,
        sampler=sampler,
    )

    if rmsnorm:
        # rmsnorm rescales each token of A / gamma to unit rms, so B carries
        # gamma and the tokens' median rms to keep the logits near the plain
        # op's
        gamma = sampler.gamma(
            size=shape.dim,
            dtype=torch.float32,
        )
        A_over_gamma = A_fp32 / gamma
        mean_square = reduce(
            tensor=A_over_gamma.square(),
            pattern="m n -> m",
            reduction="mean",
        )
        median_rms = torch.sqrt(mean_square).median()
        weight_times_gamma = weight * gamma * median_rms
        samples_rmsnorm = rmsnorm_operands(
            shape=shape,
            x=A_over_gamma,
            rstd_scale=shape.statistics.final_rstd,
            sampler=sampler,
            dtype=dtype,
        )
        return {
            "A": samples_rmsnorm["A"],
            "B": weight_times_gamma.to(dtype=dtype).mT,
            "rstd": samples_rmsnorm["rstd"],
            "target": target,
        }
    else:
        return {
            "A": A_fp32.to(dtype=dtype),
            "B": samples_gemm["B"],
            "target": target,
        }


def linear_qknorm_rope_operands(
    shape: Shape,
    size_q: int,
    size_k: int,
    size_v: int,
    sampler: Sampler,
    dtype: torch.dtype,
) -> dict:
    # a linear_qknorm_rope step's tensors: the [q | k | v] weight, x,
    # gamma, and the cotangents dq, dk and dv; dq and dk come from
    # calibrate_qk_cotangent
    size_qk = size_q + size_k
    weight = sampler.weight(
        shape=(size_qk + size_v, shape.dim),
        dtype=dtype,
    )
    x = sampler.randn(
        shape=(shape.tokens, shape.dim),
        dtype=dtype,
    )
    gamma = sampler.gamma(
        size=NUM_QK_NORM_WEIGHTS * shape.head_dim,
        dtype=dtype,
    )
    random_dq = sampler.randn(
        shape=(shape.tokens, size_q),
        dtype=dtype,
    )
    random_dk = sampler.randn(
        shape=(shape.tokens, size_k),
        dtype=dtype,
    )
    dv = sampler.randn(
        shape=(shape.tokens, size_v),
        dtype=dtype,
    )

    # q and k before the norm, as the GEMM's fp32 accumulator holds them
    weight_qk = weight[:size_qk]
    assert weight_qk.shape == (size_qk, shape.dim)
    accumulator = torch.mm(
        input=x,
        mat2=weight_qk.mT,
        out_dtype=torch.float32,
    )
    random_qk = torch.cat(
        tensors=(random_dq, random_dk),
        dim=1,
    )
    cotangent = calibrate_qk_cotangent(
        shape=shape,
        random_cotangent=random_qk,
        qk_preact=accumulator,
        gamma=gamma,
        sampler=sampler,
        dtype=dtype,
    )
    cotangent_q, cotangent_k = torch.split(
        tensor=cotangent,
        split_size_or_sections=(size_q, size_k),
        dim=1,
    )
    return {
        "weight": weight,
        "x": x,
        "gamma": gamma,
        "dq": cotangent_q.contiguous(),
        "dk": cotangent_k.contiguous(),
        "dv": dv,
    }


def gemm_residual_rmsnorm_gemm_operands(
    shape: Shape,
    in_features: int,
    out_features: int,
    sampler: Sampler,
    dtype: torch.dtype,
) -> dict:
    # gemm_residual_rmsnorm_gemm's leaves, drawn as block_input draws them:
    # the residual stream x; y and w0, the GEMM that adds onto it; w1, the
    # next GEMM's weight; and wn, the norm's weight between them
    x = sampler.randn(
        shape=(shape.tokens, shape.dim),
        dtype=dtype,
    )
    y = sampler.randn(
        shape=(shape.tokens, in_features),
        dtype=dtype,
    )
    w0 = sampler.weight(
        shape=(shape.dim, in_features),
        dtype=dtype,
    )
    w1 = sampler.weight(
        shape=(out_features, shape.dim),
        dtype=dtype,
    )
    wn = sampler.gamma(
        size=shape.dim,
        dtype=dtype,
    )
    return {
        "x": ACTIVATION_STD * x,
        "y": ACTIVATION_STD * y,
        "w0": w0,
        "w1": w1,
        "wn": wn,
    }


def block_operands(
    shape: Shape,
    name: str,
    sampler: Sampler,
    dtype: torch.dtype,
) -> dict:
    # a block's leaves, a cotangent per output and rope's operands; name
    # is family and place, e.g. llama3.block_post. Activations and their
    # cotangents are (B, T, width), weights 2D, positions (B * T,)
    block = BLOCKS[name]
    inputs = {}

    for leaf_name in block.leaves:
        inputs[leaf_name] = block_input(
            shape=shape,
            name=leaf_name,
            is_final=block.is_final,
            sampler=sampler,
            dtype=dtype,
        )

    # the previous block's v, read in place: the last columns of a
    # NaN-filled [Q | K | V], so any read outside them shows up
    if block.has_xsa:
        v0 = inputs["v0"]
        batch, length, width = v0.shape
        buffer = torch.full(
            size=(batch, length, 2 * shape.dim + width),
            fill_value=math.nan,
            dtype=v0.dtype,
            device=v0.device,
        )
        v_columns = buffer[..., 2 * shape.dim :]
        v_columns.copy_(v0)
        inputs["v0"] = v_columns

    # a final block's loss: its cotangent is 1, as a backward starts, and
    # the targets have some ignored
    if block.is_final:
        inputs["grad_loss"] = torch.ones(
            size=(),
            dtype=torch.float32,
            device="cuda",
        )
        targets = sampler.targets(
            size=shape.tokens,
            vocab_size=shape.vocab_size,
        )
        # one per token, (B, T) as a batch's labels
        inputs["targets"] = rearrange(
            tensor=targets,
            pattern="(b t) -> b t",
            t=models.SEQUENCE_LENGTH,
        )

    # a cotangent per other output, at its declared width
    for output in block.outputs:
        inputs[f"grad_{output.name}"] = sampler.randn(
            shape=batched_shape(
                shape=shape,
                width=output.dim_expansion * shape.dim,
            ),
            dtype=dtype,
        )

    inputs["positions"] = token_positions(shape=shape)
    pair_frequencies = inv_freq(shape=shape)

    # q's and k's heads alone; a block has as many k heads as q heads
    assert shape.kv_heads == shape.heads
    assert shape.heads * shape.head_dim == shape.dim
    qk_heads = 2 * shape.heads
    frequencies = make_rope_frequencies(
        inv_freq=pair_frequencies,
        num_heads=qk_heads,
    )
    assert frequencies.shape == (2 * shape.dim,)

    # v's columns at frequency 0, unrotated
    if block.rope_on_qkv:
        v_frequencies = torch.zeros(
            size=(shape.dim,),
            dtype=frequencies.dtype,
            device=frequencies.device,
        )
        frequencies = torch.cat(tensors=(frequencies, v_frequencies))
        assert frequencies.shape == (3 * shape.dim,)

    inputs["frequencies"] = frequencies
    return inputs
