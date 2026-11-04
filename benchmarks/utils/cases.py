import torch
import functools
from collections import defaultdict
from typing import Callable, NamedTuple
from torch._inductor import list_mode_options

# per mode; always fullgraph and static
COMPILE_OPTIONS = {
    "default": {},
    "maa": {"mode": "max-autotune-no-cudagraphs"},
}
MODES = ("eager", "default", "maa")
# rows per oracle chunk: all of M in fp64 does not fit
ORACLE_CHUNK_SIZE = 1024
# rows an oracle checks when M has more, and their seed
ORACLE_NUM_ROWS = 1024
ORACLE_ROWS_SEED = 0


class Implementation(NamedTuple):
    fn: Callable
    inputs: dict


class TrainingImplementation(NamedTuple):
    forward: Callable
    inputs: dict
    # inputs to differentiate
    leaves: tuple[str, ...]
    # one per output
    cotangents: tuple[str, ...]
    # aligns outputs with torch's before comparing
    postprocessing_fn: Callable | None = None


class Case(NamedTuple):
    implementations: dict[str, Implementation | TrainingImplementation]
    # fp64 reference per implementation
    oracles: dict[str, Implementation]
    # in return order
    outputs: tuple[str, ...]


def oracle_rows(count: int) -> torch.Tensor | None:
    # rows the oracle checks: all (None) if count <= ORACLE_NUM_ROWS,
    # else a fixed random sample of that many, sorted
    if count <= ORACLE_NUM_ROWS:
        return None
    else:
        generator = torch.Generator(device="cuda")
        generator.manual_seed(ORACLE_ROWS_SEED)
        permutation = torch.randperm(
            n=count,
            generator=generator,
            device="cuda",
        )
        rows = permutation[:ORACLE_NUM_ROWS]
        return rows.sort().values


def select_oracle_rows(
    output: torch.Tensor,
    m: int,
) -> torch.Tensor:
    # the output's rows its oracle checked, when it checked fewer than all
    if output.shape[0] == m:
        return output
    else:
        indices = oracle_rows(count=output.shape[0])
        assert indices.shape[0] == m
        return output.index_select(
            dim=0,
            index=indices,
        )


def _chunk_inputs(
    inputs: dict,
    mapped: list[str],
    leaves: tuple[str, ...],
    start: int,
    stop: int,
    dtype: torch.dtype,
) -> dict:
    # mapped inputs sliced along M, floats widened to dtype, leaves made
    arguments = {}
    for name, value in inputs.items():
        if name in mapped:
            value = value[start:stop]

        if isinstance(value, torch.Tensor) and value.is_floating_point():
            arguments[name] = value.to(dtype=dtype)
        else:
            arguments[name] = value


def make_oracle_fn(
    fn: Callable,
    in_dims: dict[str, int | None],
    out_dims: tuple[int | None, ...],
    reduce_dims: tuple[int | None, ...],
    leaves: tuple[str, ...] = (),
    chunk_size: int = ORACLE_CHUNK_SIZE,
) -> Callable:
    for dim in (*in_dims.values(), *out_dims, *reduce_dims):
        assert dim in (0, None)

    assert len(out_dims) == len(reduce_dims)
    for out_dim, reduce_dim in zip(out_dims, reduce_dims, strict=True):
        if reduce_dim == 0:
            assert out_dim == 0

    mapped = []
    for name, dim in in_dims.items():
        if dim == 0:
            mapped.append(name)

    @functools.wraps(fn)
    def oracle_fn(**kwargs: object) -> tuple:

        return tuple(oracle_outputs)

    return oracle_fn


def _step_forward(
    fn: Callable,
    arguments: dict,
) -> tuple:
    # the forward; returns its outputs
    outputs = fn(**arguments)
    if isinstance(outputs, torch.Tensor):
        return (outputs,)
    else:
        return tuple(outputs)


def _step_backward(
    outputs: tuple,
    leaf_values: list[torch.Tensor],
    grad_outputs: list[torch.Tensor],
) -> tuple:
    # the leaves' grads; returns the outputs, then the grads
    grads = torch.autograd.grad(
        outputs=outputs,
        inputs=leaf_values,
        grad_outputs=grad_outputs,
    )
    detached = []
    for output in outputs:
        detached.append(output.detach())

    return tuple(detached) + grads


def make_step_fn(
    forward: Callable,
    leaves: tuple[str, ...],
    cotangents: tuple[str, ...],
) -> Callable:
    # forward and backward on leaves made beforehand; returns outputs,
    # then grads
    def step_fn(**inputs: object) -> tuple:
        arguments = inputs.copy()
        grad_outputs = []
        for name in cotangents:
            grad_outputs.append(arguments.pop(name))

        leaf_values = []
        for name in leaves:
            leaf_values.append(arguments[name])

        outputs = _step_forward(
            fn=forward,
            arguments=arguments,
        )
        return _step_backward(
            outputs=outputs,
            leaf_values=leaf_values,
            grad_outputs=grad_outputs,
        )

    return step_fn


def make_batch_mean_loss_fn(fn: Callable) -> Callable:
    def loss_fn(count: int, **inputs: object) -> torch.Tensor:
        loss = fn(reduction="sum", **inputs)
        return loss / count

    return loss_fn


def compiled(
    fn: Callable,
    mode: str,
) -> Callable:
    # fn as written for eager, else one torch.compile of it
    if mode == "eager":
        return fn
    else:
        return torch.compile(
            fn,
            fullgraph=True,
            dynamic=False,
            **COMPILE_OPTIONS[mode],
        )
