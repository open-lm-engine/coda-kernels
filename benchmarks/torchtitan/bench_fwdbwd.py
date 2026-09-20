import os
import json
import torch
import argparse
from typing import Callable
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.models.llama3 import Transformer, TransformerModelArgs

from benchmarks import bench_utils
from benchmarks.torchtitan import liger_utils
from coda.kernels.blocks.llama3 import ALLOW_INPLACE_GRAD_OUTPUT


_BATCH = 4
_LENGTH = 8192

_NUM_WARMUP = 10
_NUM_ITERATIONS = 30
_NUM_TRACE_WARMUP = 3
_NUM_TRACE_ITERATIONS = 5

_MODEL_ARGS = TransformerModelArgs(
    dim=2048,
    n_layers=16,
    n_heads=32,
    # llama3 1B ships GQA: 32 query heads over 8 KV heads
    n_kv_heads=8,
    ffn_dim_multiplier=1.5,
    multiple_of=1024,
    rope_theta=500000,
    max_seq_len=_LENGTH,
    attn_type="fa3",
)


@torch.no_grad()
def build(name: str, seed: int) -> Transformer:
    torch.manual_seed(seed)
    with torch.device("cuda"):
        model = Transformer(_MODEL_ARGS)
        model.init_weights()
    # parameters only: model.to(bf16) would also cast the complex freqs_cis buffer
    # and silently throw away its imaginary part
    for parameter in model.parameters():
        parameter.data = parameter.data.to(dtype=torch.bfloat16)
    return model


def bf16_cross_entropy_loss(pred: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cross_entropy(
        pred.flatten(0, 1),
        targets.flatten(0, 1),
        reduction="sum",
        ignore_index=IGNORE_INDEX,
    )


def make_forward_fn(
    name: str,
    model: Transformer,
    positions: torch.Tensor,
    compile_mode: str | None,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    if name == "coda":
        def _fn(tokens: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            return model(tokens=tokens, targets=targets, positions=positions)

    elif name == "liger":
        def _fn(tokens: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            return liger_utils.cross_entropy(pred=model(tokens), targets=targets)

    elif name == "torch":
        def _fn(tokens: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            return bf16_cross_entropy_loss(pred=model(tokens), targets=targets)

    else:
        raise NotImplementedError

    if compile_mode is not None:
        compile_kwargs: dict[str, bool | str] = {
            "dynamic": False,
            "fullgraph": True,
        }
        if compile_mode != "default":
            compile_kwargs["mode"] = compile_mode
        fn_maybe_compiled = torch.compile(_fn, **compile_kwargs)
    else:
        fn_maybe_compiled = _fn

    return fn_maybe_compiled


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", choices=("coda", "liger", "torch"), required=True)
    parser.add_argument("--compile", choices=("default", "max-autotune-no-cudagraphs"), default=None)
    parser.add_argument("--trace", type=str, default=None)
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args()

    if args.json is not None:
        assert not os.path.exists(args.json), f"{args.json} exists"

    torch.manual_seed(1)
    shape = (_BATCH, _LENGTH)
    vocab_size = _MODEL_ARGS.vocab_size
    tokens = torch.randint(0, vocab_size, shape, device="cuda")
    targets = torch.randint(0, vocab_size, shape, device="cuda")
    positions = None
    if args.name == "coda":
        # coda takes int32 targets and explicit positions, both made once outside the timed region
        targets = targets.to(dtype=torch.int32)

    model = build(name=args.name, seed=0)
    forward_fn = make_forward_fn(
        name=args.name,
        model=model,
        positions=positions,
        compile_mode=args.compile,
    )

    parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    def forward_backward() -> torch.Tensor:
        loss = forward_fn(tokens=tokens, targets=targets)
        torch.autograd.grad(outputs=loss, inputs=parameters)
        return loss

    if args.trace is not None:
        for _ in range(_NUM_TRACE_WARMUP):
            forward_backward()

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
        ) as profiler:
            for _ in range(_NUM_TRACE_ITERATIONS):
                forward_backward()

        profiler.export_chrome_trace(args.trace)
        print(f"{args.name:<6} trace -> {args.trace}")
        return

    time_ms = bench_utils.do_bench_count(
        forward_backward,
        warmup=_NUM_WARMUP,
        rep=_NUM_ITERATIONS,
    )
    torch.cuda.reset_peak_memory_stats()
    loss = forward_backward()
    memory_gib = torch.cuda.max_memory_allocated() / (2 ** 30)

    ntokens = _BATCH * _LENGTH
    device = torch.cuda.current_device()
    record = {
        "name": args.name,
        "gpu_name": torch.cuda.get_device_name(device),
        "gpu_uuid": str(torch.cuda.get_device_properties(device).uuid),
        "gpu_visible": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
        "batch": _BATCH,
        "length": _LENGTH,
        "tokens": ntokens,
        "vocab": vocab_size,
        "compile": args.compile,
        "inplace_grad": ALLOW_INPLACE_GRAD_OUTPUT if args.name == "coda" else None,
        "time_ms": time_ms,
        "memory_gib": memory_gib,
        "tokens_per_s": ntokens / (time_ms / 1e3),
        "loss_per_token": loss.item() / ntokens,
    }
    print(
        f"{args.name:<6}",
        f"{time_ms:9.3f} ms",
        f"{memory_gib:6.2f} GiB",
        f"{record['tokens_per_s'] / 1e3:8.2f} ktok/s",
        f"loss/tok {record['loss_per_token']:.4f}",
        sep="  ",
    )
    if args.json is not None:
        with open(args.json, "x") as handle:
            handle.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
