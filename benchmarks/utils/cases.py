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
        return output.index_select(
            dim=0,
            index=indices,
        )


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


def make_callables(case: Case) -> dict[str, dict[str, Callable]]:
    # each implementation's callables, by its name and then by label:
    # CODA's as written, torch's eager or compiled once per mode
    result = defaultdict(dict)
    for name, implementation in case.implementations.items():
        if name == "coda":
            result[name][name] = functools.partial(
                implementation.fn,
                **implementation.inputs,
            )
        else:
            for mode in MODES:
                result[name][f"{name}/{mode}"] = functools.partial(
                    compiled(
                        fn=implementation.fn,
                        mode=mode,
                    ),
                    **implementation.inputs,
                )
    return result


def oracle_outputs(oracle: Implementation) -> tuple:
    # the oracle over fp64 copies of its floating-point inputs, the rest
    # (integer tensors, scalars, layouts) as they are
    arguments = {}
    for name, value in oracle.inputs.items():
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            arguments[name] = value.to(dtype=torch.float64)
        else:
            arguments[name] = value
    outputs = oracle.fn(**arguments)

    if isinstance(outputs, torch.Tensor):
        return (outputs,)
    else:
        return tuple(outputs)


def precision(case: Case) -> dict[str, dict[str, float]]:
    # every call's rel-L2 by output, against its implementation's oracle
    errors = defaultdict(dict)
    for name, variants in make_callables(case=case).items():
        oracle_values = oracle_outputs(oracle=case.oracles[name])
        for label, call in variants.items():
            outputs = call()

            if isinstance(outputs, torch.Tensor):
                values = (outputs,)
            else:
                values = tuple(outputs)

            for output, value, oracle_value in zip(case.outputs, values, oracle_values, strict=True):
                # rel-L2: the output's error over the size of the oracle's
                assert oracle_value.dtype == torch.float64
                difference = value.to(dtype=torch.float64) - oracle_value
                error = torch.linalg.vector_norm(difference)
                size = torch.linalg.vector_norm(oracle_value)
                errors[label][output] = (error / size).item()
    return errors
