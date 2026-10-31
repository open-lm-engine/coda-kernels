import torch
import functools
from collections import defaultdict
from typing import Callable, NamedTuple

# torch.compile's options by mode, always fullgraph and static
COMPILE_OPTIONS = {
    "default": {},
    "maa": {"mode": "max-autotune-no-cudagraphs"},
}
# every mode torch runs in: eager, then each compile mode
MODES = ("eager", "default", "maa")


class Implementation(NamedTuple):
    # a function and the exact inputs it is called with
    fn: Callable
    inputs: dict


class Case(NamedTuple):
    # CODA's implementation and torch's, by name
    implementations: dict[str, Implementation]
    # the oracle that answers each of them, by the same name: the same
    # computation in the same layout, run on fp64 copies of its inputs
    oracles: dict[str, Implementation]
    # the names of the outputs every function returns, in order
    outputs: tuple[str, ...]


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
