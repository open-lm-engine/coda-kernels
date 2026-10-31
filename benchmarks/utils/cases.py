import torch
import functools
from typing import Callable, NamedTuple


COMPILE_OPTIONS = {
    "default": {},
    "maa": {"mode": "max-autotune-no-cudagraphs"},
}


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


def calls(
    case: Case,
    modes: tuple[str, ...],
) -> dict[str, dict[str, Callable]]:
    # each implementation's calls, by its name and then by label: CODA's
    # as written, torch's eager or compiled once per mode
    result = {}
    for name, implementation in case.implementations.items():
        result[name] = {}

        if name == "coda":
            result[name][name] = functools.partial(
                implementation.fn,
                **implementation.inputs,
            )
        else:
            for mode in modes:
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
