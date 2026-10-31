import torch
from typing import NamedTuple
from collections import defaultdict

from benchmarks.utils.cases import Case, make_callables, select_output_by_m


class Precision(NamedTuple):
    # e.g., {"coda": {"out": float}, "torch/eager": {"out": float}}
    errors: dict[str, dict[str, float]]
    # least possible error per output, e.g., {"out": float}
    errors_minimum: dict[str, float]


def relative_l2_error(
    output: torch.Tensor,
    oracle: torch.Tensor,
) -> float:
    # ||output - oracle|| / ||oracle||
    assert oracle.dtype == torch.float64
    difference = output.to(dtype=torch.float64) - oracle
    error = torch.linalg.vector_norm(difference)
    size = torch.linalg.vector_norm(oracle)
    return (error / size).item()


def precision(case: Case) -> Precision:
    # CODA first: its output dtypes apply to every call
    callables = make_callables(case=case)
    assert next(iter(callables)) == "coda"
    oracle_outputs_dict = {}
    dtypes = {}
    errors = defaultdict(dict)
    errors_minimum = {}
    for impl_name, variants in callables.items():
        oracle_impl = case.oracles[impl_name]

        # run each oracle once; implementations may share one
        if id(oracle_impl) not in oracle_outputs_dict:
            oracle_outputs_dict[id(oracle_impl)] = oracle_impl.fn(**oracle_impl.inputs)

        oracle_outputs = oracle_outputs_dict[id(oracle_impl)]
        for variant, call in variants.items():
            # read the second call, to catch one that overwrites its inputs
            call()
            outputs = call()

            if isinstance(outputs, torch.Tensor):
                outputs = (outputs,)

            for output_name, output, oracle_output in zip(
                case.outputs, outputs, oracle_outputs, strict=True,
            ):
                # CODA sets each output's dtype and least possible error
                if variant == "coda":
                    dtypes[output_name] = output.dtype
                    errors_minimum[output_name] = relative_l2_error(
                        output=oracle_output.to(dtype=output.dtype),
                        oracle=oracle_output,
                    )

                # an output wider than CODA's (fp32 or fp64 where CODA stores 16
                # bits) is rounded to CODA's dtype
                if (
                    output.dtype in {torch.float32, torch.float64}
                    and dtypes[output_name] in {torch.bfloat16, torch.float16}
                ):
                    output = output.to(dtype=dtypes[output_name])
                else:
                    assert output.dtype == dtypes[output_name]

                # along M, keep what the oracle answered (a scalar has no M)
                if oracle_output.ndim > 0:
                    output = select_output_by_m(
                        output=output,
                        m=oracle_output.shape[0],
                    )

                errors[variant][output_name] = relative_l2_error(
                    output=output,
                    oracle=oracle_output,
                )

            # free each call's outputs; all at once do not fit at scale
            del outputs, output

    return Precision(
        errors=errors,
        errors_minimum=errors_minimum,
    )
