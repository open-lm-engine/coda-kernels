import torch
from typing import NamedTuple
from collections import defaultdict

from benchmarks.utils.cases import (
    Case,
    ImplementationFwdBwd,
    make_callables,
    select_oracle_rows,
)


class Precision(NamedTuple):
    # e.g., {"coda/eager": {"out": float}, "torch/eager": {"out": float}}
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
    # CODA's eager call runs first: its output dtypes are the reference
    # every call is compared in, and they set the least possible error
    callables = make_callables(case=case)
    coda_eager = "coda/eager"
    assert next(iter(callables)) == "coda"
    assert next(iter(callables["coda"])) == coda_eager

    dtypes_reference = {}
    errors = defaultdict(dict)
    errors_minimum = {}
    for impl_name, impl_variants in callables.items():
        # each implementation runs its own oracle; outputs are not shared, as
        # they are freed after each implementation to save memory
        oracle_outputs = case.oracles[impl_name].run()
        for variant, call in impl_variants.items():
            # read the second call, to catch one that overwrites its inputs
            if call.setup_fn is not None:
                call.setup_fn()

            call.run()

            if call.setup_fn is not None:
                call.setup_fn()

            outputs = call.run()
            # the outputs as the case names them: a lone tensor as a tuple,
            # and a training step's through its postprocessing_fn
            if isinstance(outputs, torch.Tensor):
                outputs = (outputs,)
            else:
                outputs = tuple(outputs)

            implementation = case.implementations[impl_name]
            if (
                isinstance(implementation, ImplementationFwdBwd)
                and implementation.postprocessing_fn is not None
            ):
                outputs = implementation.postprocessing_fn(outputs)

            for output_name, output, oracle_output in zip(
                case.outputs,
                outputs,
                oracle_outputs,
                strict=True,
            ):
                # CODA's eager call records each output's reference dtype, and
                # its least possible error: the exact answer's own error once
                # rounded to that dtype
                if variant == coda_eager:
                    dtypes_reference[output_name] = output.dtype
                    errors_minimum[output_name] = relative_l2_error(
                        output=oracle_output.to(dtype=output.dtype),
                        oracle=oracle_output,
                    )

                # CODA's dtype is the reference: an output with more bits is
                # rounded to it, one with fewer is kept as it is; only floating
                # point outputs ever differ from it
                dtype_reference = dtypes_reference[output_name]
                if output.dtype.itemsize > dtype_reference.itemsize:
                    assert output.dtype.is_floating_point
                    assert dtype_reference.is_floating_point
                    output = output.to(dtype=dtype_reference)
                elif output.dtype.itemsize == dtype_reference.itemsize:
                    assert output.dtype == dtype_reference
                else:
                    assert output.dtype.is_floating_point
                    assert dtype_reference.is_floating_point

                # along M, keep what the oracle answered (a scalar has no M)
                if oracle_output.ndim > 0:
                    output = select_oracle_rows(
                        output=output,
                        m=oracle_output.shape[0],
                    )

                errors[variant][output_name] = relative_l2_error(
                    output=output,
                    oracle=oracle_output,
                )

            # free each call's outputs; all at once do not fit at scale
            del outputs, output

        # free its fp64 outputs before the next oracle runs, to save memory
        del oracle_outputs, oracle_output

    return Precision(
        errors=errors,
        errors_minimum=errors_minimum,
    )
