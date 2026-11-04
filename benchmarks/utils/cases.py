import torch
import functools
from collections import defaultdict
from typing import Callable, NamedTuple
from torch._inductor import list_mode_options
from benchmarks.backends import torchtitan_llama3

# the options the extra compile settings add, TorchTitan's
EXTRA_COMPILE_OPTIONS = torchtitan_llama3.EXTRA_COMPILE_OPTIONS
# torch.compile's (mode, options) by label, always fullgraph and static:
# eager, then each mode plain and with EXTRA_COMPILE_OPTIONS
COMPILE_SETTINGS = {
    "eager": ("eager", {}),
    "default": ("default", {}),
    "default-extra": ("default", EXTRA_COMPILE_OPTIONS),
    "maa": ("max-autotune-no-cudagraphs", {}),
    "maa-extra": ("max-autotune-no-cudagraphs", EXTRA_COMPILE_OPTIONS),
}
# the dtype an oracle computes in
ORACLE_DTYPE = torch.float64
# rows per oracle chunk: all of M in fp64 does not fit
ORACLE_CHUNK_SIZE = 1024
# rows an oracle checks when M has more, and their seed
ORACLE_NUM_ROWS = 1024
ORACLE_ROWS_SEED = 0


class Implementation(NamedTuple):
    fn: Callable
    inputs: dict
    # runs before each call, untimed; None if there is nothing to reset
    setup_fn: Callable | None = None

    def run(self) -> object:
        return self.fn(**self.inputs)


class ImplementationFwdBwd(NamedTuple):
    # the forward; make_callables builds the training step from it
    fn: Callable
    inputs: dict
    # inputs to differentiate
    leaves: tuple[str, ...]
    # one per output
    cotangents: tuple[str, ...]
    # aligns outputs with torch's before comparing
    postprocessing_fn: Callable | None = None


class Case(NamedTuple):
    implementations: dict[str, Implementation | ImplementationFwdBwd]
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


def _make_leaves(
    inputs: dict,
    leaves: tuple[str, ...],
) -> dict:
    # a copy of inputs whose leaves are fresh tensors that require grad,
    # sharing the original's memory
    made = inputs.copy()
    for name in leaves:
        made[name] = (
            inputs[name]
            .detach()
            .requires_grad_(True)
        )

    return made


def _chunk_inputs(
    inputs: dict,
    mapped: list[str],
    leaves: tuple[str, ...],
    start: int,
    stop: int,
) -> dict:
    arguments = {}
    for name, value in inputs.items():
        if name in mapped:
            assert 0 <= start < stop <= value.shape[0]
            value = value[start:stop, ...]

        if isinstance(value, torch.Tensor) and value.is_floating_point():
            arguments[name] = value.to(dtype=ORACLE_DTYPE)
        else:
            arguments[name] = value

    return _make_leaves(
        inputs=arguments,
        leaves=leaves,
    )


def make_oracle_fn(
    fn: Callable,
    in_dims: dict[str, int | None],
    out_dims: tuple[int | None, ...],
    reduce_dims: tuple[int | None, ...],
    leaves: tuple[str, ...] = (),
    chunk_size: int = ORACLE_CHUNK_SIZE,
) -> Callable:
    # runs fn in fp64 on chunk_size rows of M at a time; each output is
    # summed over the chunks (reduce_dims 0), kept per row (out_dims 0),
    # or taken from the first chunk (out_dims None)
    for dim in (*in_dims.values(), *out_dims, *reduce_dims):
        # each dim is 0 (split along M) or None (whole); only a
        # split output can be summed
        assert dim in (0, None)

    assert len(out_dims) == len(reduce_dims)
    for out_dim, reduce_dim in zip(out_dims, reduce_dims, strict=True):
        if reduce_dim == 0:
            assert out_dim == 0

    mapped = []
    for name, dim in in_dims.items():
        if dim == 0:
            mapped.append(name)

    def get_mapped_size(inputs: dict) -> int:
        # M; every tensor input is listed, and the mapped ones agree on it
        tensor_names = []
        for name, value in inputs.items():
            if isinstance(value, torch.Tensor):
                tensor_names.append(name)

        assert set(tensor_names) == set(in_dims)
        sizes = set()
        for name in mapped:
            sizes.add(inputs[name].shape[0])

        assert len(sizes) <= 1
        if len(sizes) == 1:
            return sizes.pop()
        else:
            return 1

    @functools.wraps(fn)
    def oracle_fn(**kwargs: object) -> tuple:
        count = get_mapped_size(inputs=kwargs)
        rows = oracle_rows(count=count)
        # by output position: each chunk's checked rows (out_dims 0), the
        # sum over chunks (reduce_dims 0), or the first chunk's value
        outputs_chunked = defaultdict(list)
        outputs_reduced = {}
        outputs_unmapped = {}
        for start in range(0, count, chunk_size):
            # the last chunk stops at M
            stop = min(start + chunk_size, count)
            kwargs_chunked = _chunk_inputs(
                inputs=kwargs,
                mapped=mapped,
                leaves=leaves,
                start=start,
                stop=stop,
            )
            outputs = fn(**kwargs_chunked)
            if isinstance(outputs, torch.Tensor):
                outputs = (outputs,)

            assert len(outputs) == len(out_dims)
            # the checked rows in this chunk, counted from its start
            if rows is None:
                chunk_rows = None
            else:
                inside = (rows >= start) & (rows < stop)
                chunk_rows = rows[inside] - start

            for position, output in enumerate(outputs):
                if reduce_dims[position] == 0:
                    if position in outputs_reduced:
                        outputs_reduced[position] = outputs_reduced[position] + output
                    else:
                        outputs_reduced[position] = output
                elif out_dims[position] == 0:
                    if chunk_rows is None:
                        outputs_chunked[position].append(output)
                    else:
                        outputs_chunked[position].append(
                            output.index_select(
                                dim=0,
                                index=chunk_rows,
                            ),
                        )
                elif position not in outputs_unmapped:
                    # no M, so every chunk gives the same value
                    outputs_unmapped[position] = output

            # free this chunk before the next
            del kwargs_chunked, outputs, output

        oracle_outputs = []
        for position in range(len(out_dims)):
            if position in outputs_reduced:
                oracle_outputs.append(outputs_reduced[position])
            elif position in outputs_chunked:
                oracle_outputs.append(torch.cat(outputs_chunked[position], dim=0))
            else:
                oracle_outputs.append(outputs_unmapped[position])

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
    fn: Callable,
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
            fn=fn,
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
    name: str,
    options: dict[str, object],
) -> Callable:
    # one name per call, so no call reuses another's graph
    if mode == "eager":
        return fn
    elif len(options) == 0:
        return torch.compile(
            fn,
            fullgraph=True,
            dynamic=False,
            name=name,
            mode=mode,
        )
    else:
        # torch.compile takes a mode or options, not both: the mode as its
        # options, then these
        compile_options = list_mode_options(
            mode=mode,
            dynamic=False,
        ).copy()
        compile_options.update(options)
        return torch.compile(
            fn,
            fullgraph=True,
            dynamic=False,
            name=name,
            options=compile_options,
        )


def _make_op_callables(
    name: str,
    implementation: Implementation,
    setup_fn: Callable | None,
) -> dict[str, Implementation]:
    # one call per compile setting
    callables = {}
    for setting_label, (mode, options) in COMPILE_SETTINGS.items():
        label = f"{name}/{setting_label}"
        callables[label] = Implementation(
            fn=compiled(
                fn=implementation.fn,
                mode=mode,
                name=label,
                options=options,
            ),
            inputs=implementation.inputs,
            setup_fn=setup_fn,
        )

    return callables


def _make_fwdbwd_callables(
    name: str,
    implementation: ImplementationFwdBwd,
) -> dict[str, Implementation]:
    # per compile setting, the whole step and, when compiled, the forward alone
    inputs = _make_leaves(
        inputs=implementation.inputs,
        leaves=implementation.leaves,
    )
    # a backward can overwrite its cotangents, so the calls get copies and
    # setup_fn resets them before each call
    for cotangent in implementation.cotangents:
        inputs[cotangent] = (
            implementation.inputs[cotangent]
            .detach()
            .clone()
        )

    def setup_fn() -> None:
        for cotangent in implementation.cotangents:
            inputs[cotangent].copy_(implementation.inputs[cotangent])

    step_fn = make_step_fn(
        fn=implementation.fn,
        leaves=implementation.leaves,
        cotangents=implementation.cotangents,
    )
    # forward and backward compiled together, as one graph; training usually
    # compiles only the forward, but this is often faster for torch
    callables = _make_op_callables(
        name=name,
        implementation=Implementation(
            fn=step_fn,
            inputs=inputs,
        ),
        setup_fn=setup_fn,
    )

    # the standard way: torch.compile on the forward, AOTAutograd for the backward
    for setting_label, (mode, options) in COMPILE_SETTINGS.items():
        if mode != "eager":
            forward_label = f"{name}/{setting_label}/forward"
            forward = compiled(
                fn=implementation.fn,
                mode=mode,
                name=forward_label,
                options=options,
            )
            callables[forward_label] = Implementation(
                fn=make_step_fn(
                    fn=forward,
                    leaves=implementation.leaves,
                    cotangents=implementation.cotangents,
                ),
                inputs=inputs,
                setup_fn=setup_fn,
            )

    return callables


def make_callables(case: Case) -> dict[str, dict[str, Implementation]]:
    # every implementation's calls, by label
    callables = {}
    for name, implementation in case.implementations.items():
        if isinstance(implementation, ImplementationFwdBwd):
            callables[name] = _make_fwdbwd_callables(
                name=name,
                implementation=implementation,
            )
        else:
            callables[name] = _make_op_callables(
                name=name,
                implementation=implementation,
                setup_fn=None,
            )

    return callables
