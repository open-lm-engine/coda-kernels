import copy
import torch
import inspect
import functools
import dataclasses
from typing import Callable

from quack.autotuner import autotune, Autotuner, AutotuneConfig
from quack.epilogue.frontend import EpiMod
from quack.gemm_config import GemmConfig, get_all_configs
from quack.gemm_interface import prune_invalid_gemm_configs

from coda.core.ops.constants import AUTOTUNE_CACHE_RESULTS


# https://github.com/Dao-AILab/quack/blob/v0.6.5/quack/epilogue/ops.py#L930
GATED_TILE_N_MULTIPLE_OF = 32


def _extend_configs(
    configs: list[GemmConfig],
    fn: Callable[[GemmConfig], GemmConfig],
) -> list[GemmConfig]:
    assert isinstance(configs, list)
    assert len(configs) == len(set(configs))
    configs_extended = copy.deepcopy(configs)
    for config in configs:
        if config.device_capacity != 9:
            continue
        _config = fn(config)
        if _config in configs_extended:
            continue
        configs_extended.append(_config)
    return configs_extended


# https://github.com/Dao-AILab/quack/blob/v0.6.4/quack/gemm_config.py#L127
def _cooperative_compatible(config: GemmConfig) -> bool:
    return config.device_capacity != 9 or config.pingpong or config.tile_m != 192


GEMM_CONFIGS = get_all_configs()
GEMM_CONFIGS = _extend_configs(GEMM_CONFIGS, lambda config: dataclasses.replace(config, cluster_m=1, cluster_n=1))
GEMM_CONFIGS = _extend_configs(GEMM_CONFIGS, lambda config: dataclasses.replace(config, cluster_m=1, cluster_n=1, pingpong=False))
GEMM_CONFIGS = _extend_configs(GEMM_CONFIGS, lambda config: dataclasses.replace(config, is_dynamic_persistent=True))
GEMM_CONFIGS = [config for config in GEMM_CONFIGS if _cooperative_compatible(config)]


def gated_prune_fn(config: GemmConfig, named_args: dict) -> bool:
    return config.tile_n % GATED_TILE_N_MULTIPLE_OF == 0


def prune_gemm_configs(
    configs: list[AutotuneConfig],
    named_args: dict,
    prune_fn: Callable[[GemmConfig, dict], bool] | None,
    **kwargs,
) -> list[AutotuneConfig]:
    configs = prune_invalid_gemm_configs(
        configs=configs,
        named_args=named_args,
        **kwargs,
    )
    configs = [conf for conf in configs if not conf.kwargs["config"].swap_ab]
    # an op-specific rule over the config and the call's arguments: it returns whether to keep the config
    if prune_fn is not None:
        # a positional call names its arguments in named_args, a keyword call leaves them in kwargs
        configs = [
            conf for conf in configs
            if prune_fn(
                conf.kwargs["config"],
                named_args | kwargs,
            )
        ]
    return configs


def _kernel_op(
    name: str,
    mutates_args: tuple[str, ...],
) -> Callable[[Callable], Callable]:

    def _wrap(fn: Callable) -> Callable:

        @torch.library.custom_op(
            name,
            mutates_args=mutates_args,
            device_types="cuda",
        )
        @functools.wraps(fn)
        def op(*args, **kwargs) -> None:
            return fn(*args, **kwargs)

        @torch.library.register_fake(name)
        def _(*args, **kwargs) -> None:
            pass

        return op

    return _wrap


def epilogue_launch(
    epi_fn: EpiMod,
    A: torch.Tensor,
    B: torch.Tensor,
    D: torch.Tensor | None,
    C: torch.Tensor | None = None,
    *,
    epi_args: dict,
    config: GemmConfig,
    add_to_output: bool = False,
    fp8_fast_accum: bool = False,
) -> None:
    if config.is_dynamic_persistent:
        semaphore = torch.zeros(1, dtype=torch.int32, device=A.device)
    else:
        semaphore = None
    if fp8_fast_accum:
        post_init_attrs = (("fp8_slow_accum", False),)
    else:
        post_init_attrs = ()
    _ = epi_fn.gemm(
        A=A,
        B=B,
        D=D,
        C=C,
        epi_args=epi_args,
        tile_M=config.tile_m,
        tile_N=config.tile_n,
        tile_K=config.tile_k,
        cluster_M=config.cluster_m,
        cluster_N=config.cluster_n,
        pingpong=config.pingpong,
        is_dynamic_persistent=config.is_dynamic_persistent,
        max_swizzle_size=config.max_swizzle_size,
        tile_count_semaphore=semaphore,
        use_tma_gather=config.use_tma_gather,
        swap_ab=config.swap_ab,
        split_k=config.split_k,
        add_to_output=add_to_output,
        post_init_attrs=post_init_attrs,
    )


def _make_autotuner(fn: Callable, tunable: str, **kwargs) -> Autotuner:
    # the custom-op dispatcher passes positionally, and the tuner reads `key=` names from kwargs only
    assert "key" not in kwargs, f"{fn.__name__}: key= would be dropped, split the function instead"
    tuner = autotune(**kwargs)(fn)
    # callers pass everything except `tunable`; the tuner supplies it, so the tuner's signature is `fn`'s without it
    signature = inspect.signature(fn)
    assert list(signature.parameters.keys())[-1] == tunable, f"{fn.__name__}: `{tunable}` must be the last parameter"
    tuner.__signature__ = signature.replace(parameters=list(signature.parameters.values())[:-1])
    return tuner


def backend_autotune() -> Callable[[Callable], Autotuner]:

    def _wrap(fn: Callable) -> Autotuner:
        return _make_autotuner(
            fn,
            tunable="backend",
            configs=[
                AutotuneConfig(backend="quack"),
                AutotuneConfig(backend="cublas"),
            ],
            cache_results=AUTOTUNE_CACHE_RESULTS,
        )

    return _wrap


def epilogue_autotune(
    configs: list[GemmConfig] | None = None,
    prune_fn: Callable[[GemmConfig, dict], bool] | None = None,
) -> Callable[[Callable], Autotuner]:
    if configs is None:
        configs = GEMM_CONFIGS

    _prune_fn = functools.partial(
        prune_gemm_configs,
        prune_fn=prune_fn,
    )

    def _wrap(fn: Callable) -> Autotuner:
        return _make_autotuner(
            fn,
            tunable="config",
            configs=[AutotuneConfig(config=c) for c in configs],
            prune_configs_by={"early_config_prune": _prune_fn},
            cache_results=AUTOTUNE_CACHE_RESULTS,
        )

    return _wrap
