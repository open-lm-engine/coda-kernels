import copy
import torch
import functools
import dataclasses
from typing import Callable

from quack.autotuner import autotune, AutotuneConfig
from quack.epilogue.frontend import EpiMod
from quack.gemm_config import GemmConfig, get_all_configs
from quack.gemm_interface import prune_invalid_gemm_configs

from coda.core.ops.constants import AUTOTUNE_CACHE_RESULTS


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


def prune_gemm_configs(configs: list[AutotuneConfig], named_args: dict, **kwargs) -> list[AutotuneConfig]:
    configs = prune_invalid_gemm_configs(
        configs=configs,
        named_args=named_args,
        **kwargs,
    )
    configs = [conf for conf in configs if not conf.kwargs["config"].swap_ab]
    return configs


def prune_gated_gemm_configs(configs: list[AutotuneConfig], named_args: dict, **kwargs) -> list[AutotuneConfig]:
    configs = prune_gemm_configs(
        configs=configs,
        named_args=named_args,
        **kwargs,
    )
    # https://github.com/Dao-AILab/quack/blob/v0.6.4/quack/epilogue/ops.py#L928
    configs = [conf for conf in configs if conf.kwargs["config"].tile_n % 32 == 0]
    return configs


def _kernel_op(
    name: str,
    mutates_args: tuple[str, ...],
) -> Callable[[Callable], Callable]:

    def decorator(fn: Callable) -> Callable:

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

    return decorator
