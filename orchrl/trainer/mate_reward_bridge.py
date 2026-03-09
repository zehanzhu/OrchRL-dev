from __future__ import annotations

from importlib import import_module
from typing import Any, Callable

from trajectory import FunctionRewardProvider


def _import_callable(import_path: str) -> Callable[..., Any]:
    if ":" in import_path:
        module_name, attr_name = import_path.split(":", 1)
    else:
        module_name, _, attr_name = import_path.rpartition(".")
    if not module_name or not attr_name:
        raise ValueError(f"invalid callable import path: {import_path}")
    module = import_module(module_name)
    func = getattr(module, attr_name)
    if not callable(func):
        raise TypeError(f"imported object is not callable: {import_path}")
    return func


def build_reward_provider(reward_cfg: dict[str, Any]):
    provider_path = reward_cfg.get("provider")
    if not isinstance(provider_path, str) or not provider_path:
        raise ValueError("mate.reward.provider must be a non-empty import path")
    func = _import_callable(provider_path)
    return FunctionRewardProvider(func)
