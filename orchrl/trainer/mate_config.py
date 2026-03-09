from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from omegaconf import OmegaConf


def _to_plain_dict(mate_cfg: Any) -> dict[str, Any]:
    if OmegaConf.is_config(mate_cfg):
        resolved = OmegaConf.to_container(mate_cfg, resolve=True)
        if not isinstance(resolved, dict):
            raise TypeError("mate config must resolve to a dict")
        return resolved
    if isinstance(mate_cfg, Mapping):
        return dict(mate_cfg)
    raise TypeError("mate config must be a dict")


def validate_mate_config(mate_cfg: Any, agent_policy_mapping: Mapping[str, str] | None) -> dict[str, Any]:
    config_dict = _to_plain_dict(mate_cfg)
    roles = config_dict.get("roles")
    role_policy_mapping = config_dict.get("role_policy_mapping")

    if not isinstance(roles, list) or not roles:
        raise ValueError("mate.roles must be a non-empty list")
    if not isinstance(role_policy_mapping, dict) or not role_policy_mapping:
        raise ValueError("mate.role_policy_mapping must be a non-empty dict")

    known_policies = set((agent_policy_mapping or {}).values())
    for role in roles:
        if role not in role_policy_mapping:
            raise ValueError(f"mate.role_policy_mapping missing role '{role}'")
        policy_name = role_policy_mapping[role]
        if not isinstance(policy_name, str) or not policy_name:
            raise ValueError(f"mate.role_policy_mapping for role '{role}' must be a non-empty string")
        if policy_name not in known_policies:
            raise ValueError(f"unknown policy in mate.role_policy_mapping: {policy_name}")

    return config_dict
