"""Advantage estimation primitives shared between PPO and GRPO."""

from __future__ import annotations

from typing import Any


def compute_group_advantages(rewards: Any, eps: float = 1e-8) -> Any:
    """Group-relative advantage: (r_i − mean(r)) / std(r).

    Accepts a torch tensor or a Python sequence. Returns the same type the
    caller passed in, so techniques can ignore the dispatch.
    """
    try:
        import torch
    except ImportError:
        torch = None

    if torch is not None and isinstance(rewards, torch.Tensor):
        mean = rewards.mean()
        std = rewards.std() + eps
        return (rewards - mean) / std

    # Pure-Python path. Kept so the simulator code path doesn't require torch.
    seq = list(rewards)
    n = len(seq)
    mean = sum(seq) / n
    var = sum((r - mean) ** 2 for r in seq) / n
    std = var**0.5 + eps
    return [(r - mean) / std for r in seq]


def compute_gae(
    rewards: Any,
    values: Any,
    dones: Any | None = None,
    *,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Any:
    """Generalized advantage estimation. Used by PPO; not by GRPO.

    Implemented for tensor inputs only — GAE is RL-tensor-heavy enough that a
    Python fallback would be a foot-gun (and the legacy `ppo.py` already
    short-circuits to a decay formula in simulation mode).
    """
    import torch

    rewards = torch.as_tensor(rewards, dtype=torch.float32)
    values = torch.as_tensor(values, dtype=torch.float32)
    if dones is None:
        dones = torch.zeros_like(rewards)
    else:
        dones = torch.as_tensor(dones, dtype=torch.float32)

    advantages = torch.zeros_like(rewards)
    gae: torch.Tensor = torch.tensor(0.0)
    next_value: torch.Tensor = torch.tensor(0.0)
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages[t] = gae
        next_value = values[t]
    return advantages
