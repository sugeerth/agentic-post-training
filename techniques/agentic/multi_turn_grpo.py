"""Multi-turn GRPO for agent post-training.

Standard GRPO samples G responses to one prompt and normalizes the reward
across the group. Multi-turn GRPO does the same, but *the response is a
whole trajectory* — a sequence of (assistant, tool) turns terminated by a
task-success signal.

Key differences from single-turn GRPO:

  1. Advantage is defined over trajectories, not tokens. Each turn inherits
     its trajectory's group-normalized advantage; per-turn losses are
     weighted by the trajectory advantage.
  2. Reward is sparse (terminal task success ∈ {0, 1}) and optionally
     shaped by a process reward model.
  3. KL is measured to a *reference agent policy*, not a base LM, so we
     don't drag the model back toward non-agentic completions.

This module is a lightweight, no-GPU simulation of that update so the
supervisor pipeline runs anywhere. Torch path lives in
`techniques/grpo.py::GRPOTechnique` — the real training loop calls into it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class MultiTurnGRPOConfig:
    group_size: int = 8              # G — trajectories per task
    kl_coef: float = 0.05            # smaller than single-turn: agent drift is more dangerous
    clip_ratio: float = 0.2
    max_turns: int = 12              # trajectories longer than this are truncated
    reward_shaping: str = "process"  # "process" | "outcome"


class MultiTurnGRPO:
    """Simulation-only step (loss decays deterministically).

    The real update lives in `techniques/grpo.py::GRPOTechnique._step_real`
    which takes tensor `log_probs / ref_log_probs` and returns the loss.
    """

    name = "multi_turn_grpo"

    def __init__(self, config: MultiTurnGRPOConfig | None = None):
        self.config = config or MultiTurnGRPOConfig()
        self._it = 0

    def step(self, iteration: int | None = None) -> dict[str, float]:
        self._it = iteration if iteration is not None else self._it + 1
        loss = 1.6 * math.exp(-0.32 * self._it) + 0.26
        reward = min(0.93, 0.34 + 0.20 * self._it)
        kl = max(0.01, 0.16 - 0.025 * self._it)
        adv_std = max(0.15, 0.55 - 0.09 * self._it)
        return {
            "loss": round(loss, 4),
            "reward": round(reward, 4),
            "kl_divergence": round(kl, 4),
            "group_reward_std": round(adv_std, 4),
            "advantage_mean": round(0.08 + 0.04 * self._it, 4),
        }
