"""Proximal Policy Optimization (PPO) for RLHF.

Priority: 1 (Core technique)
Paper: "Proximal Policy Optimization Algorithms" (Schulman et al., 2017)

PPO is the backbone of RLHF-based alignment. It optimizes a clipped surrogate
objective to update the policy while preventing destructive large updates.
Used by OpenAI for InstructGPT and ChatGPT alignment.

When to use:
- You have a trained reward model
- You want fine-grained control over the alignment process
- You need stable training with bounded policy updates

Pros:
- Well-understood, extensively validated
- Stable training with clipped objective
- Works with any reward signal

Cons:
- Requires a separate reward model
- Memory intensive (policy + value + reward + reference models)
- Sensitive to hyperparameters (clip ratio, KL coefficient)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from techniques.base_technique import BaseTechnique, TechniqueConfig

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class PPOConfig(TechniqueConfig):
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    gae_lambda: float = 0.95
    gamma: float = 0.99
    kl_penalty_coef: float = 0.1
    target_kl: float = 0.02
    num_mini_batches: int = 4
    ppo_epochs: int = 4


class PPO(BaseTechnique):
    """Proximal Policy Optimization for RLHF alignment."""

    name = "ppo"
    description = "Proximal Policy Optimization — stable RL-based alignment with clipped objectives"
    paper_reference = "Schulman et al., 2017 — Proximal Policy Optimization Algorithms"
    priority = 1
    recommended_for = ["RLHF alignment", "instruction following", "safety training"]
    pros = ["Stable training", "Well-understood", "Works with any reward signal"]
    cons = ["Needs reward model", "Memory intensive (4 models)", "Hyperparameter sensitive"]

    def __init__(self, config: PPOConfig | None = None):
        super().__init__(config or PPOConfig())

    def compute_loss(self, **kwargs) -> Any:
        """Compute PPO clipped surrogate loss.

        Loss = -min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t)
            + value_coef * value_loss
            - entropy_coef * entropy
        """
        cfg = self.config

        if HAS_TORCH and "log_probs" in kwargs:
            log_probs = kwargs["log_probs"]
            old_log_probs = kwargs["old_log_probs"]
            advantages = kwargs["advantages"]
            returns = kwargs["returns"]
            values = kwargs["values"]

            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(values, returns)
            entropy = -(log_probs * torch.exp(log_probs)).mean()

            total_loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy

            self.metrics.update({
                "policy_loss": policy_loss.item(),
                "value_loss": value_loss.item(),
                "entropy": entropy.item(),
                "clip_fraction": ((ratio - 1.0).abs() > cfg.clip_ratio).float().mean().item(),
                "approx_kl": (old_log_probs - log_probs).mean().item(),
            })
            return total_loss

        # Simulation mode
        epoch = kwargs.get("epoch", self.step_count)
        loss = 2.0 * math.exp(-0.3 * epoch) + 0.4
        self.metrics.update({
            "loss": loss,
            "policy_loss": loss * 0.6,
            "value_loss": loss * 0.3,
            "entropy": max(0.01, 0.5 - 0.05 * epoch),
            "clip_fraction": max(0.05, 0.25 - 0.04 * epoch),
            "reward": min(0.95, 0.3 + 0.2 * epoch),
        })
        return loss

    def compute_gae(self, rewards, values, dones, gamma=0.99, lam=0.95):
        """Compute Generalized Advantage Estimation."""
        if not HAS_TORCH:
            return None, None

        advantages = torch.zeros_like(rewards)
        last_gae = 0

        for t in reversed(range(len(rewards))):
            next_value = values[t + 1] if t + 1 < len(values) else 0
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + gamma * lam * (1 - dones[t]) * last_gae

        returns = advantages + values[:len(advantages)]
        return advantages, returns
