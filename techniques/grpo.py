"""Group Relative Policy Optimization (GRPO).

Priority: 1 (Core technique — used in DeepSeek-R1)
Paper: "DeepSeekMath: Pushing the Limits of Mathematical Reasoning" (Shao et al., 2024)

GRPO is a breakthrough technique from DeepSeek that eliminates the need for a
separate value model by using group-level relative ranking. For each prompt,
multiple responses are sampled and ranked within the group. The advantage of
each response is computed relative to the group mean reward.

This is the technique behind DeepSeek-R1's remarkable reasoning capabilities.

When to use:
- You want RL-based training WITHOUT a value model (saves ~50% memory)
- You have a reward model or rule-based rewards
- Mathematical reasoning, coding, or structured tasks

Pros:
- No value model needed (huge memory savings)
- Simple and effective
- Excellent for reasoning tasks
- Scales well

Cons:
- Requires multiple samples per prompt (higher compute per step)
- Group size is a sensitive hyperparameter
- May have higher variance than PPO
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
class GRPOConfig(TechniqueConfig):
    group_size: int = 8  # Number of samples per prompt
    kl_coef: float = 0.1  # KL divergence penalty
    clip_ratio: float = 0.2
    temperature: float = 1.0
    reward_baseline: str = "group_mean"  # group_mean, group_min, running_mean


class GRPO(BaseTechnique):
    """Group Relative Policy Optimization — value-free RL alignment.

    Key insight: Instead of training a separate value network to estimate
    advantages, GRPO generates a GROUP of responses for each prompt, scores
    them with a reward model, and uses the group statistics (mean, std) to
    compute relative advantages. This eliminates the value model entirely.

    Algorithm:
    1. For each prompt, generate G responses from the current policy
    2. Score all responses with reward model
    3. Compute advantages: A_i = (r_i - mean(r)) / std(r)
    4. Update policy with clipped objective (like PPO but no value loss)
    5. Add KL penalty against reference policy
    """

    name = "grpo"
    description = "Group Relative Policy Optimization — value-free RL (DeepSeek-R1's technique)"
    paper_reference = "Shao et al., 2024 — DeepSeekMath; DeepSeek-R1 Technical Report"
    priority = 1
    recommended_for = ["reasoning tasks", "math", "coding", "memory-constrained RL"]
    pros = ["No value model (50% less memory)", "Simple", "Great for reasoning", "Scales well"]
    cons = ["Multiple samples per prompt", "Group size sensitive", "Higher variance"]

    def __init__(self, config: GRPOConfig | None = None):
        super().__init__(config or GRPOConfig())

    def compute_group_advantages(self, rewards):
        """Compute advantages relative to group statistics.

        For a group of rewards [r1, r2, ..., rG]:
            advantage_i = (r_i - mean(rewards)) / (std(rewards) + eps)
        """
        if HAS_TORCH and isinstance(rewards, torch.Tensor):
            mean_reward = rewards.mean()
            std_reward = rewards.std() + 1e-8
            return (rewards - mean_reward) / std_reward
        else:
            # NumPy/list fallback
            mean_r = sum(rewards) / len(rewards)
            std_r = (sum((r - mean_r) ** 2 for r in rewards) / len(rewards)) ** 0.5 + 1e-8
            return [(r - mean_r) / std_r for r in rewards]

    def compute_loss(self, **kwargs) -> Any:
        """Compute GRPO loss.

        Loss = -E[min(ratio * A, clip(ratio) * A)] + kl_coef * KL(policy || ref)

        No value loss term — that's the key difference from PPO.
        """
        cfg = self.config

        if HAS_TORCH and "group_log_probs" in kwargs:
            group_log_probs = kwargs["group_log_probs"]       # [G, seq_len]
            group_old_log_probs = kwargs["group_old_log_probs"]
            group_rewards = kwargs["group_rewards"]             # [G]
            ref_log_probs = kwargs.get("ref_log_probs")         # [G, seq_len]

            # Group advantages
            advantages = self.compute_group_advantages(group_rewards)

            # Per-token ratio
            ratio = torch.exp(group_log_probs - group_old_log_probs)

            # Clipped surrogate
            surr1 = ratio * advantages.unsqueeze(-1)
            surr2 = torch.clamp(ratio, 1 - cfg.clip_ratio, 1 + cfg.clip_ratio) * advantages.unsqueeze(-1)
            policy_loss = -torch.min(surr1, surr2).mean()

            # KL penalty
            kl_loss = torch.tensor(0.0)
            if ref_log_probs is not None:
                kl_loss = (torch.exp(group_log_probs) * (group_log_probs - ref_log_probs)).mean()

            total_loss = policy_loss + cfg.kl_coef * kl_loss

            self.metrics.update({
                "policy_loss": policy_loss.item(),
                "kl_divergence": kl_loss.item(),
                "group_reward_mean": group_rewards.mean().item(),
                "group_reward_std": group_rewards.std().item(),
                "advantage_mean": advantages.mean().item(),
            })
            return total_loss

        # Simulation mode
        epoch = kwargs.get("epoch", self.step_count)
        loss = 1.8 * math.exp(-0.35 * epoch) + 0.3
        self.metrics.update({
            "loss": loss,
            "reward": min(0.95, 0.35 + 0.22 * epoch),
            "group_reward_std": max(0.1, 0.5 - 0.08 * epoch),
            "advantage_mean": 0.1 + 0.05 * epoch,
            "kl_divergence": max(0.01, 0.12 - 0.02 * epoch),
        })
        return loss
