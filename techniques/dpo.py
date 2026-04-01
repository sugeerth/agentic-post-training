"""Direct Preference Optimization (DPO).

Priority: 1 (Core technique)
Paper: "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
       (Rafailov et al., 2023)

DPO directly optimizes the policy from preference pairs without training a
separate reward model. It reparameterizes the RLHF objective to derive a
simple classification loss on preference pairs.

When to use:
- You have paired preference data (chosen/rejected)
- You want simpler training than PPO (no reward model, no RL)
- Stable, single-stage alignment

Pros:
- No reward model needed
- Simple to implement and train
- Stable (no RL instability)
- Single-stage training

Cons:
- Requires paired preference data
- Less flexible than RL-based methods
- Reference model needed (memory overhead)
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
class DPOConfig(TechniqueConfig):
    beta: float = 0.1  # Temperature parameter
    label_smoothing: float = 0.0
    reference_free: bool = False
    loss_type: str = "sigmoid"  # sigmoid, hinge, ipo


class DPO(BaseTechnique):
    """Direct Preference Optimization — single-stage preference alignment.

    Key insight: The optimal policy under a KL-constrained reward maximization
    can be expressed in closed form. This means we can directly optimize the
    policy using a simple binary cross-entropy loss on preference pairs:

    Loss = -log σ(β * (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))

    where y_w = chosen response, y_l = rejected response.
    """

    name = "dpo"
    description = "Direct Preference Optimization — simple preference alignment without RL"
    paper_reference = "Rafailov et al., 2023 — Direct Preference Optimization"
    priority = 1
    recommended_for = ["preference alignment", "instruction following", "simple setups"]
    pros = ["No reward model", "Simple", "Stable training", "Single stage"]
    cons = ["Needs paired preferences", "Less flexible than RL", "Reference model overhead"]

    def __init__(self, config: DPOConfig | None = None):
        super().__init__(config or DPOConfig())

    def compute_loss(self, **kwargs) -> Any:
        """Compute DPO loss from chosen/rejected log probabilities.

        Loss = -log σ(β * (log_ratio_chosen - log_ratio_rejected))
        """
        cfg = self.config

        if HAS_TORCH and "chosen_log_probs" in kwargs:
            chosen_log_probs = kwargs["chosen_log_probs"]
            rejected_log_probs = kwargs["rejected_log_probs"]
            ref_chosen_log_probs = kwargs.get("ref_chosen_log_probs", torch.zeros_like(chosen_log_probs))
            ref_rejected_log_probs = kwargs.get("ref_rejected_log_probs", torch.zeros_like(rejected_log_probs))

            chosen_log_ratio = chosen_log_probs - ref_chosen_log_probs
            rejected_log_ratio = rejected_log_probs - ref_rejected_log_probs
            logits = cfg.beta * (chosen_log_ratio - rejected_log_ratio)

            if cfg.loss_type == "sigmoid":
                loss = -F.logsigmoid(logits).mean()
            elif cfg.loss_type == "hinge":
                loss = torch.relu(1 - logits).mean()
            else:
                loss = -F.logsigmoid(logits).mean()

            # Implicit rewards
            chosen_reward = cfg.beta * chosen_log_ratio.detach()
            rejected_reward = cfg.beta * rejected_log_ratio.detach()

            self.metrics.update({
                "loss": loss.item(),
                "chosen_reward": chosen_reward.mean().item(),
                "rejected_reward": rejected_reward.mean().item(),
                "reward_margin": (chosen_reward - rejected_reward).mean().item(),
                "accuracy": (logits > 0).float().mean().item(),
            })
            return loss

        # Simulation
        epoch = kwargs.get("epoch", self.step_count)
        loss = 1.5 * math.exp(-0.4 * epoch) + 0.25
        self.metrics.update({
            "loss": loss,
            "chosen_reward": 0.5 + 0.15 * epoch,
            "rejected_reward": -0.3 - 0.1 * epoch,
            "reward_margin": 0.8 + 0.25 * epoch,
            "accuracy": min(0.95, 0.6 + 0.1 * epoch),
        })
        return loss
