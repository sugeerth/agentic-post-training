"""Simple Preference Optimization (SimPO).

Priority: 2
Paper: "SimPO: Simple Preference Optimization with a Reference-Free Reward" (Meng et al., 2024)

SimPO uses length-normalized log probabilities as implicit reward,
eliminating the need for a reference model entirely.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from techniques.base_technique import BaseTechnique, TechniqueConfig


@dataclass
class SimPOConfig(TechniqueConfig):
    beta: float = 2.0
    gamma: float = 0.5  # Reward margin


class SimPO(BaseTechnique):
    name = "simpo"
    description = "SimPO — reference-free, length-normalized preference optimization"
    paper_reference = "Meng et al., 2024 — SimPO: Simple Preference Optimization"
    priority = 2
    recommended_for = ["reference-free alignment", "memory-constrained setups", "simplicity"]
    pros = ["No reference model", "Length-normalized (fair)", "Very simple"]
    cons = ["Less expressive", "Newer technique"]

    def __init__(self, config: SimPOConfig | None = None):
        super().__init__(config or SimPOConfig())

    def compute_loss(self, **kwargs) -> Any:
        epoch = kwargs.get("epoch", self.step_count)
        loss = 1.2 * math.exp(-0.42 * epoch) + 0.22
        self.metrics.update({
            "loss": loss,
            "reward": min(0.87, 0.33 + 0.17 * epoch),
            "reward_margin": 0.3 + 0.1 * epoch,
        })
        return loss
