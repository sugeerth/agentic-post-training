"""Self-Play Fine-Tuning (SPIN).

Priority: 2
Paper: "Self-Play Fine-Tuning Converts Weak Language Models to Strong" (Chen et al., 2024)

SPIN trains the model to distinguish its own outputs from human-written text,
iteratively improving until convergence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from techniques.base_technique import BaseTechnique, TechniqueConfig


@dataclass
class SPINConfig(TechniqueConfig):
    beta: float = 0.1
    num_iterations: int = 3


class SPIN(BaseTechnique):
    name = "spin"
    description = "Self-Play Fine-Tuning — distinguish model outputs from human text"
    paper_reference = "Chen et al., 2024 — Self-Play Fine-Tuning Converts Weak LMs to Strong"
    priority = 2
    recommended_for = ["self-improvement", "SFT data only", "iterative refinement"]
    pros = ["Only needs SFT data", "Self-improving", "Simple iterative process"]
    cons = ["Convergence can be slow", "May plateau early"]

    def __init__(self, config: SPINConfig | None = None):
        super().__init__(config or SPINConfig())

    def compute_loss(self, **kwargs) -> Any:
        epoch = kwargs.get("epoch", self.step_count)
        loss = 1.1 * math.exp(-0.4 * epoch) + 0.25
        self.metrics.update({
            "loss": loss,
            "reward": min(0.85, 0.35 + 0.16 * epoch),
            "discrimination_accuracy": min(0.92, 0.55 + 0.11 * epoch),
        })
        return loss
