"""Self-Play Fine-Tuning (SPIN).

Priority: 3 (Experimental — simulation stub only)
Paper: "Self-Play Fine-Tuning Converts Weak Language Models to Strong" (Chen et al., 2024)

⚠ SIMULATION STUB: This file only produces a plausible-looking decay curve.
The real SPIN algorithm — iterative self-play discrimination between model
outputs and human reference — is **not implemented**. Instantiating this
class emits a `FutureWarning`.

SPIN trains the model to distinguish its own outputs from human-written text,
iteratively improving until convergence.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any, ClassVar

from techniques.base_technique import BaseTechnique, TechniqueConfig


@dataclass
class SPINConfig(TechniqueConfig):
    beta: float = 0.1
    num_iterations: int = 3


class SPIN(BaseTechnique):
    name = "spin"
    description = "Self-Play Fine-Tuning — distinguish model outputs from human text"
    paper_reference = "Chen et al., 2024 — Self-Play Fine-Tuning Converts Weak LMs to Strong"
    priority = 3
    is_experimental = True
    recommended_for: ClassVar[list[str]] = ["self-improvement", "SFT data only", "iterative refinement"]
    pros: ClassVar[list[str]] = ["Only needs SFT data", "Self-improving", "Simple iterative process"]
    cons: ClassVar[list[str]] = ["Convergence can be slow", "May plateau early"]

    def __init__(self, config: SPINConfig | None = None):
        warnings.warn(
            "SPIN is a simulation stub — the real iterative self-play loop is "
            "not implemented. See techniques/spin.py docstring.",
            FutureWarning, stacklevel=2,
        )
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
