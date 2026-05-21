"""Simple Preference Optimization (SimPO).

Priority: 3 (Experimental — simulation stub only)
Paper: "SimPO: Simple Preference Optimization with a Reference-Free Reward" (Meng et al., 2024)

⚠ SIMULATION STUB: The real SimPO loss (length-normalized log-prob ratio
with a fixed margin γ) is available as a primitive in
`techniques._base.ratio_loss.pairwise_ratio_loss(..., loss_type=SIMPO)` —
but this top-level `SimPO` class still runs only the decay simulation.
Instantiating it emits a `FutureWarning`. Phase 2 follow-up: wire the
primitive into a real `SimPOTechnique`.

SimPO uses length-normalized log probabilities as implicit reward,
eliminating the need for a reference model entirely.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any, ClassVar

from techniques.base_technique import BaseTechnique, TechniqueConfig


@dataclass
class SimPOConfig(TechniqueConfig):
    beta: float = 2.0
    gamma: float = 0.5  # Reward margin


class SimPO(BaseTechnique):
    name = "simpo"
    description = "SimPO — reference-free, length-normalized preference optimization"
    paper_reference = "Meng et al., 2024 — SimPO: Simple Preference Optimization"
    priority = 3
    is_experimental = True
    recommended_for: ClassVar[list[str]] = ["reference-free alignment", "memory-constrained setups", "simplicity"]
    pros: ClassVar[list[str]] = ["No reference model", "Length-normalized (fair)", "Very simple"]
    cons: ClassVar[list[str]] = ["Less expressive", "Newer technique"]

    def __init__(self, config: SimPOConfig | None = None):
        warnings.warn(
            "SimPO is a simulation stub — top-level class runs only the decay "
            "curve. The real length-normalized log-prob ratio is implemented "
            "as a primitive in techniques._base.ratio_loss.pairwise_ratio_loss; "
            "wire it through a real SimPOTechnique to use it.",
            FutureWarning, stacklevel=2,
        )
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
