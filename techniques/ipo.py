"""Identity Preference Optimization (IPO).

Priority: 3 (Experimental — simulation stub only)
Paper: "A General Theoretical Paradigm to Understand Learning from Human Feedback"
       (Azar et al., 2023)

⚠ SIMULATION STUB: The real IPO loss `(logits − 1/(2β))²` is available as a
primitive in `techniques._base.ratio_loss.pairwise_ratio_loss(..., loss_type=IPO)`
— but this top-level `IPO` class still runs only the decay simulation.
Instantiating it emits a `FutureWarning`. Phase 2 follow-up: wire the
primitive through a real `IPOTechnique`.

IPO adds regularization to address DPO's overfitting issues.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any, ClassVar

from techniques.base_technique import BaseTechnique, TechniqueConfig


@dataclass
class IPOConfig(TechniqueConfig):
    tau: float = 0.1  # Regularization strength


class IPO(BaseTechnique):
    name = "ipo"
    description = "Identity Preference Optimization — regularized DPO variant"
    paper_reference = "Azar et al., 2023 — A General Theoretical Paradigm for Learning from HF"
    priority = 3
    is_experimental = True
    recommended_for: ClassVar[list[str]] = ["when DPO overfits", "regularized alignment", "research"]
    pros: ClassVar[list[str]] = ["Better regularization than DPO", "Theoretically grounded"]
    cons: ClassVar[list[str]] = ["Marginal gains in practice", "Less adopted"]

    def __init__(self, config: IPOConfig | None = None):
        warnings.warn(
            "IPO is a simulation stub — top-level class runs only the decay "
            "curve. The real (logits − 1/(2β))² loss is implemented as a "
            "primitive in techniques._base.ratio_loss.pairwise_ratio_loss; "
            "wire it through a real IPOTechnique to use it.",
            FutureWarning, stacklevel=2,
        )
        super().__init__(config or IPOConfig())

    def compute_loss(self, **kwargs) -> Any:
        epoch = kwargs.get("epoch", self.step_count)
        loss = 1.3 * math.exp(-0.38 * epoch) + 0.27
        self.metrics.update({
            "loss": loss,
            "reward": min(0.86, 0.32 + 0.16 * epoch),
        })
        return loss
