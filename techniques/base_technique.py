"""Base class for all post-training techniques."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar


@dataclass
class TechniqueConfig:
    """Base configuration for all techniques."""
    learning_rate: float = 2e-5
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    epochs: int = 3
    max_length: int = 512
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    seed: int = 42
    fp16: bool = False
    bf16: bool = True
    logging_steps: int = 10
    save_steps: int = 100
    output_dir: str = "./output"


class BaseTechnique(ABC):
    """Abstract base class for all post-training techniques.

    Every technique must define:
    - name, description, paper_reference
    - priority (1=core, 2=advanced, 3=experimental)
    - recommended_for: list of use cases
    - configure(): set up the technique
    - train_step(): single training step
    - compute_loss(): loss computation
    - get_metrics(): current metrics
    """

    name: str = "base"
    description: str = "Base technique"
    paper_reference: str = ""
    priority: int = 1
    recommended_for: ClassVar[list[str]] = []
    pros: ClassVar[list[str]] = []
    cons: ClassVar[list[str]] = []
    # Subclasses set this to True when they ship only the simulation path —
    # i.e. the real algorithm is not yet implemented. Instantiating an
    # experimental technique emits a FutureWarning so users aren't surprised
    # by numbers that look real but aren't.
    is_experimental: bool = False

    def __init__(self, config: TechniqueConfig | None = None):
        self.config = config or self._default_config()
        self.metrics: dict[str, float] = {}
        self.step_count = 0

    def _default_config(self) -> TechniqueConfig:
        return TechniqueConfig()

    @abstractmethod
    def compute_loss(self, **kwargs) -> Any:
        """Compute the technique-specific loss."""
        ...

    def train_step(self, **kwargs) -> dict[str, float]:
        """Execute a single training step. Returns metrics dict."""
        self.step_count += 1
        loss = self.compute_loss(**kwargs)
        self.metrics["loss"] = float(loss) if hasattr(loss, "__float__") else 0.0
        self.metrics["step"] = self.step_count
        return self.metrics

    def get_metrics(self) -> dict[str, float]:
        return dict(self.metrics)

    def info(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "paper": self.paper_reference,
            "priority": self.priority,
            "recommended_for": self.recommended_for,
            "pros": self.pros,
            "cons": self.cons,
        }
