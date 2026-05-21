"""Pure pipeline planner — turns a config into a list of stages.

Extracted from `CoordinatorAgent` so the planning logic is unit-testable
without asyncio, without a message bus, and without printing anything.

The Coordinator now imports `Planner` and uses it as a stage source.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PipelineStage:
    """One step in the post-training pipeline.

    Kept as a dataclass (not Pydantic) because it carries a mutable
    `result` dict that's written during execution. Config validation lives
    in `pipeline.config.PipelineConfig`, not here.
    """

    name: str
    description: str
    agent_role: str
    config: dict[str, Any] = field(default_factory=dict)
    status: str = "pending"
    result: dict[str, Any] = field(default_factory=dict)
    duration: float = 0.0


DEFAULT_STAGES: list[PipelineStage] = [
    PipelineStage("data_prep", "Prepare and validate training data", "trainer"),
    PipelineStage("technique_selection", "Select optimal training technique", "coordinator"),
    PipelineStage("training", "Execute post-training with selected technique", "trainer"),
    PipelineStage("optimization", "Quantize, prune, or distill the model", "optimizer"),
    PipelineStage("evaluation", "Benchmark and evaluate the trained model", "evaluator"),
]


def plan(config: dict[str, Any] | None = None) -> list[PipelineStage]:
    """Build a stage list from a flat config dict.

    The config is keyed by stage name — `{"training": {...}, "evaluation": {...}}`
    — so each stage's per-stage overrides are isolated. Top-level keys that
    don't match a stage name are ignored at this layer; the Coordinator may
    still read them (e.g. `technique` is consumed by the coordinator stage).
    """
    cfg = config or {}
    stages = [
        PipelineStage(s.name, s.description, s.agent_role, dict(s.config))
        for s in DEFAULT_STAGES
    ]
    for stage in stages:
        stage.config.update(cfg.get(stage.name, {}))
    return stages
