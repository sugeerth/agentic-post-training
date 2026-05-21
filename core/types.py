"""Typed data carriers used across the framework.

Frozen dataclasses, not Pydantic models — these are passed through hot loops
and Pydantic adds overhead we don't need for pure data. Pydantic is reserved
for *user-facing config*, where validation pays for itself (see `core.config`).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


@dataclass(frozen=True, slots=True)
class TrainingExample:
    """A single SFT-style training example.

    `metadata` is the escape hatch for dataset-specific fields (source,
    quality score, language tag) without bloating the core schema.
    """

    prompt: str
    response: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PreferencePair:
    """A single preference pair for DPO/ORPO/SPO/KTO/SimPO/IPO."""

    prompt: str
    chosen: str
    rejected: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RolloutBatch:
    """One rollout group for RL methods (PPO, GRPO, RLHF).

    `responses` is a list-of-lists when the technique generates multiple
    samples per prompt (GRPO group_size > 1). For PPO with single samples,
    each inner list has length 1.
    """

    prompts: list[str]
    responses: list[list[str]]
    rewards: list[list[float]]
    log_probs: Any | None = None  # tensor, shape: (batch, group, seq)
    ref_log_probs: Any | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class StepMetrics:
    """What a single technique training step returns.

    `loss` is mandatory and is the value autograd backprops on. `extras` is
    everything else worth logging — KL divergence, reward stats, advantages,
    etc. Keys are stable strings; values must be JSON-serializable.
    """

    loss: float
    step: int
    extras: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EvalResult:
    """A single benchmark result with optional confidence interval.

    `n` is the number of items evaluated. `ci_low` / `ci_high` are the 95% CI
    bounds when the evaluator computes them; otherwise `None`.
    """

    metric_name: str
    value: float
    n: int
    ci_low: float | None = None
    ci_high: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class TrainingJob:
    """Static description of a training job, handed to a Backend.

    Frozen on purpose: once a backend accepts a job, the spec doesn't change.
    Mutable progress (which step, which checkpoint) lives in `JobHandle`.
    """

    job_id: str
    technique: str  # registry key, e.g. "grpo"
    technique_config: Mapping[str, Any]
    model_name: str
    dataset: str  # registry key, e.g. "ultrafeedback"
    output_dir: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class JobHandle:
    """Mutable view into a running job. Backends return one of these.

    Not frozen because `status`, `last_checkpoint`, and `metrics` evolve.
    Everything else is immutable post-construction.
    """

    job_id: str
    status: JobStatus
    backend: str
    last_checkpoint: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    error: str | None = None
