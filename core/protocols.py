"""Structural typing for plug-in components.

Each protocol describes the *shape* of an object that conforms to a slot —
technique, dataset loader, evaluator, backend, reporter, agent. We use
`typing.Protocol`, not ABCs, because:

  1. Implementers don't need to inherit anything to conform.
  2. Third-party plug-ins can satisfy the protocol without importing us.
  3. Tests can pass a tiny fake without subclassing.

The cost: type errors are caught by `mypy --strict`, not at import time.
That trade is worth it.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol, runtime_checkable

from core.types import (
    EvalResult,
    JobHandle,
    PreferencePair,
    RolloutBatch,
    StepMetrics,
    TrainingExample,
    TrainingJob,
)

# `runtime_checkable` lets us do `isinstance(obj, Technique)` for friendly
# error messages in the registry. It only checks method *names*, not
# signatures — full enforcement is mypy's job.


@runtime_checkable
class DatasetLoader(Protocol):
    """Anything that yields TrainingExamples or PreferencePairs.

    Implementations should be lazy where possible — return a generator, not
    a materialized list — so we can stream large datasets without holding
    them in memory.
    """

    name: str

    def load(
        self, split: str
    ) -> Iterable[TrainingExample | PreferencePair]: ...


@runtime_checkable
class Technique(Protocol):
    """A post-training method (DPO, GRPO, PPO, …).

    Lifecycle:
      1. `prepare(model, tokenizer, cfg)` — wire optimizer, ref model, LoRA.
      2. `step(batch)` — one optimization step, returns StepMetrics.
      3. `save(path)` — persist adapter / full weights.

    Techniques do NOT own the training loop, the data loader, or the device.
    Those are the trainer's job. A technique is *just the math*.
    """

    name: str

    def prepare(
        self, model: Any, tokenizer: Any, cfg: Any
    ) -> None: ...

    def step(
        self, batch: PreferencePair | TrainingExample | RolloutBatch
    ) -> StepMetrics: ...

    def save(self, path: str) -> None: ...


@runtime_checkable
class Evaluator(Protocol):
    """A benchmark or judge that scores a model.

    Stateless from the caller's POV: given `(model, tokenizer)`, return one
    `EvalResult`. Internal caching is the evaluator's business.
    """

    name: str

    def evaluate(self, model: Any, tokenizer: Any) -> EvalResult: ...


@runtime_checkable
class Backend(Protocol):
    """A place to run a TrainingJob — local, Colab, Kaggle, Vast.ai, ….

    `dry_run` MUST print the plan + estimated cost without spending money.
    `launch` is allowed to spend money but must respect `confirmed=False`
    semantics for paid backends (see VastAIBackend in Phase 4).
    """

    name: str

    def dry_run(self, job: TrainingJob) -> dict[str, Any]: ...

    def launch(self, job: TrainingJob) -> JobHandle: ...


@runtime_checkable
class Reporter(Protocol):
    """Pluggable observer for agent / training events.

    Replaces the scattered `print()` calls in the legacy code. Implementations:
      - `ConsoleReporter` (TTY-friendly, ANSI colors)
      - `JSONLReporter` (machine-readable, one event per line)
      - `NoopReporter` (silent, for tests)

    The contract is one method; everything else is an event payload.
    """

    def emit(self, event: str, payload: dict[str, Any]) -> None: ...


@runtime_checkable
class Agent(Protocol):
    """An agent on the message bus.

    Agents are message handlers, not threads or processes. They receive a
    message, optionally produce a reply, and persist nothing the bus
    doesn't see. State that must survive crash should live in the
    pipeline's `progress.json`, not in the agent.
    """

    role: str

    def handle(self, message: Any) -> Any | None: ...
