"""Data carriers for behavior evals. Pure data, no logic.

The whole eval stack speaks these six types:

    Task ──► (policy acts, env responds) ──► Trajectory ──► Grade

Frozen dataclasses so a Task or Grade can't be mutated after it's logged.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """A callable tool, described the way a model sees it.

    `params` maps argument name -> JSON type ("string", "integer", "number",
    "boolean"). Every param is required; optional args are a source of
    ambiguity we deliberately leave out.
    """

    name: str
    description: str
    params: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ToolCall:
    name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Action:
    """One model output: either a tool call or a final text answer.

    `confidence` is the model's stated P(correct) in [0, 1]. Only graders that
    measure calibration read it.
    """

    text: str = ""
    tool_call: ToolCall | None = None
    confidence: float | None = None


@dataclass(frozen=True, slots=True)
class Task:
    """One eval item.

    `reference` is the ideal action sequence. It is used ONLY by graders,
    the simulated policy, and failure->training-data conversion. Policies
    under test never see it (they get an `Observation`).

    `tags` are slice keys for diagnostics, e.g. ("depth=4",).
    """

    id: str
    prompt: str
    tools: tuple[ToolSpec, ...] = ()
    reference: tuple[Action, ...] = ()
    data: Mapping[str, Any] = field(default_factory=dict)
    tags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Turn:
    """An action plus what the environment returned (None = episode over)."""

    action: Action
    result: str | None = None


@dataclass(frozen=True, slots=True)
class Observation:
    """Everything a policy is allowed to see."""

    task_id: str
    prompt: str
    tools: tuple[ToolSpec, ...]
    history: tuple[Turn, ...] = ()


@dataclass(frozen=True, slots=True)
class Trajectory:
    task_id: str
    env: str
    turns: tuple[Turn, ...]

    @property
    def final(self) -> Action | None:
        return self.turns[-1].action if self.turns else None


@dataclass(frozen=True, slots=True)
class Grade:
    """Verdict on one trajectory.

    `failure` is a short, stable tag ("missing_arg", "hallucination", …) and
    is the unit we count, cluster and turn into training data.
    """

    task_id: str
    env: str
    passed: bool
    score: float
    failure: str | None = None
    tags: tuple[str, ...] = ()
    details: Mapping[str, Any] = field(default_factory=dict)
