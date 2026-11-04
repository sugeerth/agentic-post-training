"""Failures -> diagnostics and training data.

The loop this module closes:  eval finds failure -> failure becomes a
preference pair (chosen = reference, rejected = what the model did) -> pair
feeds DPO/ORPO/KTO in `techniques/`.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Sequence

from core.types import PreferencePair
from evals.types import Action, Grade, Task, Trajectory


def failure_counts(grades: Iterable[Grade]) -> Counter[str]:
    return Counter(g.failure for g in grades if g.failure)


def render_action(action: Action) -> str:
    if action.tool_call:
        call = {"name": action.tool_call.name, "arguments": dict(action.tool_call.arguments)}
        return f"<tool_call>{json.dumps(call)}</tool_call>"
    if action.confidence is not None:
        return f"{action.text}\nconfidence: {action.confidence:.2f}"
    return action.text


def render(actions: Sequence[Action]) -> str:
    return "\n".join(render_action(a) for a in actions)


def to_preference_pairs(
    tasks: Sequence[Task], trajectories: Sequence[Trajectory], grades: Sequence[Grade]
) -> list[PreferencePair]:
    """One pair per failed trajectory. Grades and trajectories must be aligned."""
    by_id = {t.id: t for t in tasks}
    pairs = []
    for traj, grade in zip(trajectories, grades, strict=True):
        if grade.passed:
            continue
        task = by_id[traj.task_id]
        pairs.append(
            PreferencePair(
                prompt=task.prompt,
                chosen=render(task.reference),
                rejected=render([t.action for t in traj.turns]),
                metadata={"env": grade.env, "task_id": task.id, "failure": grade.failure},
            )
        )
    return pairs
