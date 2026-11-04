"""Instruction following: verifiable formatting/content constraints (IFEval-style).

Every constraint is a pure function `(response, arg) -> bool`, so grading is
deterministic and needs no judge model. A task passes only if every
constraint holds; `score` is the fraction satisfied (partial credit).
"""

from __future__ import annotations

import json
from collections.abc import Callable

from evals.environment import Environment
from evals.types import Action, Grade, Task, Trajectory


def _is_json_object(text: str, _: object) -> bool:
    try:
        return isinstance(json.loads(text), dict)
    except json.JSONDecodeError:
        return False


CHECKS: dict[str, Callable[[str, object], bool]] = {
    "max_words": lambda r, n: len(r.split()) <= int(n),  # type: ignore[call-overload]
    "include": lambda r, w: str(w).lower() in r.lower(),
    "exclude": lambda r, w: str(w).lower() not in r.lower(),
    "lowercase": lambda r, _: r == r.lower(),
    "no_commas": lambda r, _: "," not in r,
    "json_object": _is_json_object,
    "ends_with": lambda r, s: r.rstrip().endswith(str(s)),
    "n_bullets": lambda r, n: sum(ln.startswith("- ") for ln in r.splitlines()) == int(n),  # type: ignore[call-overload]
}

# (prompt, constraints, a response that satisfies them)
_CASES: list[tuple[str, list[tuple[str, object]], str]] = [
    (
        "Describe a code review in at most 12 words, all lowercase.",
        [("max_words", 12), ("lowercase", None)],
        "a teammate reads your change and suggests fixes before merge.",
    ),
    (
        "Explain retries without using commas. Mention 'backoff'.",
        [("no_commas", None), ("include", "backoff")],
        "Retry failed calls with exponential backoff so you do not overload the server.",
    ),
    (
        'Return a JSON object with keys "status" and "count". No prose.',
        [("json_object", None), ("include", "status")],
        '{"status": "ok", "count": 3}',
    ),
    (
        "List exactly 3 testing tips as '- ' bullets.",
        [("n_bullets", 3)],
        "- Test behavior not implementation\n- Keep tests fast\n- One assert per idea",
    ),
    (
        "Summarize RL in one sentence that ends with 'Done.' Do not say 'reward'.",
        [("exclude", "reward"), ("ends_with", "Done.")],
        "An agent learns by trial and error from feedback. Done.",
    ),
    (
        "Name a sorting algorithm in under 5 words, lowercase, no commas.",
        [("max_words", 5), ("lowercase", None), ("no_commas", None)],
        "merge sort",
    ),
    (
        "Give 2 bullets on logging. Include the word 'structured'.",
        [("n_bullets", 2), ("include", "structured")],
        "- Prefer structured logs\n- Log at boundaries",
    ),
    (
        'Reply with a JSON object containing "ok": true and nothing else.',
        [("json_object", None), ("exclude", "sure")],
        '{"ok": true}',
    ),
]


class InstructionFollowingEnv(Environment):
    name = "instruction_following"

    def tasks(self) -> list[Task]:
        return [
            Task(
                id=f"if-{i:02d}",
                prompt=prompt,
                reference=(Action(text=response),),
                data={"constraints": constraints},
                tags=tuple(f"constraint={name}" for name, _ in constraints),
            )
            for i, (prompt, constraints, response) in enumerate(_CASES)
        ]

    def grade(self, task: Task, trajectory: Trajectory) -> Grade:
        text = trajectory.final.text if trajectory.final else ""
        violated = [name for name, arg in task.data["constraints"] if not CHECKS[name](text, arg)]
        n = len(task.data["constraints"])
        return Grade(
            task.id,
            self.name,
            passed=not violated,
            score=(n - len(violated)) / n,
            failure=f"violated:{violated[0]}" if violated else None,
            tags=task.tags,
            details={"violated": violated},
        )
