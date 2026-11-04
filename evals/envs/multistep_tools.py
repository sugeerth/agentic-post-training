"""Multi-step tool use / long-horizon: follow a chain of tool results to an answer.

Each task is a chain of duplicate tickets of length `depth`. The model must
call `get_ticket` repeatedly until it reaches the open root ticket, then
report its assignee. Tagging every task with `depth=N` gives a horizon curve:
pass rate as a function of how many dependent steps the task needs.
"""

from __future__ import annotations

import json
import random
from itertools import pairwise

from evals.environment import Environment
from evals.envs._schema import schema_failure
from evals.types import Action, Grade, Task, ToolCall, ToolSpec, Trajectory

TOOLS = (ToolSpec("get_ticket", "Fetch a support ticket by id.", {"ticket_id": "string"}),)
DEPTHS = (1, 2, 3, 4, 6, 8)
PER_DEPTH = 3
SLACK = 2  # extra steps allowed beyond the optimal number
_PEOPLE = ("ana", "bao", "chen", "dara", "eli", "fatima", "goran", "hana", "ivan", "jun")


def _make_task(i: int, depth: int, rng: random.Random) -> Task:
    ids = [f"T-{rng.randint(1000, 9999)}" for _ in range(depth + 1)]
    while len(set(ids)) < len(ids):  # ids must be unique within a chain
        ids = [f"T-{rng.randint(1000, 9999)}" for _ in range(depth + 1)]
    assignee = rng.choice(_PEOPLE)
    tickets: dict[str, dict[str, str]] = {
        a: {"id": a, "status": "duplicate", "duplicate_of": b} for a, b in pairwise(ids)
    }
    tickets[ids[-1]] = {"id": ids[-1], "status": "open", "assignee": assignee}
    reference = (
        *(Action(tool_call=ToolCall("get_ticket", {"ticket_id": t})) for t in ids),
        Action(text=assignee),
    )
    return Task(
        id=f"ms-{i:02d}",
        prompt=(
            f"Ticket {ids[0]} may be a duplicate. Follow duplicate_of links to the "
            "root (open) ticket and reply with only its assignee."
        ),
        tools=TOOLS,
        reference=reference,
        data={"tickets": tickets, "answer": assignee, "depth": depth},
        tags=(f"depth={depth}",),
    )


class MultiStepToolsEnv(Environment):
    name = "multistep_tools"

    def tasks(self) -> list[Task]:
        rng = random.Random(0)
        depths = [d for d in DEPTHS for _ in range(PER_DEPTH)]
        return [_make_task(i, d, rng) for i, d in enumerate(depths)]

    def max_steps_for(self, task: Task) -> int:
        return len(task.reference) + SLACK

    def respond(self, task: Task, action: Action) -> str | None:
        call = action.tool_call
        assert call is not None
        failure = schema_failure(call, task.tools)
        if failure:
            return f"error: {failure}"
        ticket = task.data["tickets"].get(call.arguments["ticket_id"])
        return json.dumps(ticket) if ticket else "error: not_found"

    def grade(self, task: Task, trajectory: Trajectory) -> Grade:
        final = trajectory.final
        answer = (final.text if final else "").strip().lower()
        passed = final is not None and final.tool_call is None and answer == task.data["answer"]
        failure: str | None = None
        if not passed:
            calls = [t.action.tool_call for t in trajectory.turns if t.action.tool_call]
            schema_errors = [f for c in calls if (f := schema_failure(c, task.tools))]
            if final is not None and final.tool_call is not None:
                failure = "exceeded_budget"
            elif schema_errors:
                failure = schema_errors[0]
            elif len(calls) < len(task.reference) - 1:
                failure = "answered_early"
            else:
                failure = "wrong_answer"
        steps = len(trajectory.turns)
        return Grade(
            task.id, self.name, passed, float(passed), failure, task.tags, {"steps": steps}
        )
