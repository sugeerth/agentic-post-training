"""Function calling: pick the right tool and fill its arguments correctly.

Single turn. The grader checks, in order: did the model call anything, is the
call schema-valid, is it the right tool, are the values right.
"""

from __future__ import annotations

from evals.environment import Environment
from evals.envs._schema import schema_failure
from evals.types import Action, Grade, Task, ToolCall, ToolSpec, Trajectory

TOOLS = (
    ToolSpec("get_weather", "Current weather for a city.", {"city": "string", "unit": "string"}),
    ToolSpec(
        "send_email", "Send an email.", {"to": "string", "subject": "string", "body": "string"}
    ),
    ToolSpec(
        "create_event",
        "Create a calendar event.",
        {"title": "string", "date": "string", "duration_minutes": "integer"},
    ),
    ToolSpec(
        "search_flights",
        "Search flights.",
        {"origin": "string", "destination": "string", "date": "string", "passengers": "integer"},
    ),
    ToolSpec(
        "convert_currency",
        "Convert money between currencies.",
        {"amount": "number", "source": "string", "target": "string"},
    ),
)

# (prompt, tool, expected arguments)
_CASES: list[tuple[str, str, dict[str, object]]] = [
    ("What's the weather in Paris in celsius?", "get_weather", {"city": "Paris", "unit": "c"}),
    (
        "Is it hot in Austin right now? Use fahrenheit.",
        "get_weather",
        {"city": "Austin", "unit": "f"},
    ),
    (
        "Email bob@acme.com with subject 'Q3' saying 'Draft attached.'",
        "send_email",
        {"to": "bob@acme.com", "subject": "Q3", "body": "Draft attached."},
    ),
    (
        "Book 'Design review' on 2026-11-12 for 45 minutes.",
        "create_event",
        {"title": "Design review", "date": "2026-11-12", "duration_minutes": 45},
    ),
    (
        "Put a 90 minute 'Offsite' on my calendar for 2026-12-01.",
        "create_event",
        {"title": "Offsite", "date": "2026-12-01", "duration_minutes": 90},
    ),
    (
        "Find flights SFO to JFK on 2026-12-20 for 2 people.",
        "search_flights",
        {"origin": "SFO", "destination": "JFK", "date": "2026-12-20", "passengers": 2},
    ),
    (
        "One adult, LHR -> NRT, 2027-01-05. What flights exist?",
        "search_flights",
        {"origin": "LHR", "destination": "NRT", "date": "2027-01-05", "passengers": 1},
    ),
    (
        "How much is 250 USD in EUR?",
        "convert_currency",
        {"amount": 250, "source": "USD", "target": "EUR"},
    ),
    (
        "Convert 19.99 GBP to JPY.",
        "convert_currency",
        {"amount": 19.99, "source": "GBP", "target": "JPY"},
    ),
    (
        "Tell ops@acme.com 'Deploy done' with subject 'Release 4.2'.",
        "send_email",
        {"to": "ops@acme.com", "subject": "Release 4.2", "body": "Deploy done"},
    ),
]


def _norm(v: object) -> object:
    return v.strip().lower() if isinstance(v, str) else v


class FunctionCallingEnv(Environment):
    name = "function_calling"

    def tasks(self) -> list[Task]:
        return [
            Task(
                id=f"fc-{i:02d}",
                prompt=prompt,
                tools=TOOLS,
                reference=(Action(tool_call=ToolCall(tool, args)),),
                tags=(f"tool={tool}",),
            )
            for i, (prompt, tool, args) in enumerate(_CASES)
        ]

    def grade(self, task: Task, trajectory: Trajectory) -> Grade:
        expected = task.reference[0].tool_call
        call = trajectory.final.tool_call if trajectory.final else None
        assert expected is not None
        if call is None:
            failure: str | None = "no_call"
        else:
            failure = schema_failure(call, task.tools)
            if failure is None and call.name != expected.name:
                failure = "wrong_tool"
            elif failure is None and any(
                _norm(call.arguments[k]) != _norm(v) for k, v in expected.arguments.items()
            ):
                failure = "wrong_arg_value"
        return Grade(
            task.id, self.name, failure is None, float(failure is None), failure, task.tags
        )
