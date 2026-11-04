"""Tool-call schema validation shared by every tool-using environment."""

from __future__ import annotations

from evals.types import ToolCall, ToolSpec

_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
}


def _type_ok(value: object, json_type: str) -> bool:
    if isinstance(value, bool) and json_type != "boolean":
        return False  # bool is an int subclass in Python; JSON disagrees.
    return isinstance(value, _JSON_TYPES[json_type])


def schema_failure(call: ToolCall, tools: tuple[ToolSpec, ...]) -> str | None:
    """Return a failure tag if `call` violates the declared schema, else None."""
    spec = next((t for t in tools if t.name == call.name), None)
    if spec is None:
        return "hallucinated_tool"
    if any(p not in call.arguments for p in spec.params):
        return "missing_arg"
    if any(a not in spec.params for a in call.arguments):
        return "unknown_arg"
    if any(not _type_ok(call.arguments[p], t) for p, t in spec.params.items()):
        return "bad_arg_type"
    return None
