"""Tool calling, generic over any registry {name: function(str) -> str} (each use case brings its own tools).

The agent calls a tool by writing <name>ARGUMENT</name>; the result comes back as <result>VALUE</result>.
"""
from __future__ import annotations

import re
from collections.abc import Callable

Tools = dict[str, Callable[[str], str]]


def last_call(text: str, tools: Tools) -> tuple[str, str] | None:
    """The last closed tool call in `text` as (tool name, argument), or None."""
    best = None
    for name in tools:
        for m in re.finditer(rf"<{name}>(.*?)</{name}>", text, re.S):
            if best is None or m.end() > best[2]:
                best = (name, m.group(1), m.end())
    return (best[0], best[1]) if best else None


def run_call(text: str, tools: Tools) -> str | None:
    """Execute the last tool call in `text`; None if there is none."""
    call = last_call(text, tools)
    return None if call is None else tools[call[0]](call[1])
