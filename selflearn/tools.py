"""Tools the agent can call, as a registry {name: function(str) -> str}.

The agent calls a tool by writing <name>ARGUMENT</name>; the result comes back as <result>VALUE</result>.
Adding a tool = adding one entry to TOOLS; the rollout code does not change (open for extension).
"""
from __future__ import annotations

import ast
import operator
import re
from collections.abc import Callable

_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod}
_UN = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Mult) and abs(left) * abs(right) > 1e15:
            raise ValueError("number too large")
        return _BIN[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UN:
        return _UN[type(node.op)](_eval(node.operand))
    raise ValueError("only numbers, + - * / // % and parentheses are allowed")


def calc(expr: str) -> str:
    """Safe arithmetic: parses the expression and evaluates only numbers and arithmetic operators. Never runs code."""
    expr = expr.strip().replace(",", "").replace("×", "*").replace("÷", "/")
    if not expr or len(expr) > 200:
        return "error: empty or too long"
    try:
        v = _eval(ast.parse(expr, mode="eval"))
    except ZeroDivisionError:
        return "error: division by zero"
    except (ValueError, SyntaxError, TypeError, RecursionError) as e:
        return f"error: {e}"[:80]
    if isinstance(v, float):
        return str(int(v)) if v.is_integer() else f"{v:.6g}"
    return str(v)


TOOLS: dict[str, Callable[[str], str]] = {"calc": calc}


def last_call(text: str, tools: dict[str, Callable[[str], str]]) -> tuple[str, str] | None:
    """The last closed tool call in `text` as (tool name, argument), or None."""
    best = None
    for name in tools:
        for m in re.finditer(rf"<{name}>(.*?)</{name}>", text, re.S):
            if best is None or m.end() > best[2]:
                best = (name, m.group(1), m.end())
    return (best[0], best[1]) if best else None


def run_call(text: str, tools: dict[str, Callable[[str], str]]) -> str | None:
    """Execute the last tool call in `text`; None if there is none."""
    call = last_call(text, tools)
    return None if call is None else tools[call[0]](call[1])
