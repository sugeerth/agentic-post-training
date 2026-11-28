"""Use case 1: multi-step word problems, solved with a calculator tool. Level = number of steps (1-6)."""
from __future__ import annotations

import ast
import operator
import random
import re

from selflearn.usecase import Task, UseCase

SYSTEM_PROMPT = (
    "You solve math word problems step by step. You have a calculator.\n"
    "To use it, write <calc>EXPRESSION</calc> and stop. The result comes back as <result>VALUE</result>.\n"
    "You may use the calculator several times. When you are sure, write <answer>NUMBER</answer>.\n"
    "Example:\n"
    "Question: A shop has 12 pens. It buys 3 packs of 25 pens. How many pens does the shop have now?\n"
    "<calc>12 + 3 * 25</calc>\n<result>87</result>\n<answer>87</answer>"
)

THEMES = (
    ("A warehouse", "the warehouse", "boxes", "trucks", "rooms"),
    ("A library", "the library", "books", "carts", "shelves"),
    ("A farm", "the farm", "eggs", "baskets", "crates"),
    ("A bakery", "the bakery", "cookies", "trays", "jars"),
    ("A school", "the school", "pencils", "packs", "classrooms"),
)
MULT_WORD = {2: "doubles", 3: "triples", 4: "quadruples"}
LEVELS = (1, 2, 3, 4, 5, 6)


def _ranges(level: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """(start value range, operand range): numbers grow with the level."""
    if level <= 2:
        return (10, 99), (2, 30)
    if level <= 4:
        return (100, 999), (11, 99)
    return (1000, 9999), (12, 99)


def make_task(level: int, rng: random.Random) -> Task:
    """A word problem with `level` operations; every intermediate value stays a positive integer."""
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}")
    subj, short, unit, container, group = rng.choice(THEMES)
    (lo, hi), (olo, ohi) = _ranges(level)
    value = rng.randint(lo, hi)
    parts = [f"{subj} has {value} {unit}."]
    ops = 0
    while ops < level:
        kind = rng.choice(("groups", "add", "sub", "mul", "div"))
        if kind == "groups":
            k, m = rng.randint(2, 12 if level <= 2 else 40), rng.randint(olo, ohi)
            value += k * m
            parts.append(f"It receives {k} {container} with {m} {unit} each.")
        elif kind == "add":
            n = rng.randint(olo, ohi * 10)
            value += n
            parts.append(f"It gets {n} more {unit}.")
        elif kind == "sub":
            if value < 4:
                continue
            n = rng.randint(1, max(1, value * 3 // 4))
            value -= n
            parts.append(f"It gives away {n} {unit}.")
        elif kind == "mul":
            f = rng.choice((2, 3, 4))
            value *= f
            parts.append(f"Then the number of {unit} at {short} {MULT_WORD[f]}.")
        else:  # exact division only, so the answer stays an integer
            divisors = [d for d in range(2, 13) if value % d == 0]
            if not divisors:
                continue
            d = rng.choice(divisors)
            value //= d
            parts.append(f"It splits the {unit} equally into {d} {group} and keeps only one of them.")
        ops += 1
    parts.append(f"How many {unit} does {short} have now?")
    return Task(" ".join(parts), value, level)


# ----------------------------------------------------------------------------- the calculator tool
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


# ----------------------------------------------------------------------------- shortcut probes for the env audit
# Number-copying habits a small model falls into; an exact-answer reward should pay them (almost) nothing.
def _probe(guess):
    def play(task: Task, ep) -> str:
        nums = [int(x) for x in re.findall(r"\d+", task.question)]
        final = f"<answer>{guess(nums) if nums else 0}</answer>"
        ep.step(final)
        return final
    return play


PROBES = {
    "answer the first number": _probe(lambda n: n[0]),
    "answer the last number": _probe(lambda n: n[-1]),
    "answer the sum of all numbers": _probe(sum),
}


USE_CASE = UseCase(name="math", system_prompt=SYSTEM_PROMPT, tools={"calc": calc}, levels=LEVELS,
                   make_task=make_task, max_new=96, eval_per_level=20, probes=PROBES,
                   summary="taught itself to use a calculator on multi-step word problems",
                   level_note="level = number of steps in the problem")
