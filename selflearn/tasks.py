"""Task generator: multi-step word problems with an exact integer answer and a difficulty knob (level = number of steps).

Pure Python, deterministic given the rng, so the held-out set is identical across runs.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

LEVELS = (1, 2, 3, 4, 5, 6)

THEMES = (
    ("A warehouse", "the warehouse", "boxes", "trucks", "rooms"),
    ("A library", "the library", "books", "carts", "shelves"),
    ("A farm", "the farm", "eggs", "baskets", "crates"),
    ("A bakery", "the bakery", "cookies", "trays", "jars"),
    ("A school", "the school", "pencils", "packs", "classrooms"),
)
MULT_WORD = {2: "doubles", 3: "triples", 4: "quadruples"}


@dataclass(frozen=True)
class Task:
    question: str
    answer: int
    level: int


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


def make_eval_set(per_level: int = 20, seed: int = 1234) -> list[Task]:
    """Fixed held-out set: the same questions every run, so before/after numbers compare like for like."""
    rng = random.Random(seed)
    return [make_task(level, rng) for level in LEVELS for _ in range(per_level)]


def make_train_task(level: int, rng: random.Random, exclude: set[str]) -> Task:
    """A fresh training task that is never one of the held-out questions."""
    while True:
        t = make_task(level, rng)
        if t.question not in exclude:
            return t
