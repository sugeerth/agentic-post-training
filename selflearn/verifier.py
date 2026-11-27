"""Default verifier: turns the agent's final text into a reward. Binary and exact; no reward model, no labels.
A use case can supply its own (see usecases/support.py, which checks an action and an amount)."""
from __future__ import annotations

import re

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.S)


def final_answer(text: str) -> str | None:
    m = ANSWER_RE.findall(text)
    return m[-1].strip() if m else None


def parse_answer(text: str) -> float | None:
    s = final_answer(text)
    if s is None:
        return None
    try:
        return float(s.replace(",", "").replace("$", "").rstrip("."))
    except ValueError:
        return None


def reward(text: str, answer: int | float | str) -> float:
    """1.0 if the last <answer> equals the true answer: numbers exactly (floats to the cent), text case-insensitively."""
    if isinstance(answer, str):
        got = final_answer(text)
        return 1.0 if got is not None and got.strip(" .'\"").lower() == answer.lower() else 0.0
    got = parse_answer(text)
    tol = 1e-6 if isinstance(answer, int) else 0.006
    return 1.0 if got is not None and abs(got - answer) < tol else 0.0
