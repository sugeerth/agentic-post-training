"""Verifier: turns the agent's final text into a reward. Binary and exact; no reward model, no human labels."""
from __future__ import annotations

import re

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.S)


def parse_answer(text: str) -> float | None:
    m = ANSWER_RE.findall(text)
    if not m:
        return None
    s = m[-1].strip().replace(",", "").replace("$", "").rstrip(".")
    try:
        return float(s)
    except ValueError:
        return None


def reward(text: str, answer: int) -> float:
    """1.0 if the last <answer> equals the true answer, else 0.0."""
    got = parse_answer(text)
    return 1.0 if got is not None and abs(got - answer) < 1e-6 else 0.0
