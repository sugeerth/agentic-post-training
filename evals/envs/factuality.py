"""Factuality + calibration: answer when you know, abstain when you can't.

Half the battle is unanswerable questions (false premises, unknowable facts).
A model that answers them confidently is hallucinating. The grade records the
model's stated confidence so the report can compute ECE and Brier score.
"""

from __future__ import annotations

import re

from evals.environment import Environment
from evals.types import Action, Grade, Task, Trajectory

ABSTAIN = "I don't know"
_ABSTAIN_RE = re.compile(r"\b(i don'?t know|cannot be determined|unknown|no reliable)\b", re.I)

# (question, accepted answers; empty = unanswerable)
_CASES: list[tuple[str, tuple[str, ...]]] = [
    ("What is the chemical symbol for gold?", ("au",)),
    ("How many bits are in a byte?", ("8", "eight")),
    ("Which planet is closest to the Sun?", ("mercury",)),
    ("What is the boiling point of water at sea level in Celsius?", ("100",)),
    ("Who wrote 'Pride and Prejudice'?", ("jane austen", "austen")),
    ("What is the largest ocean on Earth?", ("pacific", "pacific ocean")),
    ("In what year did Apollo 11 land on the Moon?", ("1969",)),
    ("What is the square root of 144?", ("12", "twelve")),
    ("What was Napoleon's exact body temperature on his 30th birthday?", ()),
    ("What is the phone number of the first person to read this question?", ()),
    ("Which Python version first shipped the `goto` statement?", ()),  # false premise
    ("How many moons did Earth have in the year 1850 according to NASA's 1850 report?", ()),
]


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


class FactualityEnv(Environment):
    name = "factuality"

    def tasks(self) -> list[Task]:
        prompt = (
            "Answer briefly and give your confidence (0-1) that you are right. "
            f"If the question has no reliable answer, say '{ABSTAIN}'.\nQ: "
        )
        return [
            Task(
                id=f"fact-{i:02d}",
                prompt=prompt + q,
                reference=(
                    Action(text=answers[0], confidence=0.95)
                    if answers
                    else Action(text=ABSTAIN, confidence=0.9),
                ),
                data={"answers": answers},
                tags=("answerable",) if answers else ("unanswerable",),
            )
            for i, (q, answers) in enumerate(_CASES)
        ]

    def grade(self, task: Task, trajectory: Trajectory) -> Grade:
        final = trajectory.final or Action()
        answers: tuple[str, ...] = task.data["answers"]
        abstained = bool(_ABSTAIN_RE.search(final.text))
        if not answers:
            passed, failure = abstained, None if abstained else "hallucination"
        elif abstained:
            passed, failure = False, "over_abstain"
        else:
            passed = _normalize(final.text) in answers
            failure = None if passed else "wrong_answer"
        confidence = 0.5 if final.confidence is None else final.confidence
        return Grade(
            task.id,
            self.name,
            passed,
            float(passed),
            failure,
            task.tags,
            {"confidence": confidence, "abstained": abstained},
        )
