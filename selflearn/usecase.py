"""A use case = everything domain-specific: the prompt, the tools, and a task generator with difficulty levels.

The agents, the learner and the Coordinator never change when you add one (open for extension, closed for
modification). See selflearn/usecases/ for the two shipped: word problems with a calculator, and a data analyst
answering business questions with SQL.
"""
from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field

from selflearn.verifier import reward as exact_answer


@dataclass(frozen=True)
class Task:
    question: str
    answer: int | float | str
    level: int


@dataclass(frozen=True)
class UseCase:
    name: str
    system_prompt: str
    tools: dict[str, Callable[[str], str]]
    levels: tuple[int, ...]
    make_task: Callable[[int, random.Random], Task]
    reward: Callable[[str, int | float | str], float] = exact_answer   # verifier: (agent text, true answer) -> 0/1
    requires: tuple[str, ...] = ()   # tools that must run successfully in the episode for any reward (evidence rule)
    probes: dict[str, Callable] = field(default_factory=dict)   # scripted shortcut policies for the env audit
    user_prefix: str = "Question: "
    summary: str = ""          # one sentence for reports and the web demo
    level_note: str = ""       # what "level" means for this use case
    max_new: int = 96          # new tokens per turn
    eval_per_level: int = 20
    eval_seed: int = 1234

    def eval_set(self, per_level: int | None = None) -> list[Task]:
        """Fixed held-out set: the same distinct questions every run, so before/after compare like for like."""
        rng, n, out = random.Random(self.eval_seed), per_level or self.eval_per_level, []
        for level in self.levels:
            seen: set[str] = set()
            for _ in range(n * 50):
                if len(seen) == n:
                    break
                t = self.make_task(level, rng)
                if t.question not in seen:
                    seen.add(t.question)
                    out.append(t)
        return out

    def validation_set(self, per_level: int, exclude: set[str]) -> list[Task]:
        """A second fixed set, disjoint from the held-out test, for picking the best checkpoint during training."""
        rng, out = random.Random(self.eval_seed + 1), []
        for level in self.levels:
            seen: set[str] = set()
            for _ in range(per_level):
                t = self.train_task(level, rng, exclude | seen)
                seen.add(t.question)
                out.append(t)
        return out

    def train_task(self, level: int, rng: random.Random, exclude: set[str]) -> Task:
        """A fresh training task that is never one of the held-out questions."""
        while True:
            t = self.make_task(level, rng)
            if t.question not in exclude:
                return t
