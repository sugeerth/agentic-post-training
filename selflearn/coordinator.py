"""Coordinator: runs the self-learning loop. Depends only on these role interfaces, never on concrete agents,
so any role can be swapped (e.g. a different Learner) without touching the loop. Imports no ML library.

  assess -> calibrate -> [propose -> act -> verify -> observe -> learn] x steps -> re-assess
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol


class Assessor(Protocol):
    def assess(self, tag: str) -> tuple[dict, list[dict]]: ...


class Proposer(Protocol):
    def calibrate(self, success_by_level: dict[int, float]) -> None: ...
    def propose(self, n: int) -> list: ...
    def observe(self, trajs: list) -> None: ...


class Actor(Protocol):
    def act(self, tasks: list, group: int, temp: float) -> list: ...


class Verifier(Protocol):
    def score(self, trajs: list) -> float: ...


class Learner(Protocol):
    def learn(self, trajs: list) -> dict: ...


class Selector(Protocol):
    def consider(self, step: int) -> None: ...
    def restore_best(self) -> int: ...


@dataclass
class Coordinator:
    assessor: Assessor
    proposer: Proposer
    actor: Actor
    verifier: Verifier
    learner: Learner
    selector: Selector | None = None   # optional: keep the best checkpoint by validation, restore it at the end

    def run(self, steps: int, tasks_per_step: int, group: int, temp: float,
            on_step: Callable[[int, dict], None] | None = None) -> dict:
        before, before_rec = self.assessor.assess("base")
        self.proposer.calibrate(before["by_level"])
        if self.selector:
            self.selector.consider(0)
        for step in range(1, steps + 1):
            tasks = self.proposer.propose(tasks_per_step)
            trajs = self.actor.act(tasks, group, temp)
            reward = self.verifier.score(trajs)
            self.proposer.observe(trajs)
            stats = self.learner.learn(trajs)
            if on_step:
                on_step(step, {"step": step, "reward": reward, "levels": [t.level for t in tasks], **stats})
            if self.selector:
                self.selector.consider(step)
        selected = self.selector.restore_best() if self.selector else steps
        after, after_rec = self.assessor.assess("trained")
        return {"before": before, "after": after, "before_records": before_rec, "after_records": after_rec,
                "selected_step": selected}
