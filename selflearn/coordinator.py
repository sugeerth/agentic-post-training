"""Coordinator: runs the self-learning loop. Depends only on these role interfaces, never on concrete agents,
so any role can be swapped (e.g. a different Learner) without touching the loop. Imports no ML library.

  assess -> [designer reviews the env] -> calibrate
         -> [propose -> designer shapes -> act -> verify -> observe -> designer watches -> learn] x steps -> re-assess
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


class Designer(Protocol):
    def review(self, baseline: list) -> bool: ...
    def shape(self, tasks: list) -> list: ...
    def observe(self, trajs: list) -> None: ...


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
    designer: Designer | None = None   # optional: audit the environment first, reshape its tasks while training

    def run(self, steps: int, tasks_per_step: int, group: int, temp: float,
            on_step: Callable[[int, dict], None] | None = None) -> dict:
        before, before_rec = self.assessor.assess("base")
        if self.designer and not self.designer.review(before_rec):
            return {"before": before, "before_records": before_rec, "blocked": True}
        self.proposer.calibrate(before["by_level"])
        if self.selector:
            self.selector.consider(0)
        for step in range(1, steps + 1):
            tasks = self.proposer.propose(tasks_per_step)
            if self.designer:
                tasks = self.designer.shape(tasks)
            trajs = self.actor.act(tasks, group, temp)
            reward = self.verifier.score(trajs)
            self.proposer.observe(trajs)
            if self.designer:
                self.designer.observe(trajs)
            stats = self.learner.learn(trajs)
            if on_step:
                on_step(step, {"step": step, "reward": reward, "levels": [t.level for t in tasks], **stats})
            if self.selector:
                self.selector.consider(step)
        selected = self.selector.restore_best() if self.selector else steps
        after, after_rec = self.assessor.assess("trained")
        return {"before": before, "after": after, "before_records": before_rec, "after_records": after_rec,
                "selected_step": selected, "blocked": False}
