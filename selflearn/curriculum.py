"""Self-paced curriculum: the agent decides what to practise next from its own success rates.

GRPO learns only from groups whose attempts disagree (some right, some wrong): if all G attempts score the same,
every advantage is zero and the group teaches nothing. With success rate p on a level, the chance that a group of
G attempts is informative is

    informative(p) = 1 - p**G - (1 - p)**G

which peaks at p = 0.5 and vanishes when the level is mastered (p -> 1) or hopeless (p -> 0). The curriculum samples
levels in proportion to that probability, so practice moves up the levels by itself as the agent improves.
Success rates start from the agent's own baseline test (it assesses itself first) and are updated from every group.
"""
from __future__ import annotations

import random

from selflearn.tasks import LEVELS


def informative(p: float, group: int) -> float:
    return 1.0 - p ** group - (1.0 - p) ** group


class Curriculum:
    def __init__(self, init: dict[int, float], group: int, ema: float = 0.3, floor: float = 0.03):
        self.p = {lv: float(init.get(lv, 0.5)) for lv in LEVELS}
        self.group, self.ema, self.floor = group, ema, floor

    def weights(self) -> dict[int, float]:
        w = {lv: informative(min(max(p, 0.02), 0.98), self.group) + self.floor for lv, p in self.p.items()}
        s = sum(w.values())
        return {lv: v / s for lv, v in w.items()}

    def sample(self, n: int, rng: random.Random) -> list[int]:
        w = self.weights()
        return rng.choices(list(w), weights=list(w.values()), k=n)

    def update(self, level: int, group_mean_reward: float) -> None:
        self.p[level] = (1 - self.ema) * self.p[level] + self.ema * group_mean_reward
