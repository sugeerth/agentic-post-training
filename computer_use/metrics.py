"""Benchmark statistics for GUI agents.

Two numbers do most of the work, and both exist because a single success rate
over a handful of episodes is a badly misleading summary of a stochastic agent.

**pass@k** — the probability that at least one of `k` attempts succeeds. It is
the number that matters whenever a retry is cheap (data collection, an agent
that can check its own work) and it is *not* the naive "did any of my `n`
samples pass", which is biased upward. The unbiased estimator from Chen et al.
2021 is used here: sample `n`, count `c`, estimate for any `k ≤ n`. The gap
between pass@1 and pass@8 is the most informative thing a GUI benchmark
reports — a large gap means the agent knows how but is unreliable, a small one
means it doesn't know how.

**Wilson score interval** — a confidence interval for the success rate that
stays inside [0, 1] and behaves at the extremes, where the normal
approximation returns nonsense like "83% ± 40%" or an interval above 1.0 for a
clean sweep. Benchmarks run 8 episodes, not 800, so the interval is usually
more honest than the point estimate.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from computer_use.types import Trajectory, TrajectoryStatus


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k: `n` attempts, `c` successes, budget `k`.

    Computed as `1 - C(n-c, k) / C(n, k)` in a numerically stable product form
    rather than with factorials, which overflow well before `n` gets
    interesting.
    """
    if k <= 0 or n <= 0:
        return 0.0
    if k > n:
        raise ValueError(f"pass@{k} needs at least {k} attempts, got n={n}")
    if c < 0 or c > n:
        raise ValueError(f"successes must be in [0, {n}], got {c}")
    if n - c < k:
        return 1.0
    return 1.0 - math.prod(1.0 - k / i for i in range(n - c + 1, n + 1))


def wilson_interval(c: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion. Default is 95%."""
    if n <= 0:
        return (0.0, 0.0)
    p = c / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return (max(0.0, center - margin), min(1.0, center + margin))


@dataclass(frozen=True)
class TaskResult:
    """Outcome of running one task `attempts` times."""

    name: str
    difficulty: str
    attempts: int
    successes: int
    mean_reward: float
    mean_steps: float
    mean_grounding: float
    pass_at: Mapping[int, float] = field(default_factory=dict)
    ci: tuple[float, float] = (0.0, 0.0)
    statuses: Mapping[str, int] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        return self.successes / self.attempts if self.attempts else 0.0

    @property
    def reliability_gap(self) -> float:
        """pass@k − pass@1 at the largest k measured.

        How much the agent gains from being allowed to retry. A large gap is
        the signature of an agent that knows the task but executes it
        unreliably — usually a grounding problem, not a planning one.
        """
        if not self.pass_at:
            return 0.0
        return self.pass_at[max(self.pass_at)] - self.pass_at.get(1, 0.0)


@dataclass(frozen=True)
class BenchmarkReport:
    """Suite-level roll-up over per-task results."""

    tasks: tuple[TaskResult, ...]
    pass_at: Mapping[int, float] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def episodes(self) -> int:
        return sum(t.attempts for t in self.tasks)

    @property
    def successes(self) -> int:
        return sum(t.successes for t in self.tasks)

    @property
    def success_rate(self) -> float:
        return self.successes / self.episodes if self.episodes else 0.0

    @property
    def ci(self) -> tuple[float, float]:
        return wilson_interval(self.successes, self.episodes)

    @property
    def mean_reward(self) -> float:
        if not self.tasks:
            return 0.0
        total = sum(t.mean_reward * t.attempts for t in self.tasks)
        return total / self.episodes if self.episodes else 0.0

    def by_difficulty(self) -> dict[str, dict[str, float]]:
        """Success rate per difficulty tier — where an agent breaks down."""
        buckets: dict[str, list[TaskResult]] = {}
        for task in self.tasks:
            buckets.setdefault(task.difficulty, []).append(task)
        out: dict[str, dict[str, float]] = {}
        for tier, results in buckets.items():
            attempts = sum(r.attempts for r in results)
            successes = sum(r.successes for r in results)
            out[tier] = {
                "tasks": len(results),
                "episodes": attempts,
                "success_rate": successes / attempts if attempts else 0.0,
            }
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "episodes": self.episodes,
            "success_rate": self.success_rate,
            "ci95": list(self.ci),
            "mean_reward": self.mean_reward,
            "pass_at": {str(k): v for k, v in self.pass_at.items()},
            "by_difficulty": self.by_difficulty(),
            "tasks": [
                {
                    "name": t.name,
                    "difficulty": t.difficulty,
                    "attempts": t.attempts,
                    "successes": t.successes,
                    "success_rate": t.success_rate,
                    "mean_reward": t.mean_reward,
                    "mean_steps": t.mean_steps,
                    "mean_grounding": t.mean_grounding,
                    "pass_at": {str(k): v for k, v in t.pass_at.items()},
                    "ci95": list(t.ci),
                    "statuses": dict(t.statuses),
                }
                for t in self.tasks
            ],
            **dict(self.metadata),
        }


def summarize_task(
    name: str,
    difficulty: str,
    trajectories: Sequence[Trajectory],
    *,
    k_values: Sequence[int] = (1, 2, 4, 8),
) -> TaskResult:
    """Roll a group of attempts at one task into a `TaskResult`."""
    n = len(trajectories)
    successes = sum(1 for t in trajectories if t.succeeded)

    grounding = [
        t.metadata.get("stats", {}).get("grounding_rate", 0.0) for t in trajectories
    ]

    return TaskResult(
        name=name,
        difficulty=difficulty,
        attempts=n,
        successes=successes,
        mean_reward=sum(t.reward for t in trajectories) / n if n else 0.0,
        mean_steps=sum(t.num_steps for t in trajectories) / n if n else 0.0,
        mean_grounding=sum(grounding) / n if n else 0.0,
        # Only report k values the sample can actually support; pass@8 from
        # 4 attempts is not a number, it is a guess.
        pass_at={k: pass_at_k(n, successes, k) for k in k_values if k <= n},
        ci=wilson_interval(successes, n),
        statuses={
            status.value: sum(1 for t in trajectories if t.status is status)
            for status in TrajectoryStatus
            if any(t.status is status for t in trajectories)
        },
    )


def summarize(
    results: Sequence[TaskResult], **metadata: Any
) -> BenchmarkReport:
    """Roll per-task results into a suite report.

    Suite-level pass@k is the mean of the per-task values, not pass@k over the
    pooled episodes — pooling would let a reliable easy task compensate for a
    hard one the agent never solves, which is exactly the failure a benchmark
    is supposed to surface.
    """
    if not results:
        return BenchmarkReport(tasks=(), metadata=metadata)

    shared_k = sorted(set.intersection(*(set(r.pass_at) for r in results)))
    pooled = {
        k: sum(r.pass_at[k] for r in results) / len(results) for k in shared_k
    }
    return BenchmarkReport(tasks=tuple(results), pass_at=pooled, metadata=metadata)
