"""Reward-hacking detector — the biggest failure mode of agentic RL.

Symptom: the reward-model score climbs monotonically while the eval
score plateaus or drops. The policy has learned to exploit the RM, not
solve the task. Prior art: OpenAI's "reward hacking" write-ups,
Anthropic's "Discovering Language Model Behaviors with Model-Written
Evaluations", Kimi K2's process-reward calibration section.

Detection heuristic (simple, robust, and easy to explain):

  1. Compute rank correlation (Spearman ρ) between per-iteration
     reward and per-iteration eval score.
  2. Compute the gap: (final reward − initial reward) − (final eval −
     initial eval), both normalized to [0,1].

  ρ < 0 OR gap > threshold  →  hacking suspected.

This is intentionally not learned — a learned detector is another RM
that can itself be gamed. This one is auditable in ten lines.
"""

from __future__ import annotations

from typing import Any

from agents.base_agent import BaseAgent


def spearman_rho(xs: list[float], ys: list[float]) -> float:
    """Spearman rank correlation. Returns 0 for constant / len<2 inputs."""
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    rx = _ranks(xs)
    ry = _ranks(ys)
    return _pearson(rx, ry)


def _ranks(vs: list[float]) -> list[float]:
    """Fractional ranks (average tied positions)."""
    order = sorted(range(len(vs)), key=lambda i: vs[i])
    ranks = [0.0] * len(vs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vs[order[j + 1]] == vs[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


class RewardHackingDetector(BaseAgent):
    """Watches reward↔eval alignment across iterations of a training run."""

    def __init__(self, name: str = "HackingWatch"):
        super().__init__(name, role="reward_hacking_detector")
        self.register_capability("audit", "Rank-correlate reward vs eval")
        self.register_capability("alert", "Flag reward hacking with severity")

    def audit(
        self,
        rewards: list[float],
        evals: list[float],
        gap_threshold: float = 0.25,
    ) -> dict[str, Any]:
        rho = spearman_rho(rewards, evals)
        if not rewards or not evals:
            gap = 0.0
        else:
            gap = (rewards[-1] - rewards[0]) - (evals[-1] - evals[0])

        # High: anti-correlation, or the reward climbed way past the eval.
        # Medium: weakly correlated OR gap-and-not-fully-aligned. A perfectly
        # aligned (ρ ≈ 1) monotone run is NOT medium — reward climbing faster
        # than eval is fine when both are climbing in lockstep.
        if rho < -0.1 or gap > gap_threshold:
            severity = "high"
            verdict = "reward hacking suspected"
        elif rho < 0.3 or (gap > gap_threshold / 2 and rho < 0.9):
            severity = "medium"
            verdict = "reward/eval drift — watch"
        else:
            severity = "low"
            verdict = "reward/eval aligned"

        return {
            "spearman_rho": round(rho, 3),
            "reward_eval_gap": round(gap, 3),
            "severity": severity,
            "verdict": verdict,
            "n_iterations": len(rewards),
        }

    async def run(self, **kwargs) -> dict[str, Any]:
        rewards: list[float] = kwargs.get("rewards", [])
        evals: list[float] = kwargs.get("evals", [])
        gap_threshold: float = kwargs.get("gap_threshold", 0.25)

        report = self.audit(rewards, evals, gap_threshold=gap_threshold)
        await self.send_message("task_result", {
            "message": (
                f"{report['verdict']} (ρ={report['spearman_rho']}, "
                f"gap={report['reward_eval_gap']}, severity={report['severity']})"
            ),
            "metrics": report,
        }, target="Coordinator")
        return report

    async def step(self, **kwargs) -> dict[str, Any]:
        return await self.run(**kwargs)
