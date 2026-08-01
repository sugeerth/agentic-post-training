"""Trajectory agent — samples, filters, and curates multi-turn agent rollouts.

Post-training an *agent* (not a base LLM) hinges on trajectory quality:

  • long-horizon tool-use rollouts are sparse and noisy
  • most rollouts fail — the reward signal is heavily imbalanced
  • credit assignment across turns is the hard problem

This agent owns rollout collection + rejection sampling (STaR / RFT-style)
so downstream trainers see a curated dataset instead of raw noise.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from agents.base_agent import BaseAgent


@dataclass
class Turn:
    role: str          # "user" | "assistant" | "tool"
    content: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Trajectory:
    task_id: str
    turns: list[Turn]
    outcome_reward: float             # terminal: task succeeded?
    process_rewards: list[float]      # per-turn shaped reward
    success: bool
    tokens: int


class TrajectoryAgent(BaseAgent):
    """Rollout generator + rejection sampler for agent trajectories."""

    def __init__(self, name: str = "TrajectoryCurator"):
        super().__init__(name, role="trajectory")
        self.pool: list[Trajectory] = []
        self.register_capability("rollout", "Sample multi-turn agent trajectories")
        self.register_capability("rejection_sampling", "Keep only high-quality rollouts")
        self.register_capability("curation", "Balance and dedupe the training set")

    def _sample_one(self, task_id: str, target_success_rate: float) -> Trajectory:
        """Simulate a rollout. Real deployments hand a live agent+env in here."""
        turn_count = random.randint(2, 8)
        success = random.random() < target_success_rate
        turns = []
        process = []
        for i in range(turn_count):
            role = "assistant" if i % 2 == 0 else "tool"
            turns.append(Turn(role=role, content=f"turn-{i}"))
            process.append(random.uniform(-0.1, 0.3))
        outcome = 1.0 if success else 0.0
        return Trajectory(
            task_id=task_id,
            turns=turns,
            outcome_reward=outcome,
            process_rewards=process,
            success=success,
            tokens=turn_count * 80,
        )

    async def run(self, **kwargs) -> dict[str, Any]:
        n = kwargs.get("num_rollouts", 64)
        target_rate = kwargs.get("initial_success_rate", 0.35)
        keep_top_frac = kwargs.get("keep_top_frac", 0.5)

        await self.send_message("status_update", {
            "message": f"Sampling {n} rollouts (baseline success ≈ {target_rate:.0%})",
        }, target="broadcast")

        self.pool = [self._sample_one(f"task-{i}", target_rate) for i in range(n)]
        successes = [t for t in self.pool if t.success]

        # Rejection sampling: keep the successful trajectories, then top-up
        # with best-effort failed ones (by highest process reward sum) so the
        # curated set doesn't collapse to zero on early iterations.
        keep_n = max(1, int(len(self.pool) * keep_top_frac))
        by_quality = sorted(
            self.pool,
            key=lambda t: (t.success, sum(t.process_rewards)),
            reverse=True,
        )
        curated = by_quality[:keep_n]

        result = {
            "sampled": len(self.pool),
            "success_rate": round(len(successes) / len(self.pool), 3),
            "curated": len(curated),
            "curated_success_rate": round(
                sum(t.success for t in curated) / len(curated), 3
            ),
            "avg_turns": round(sum(len(t.turns) for t in curated) / len(curated), 1),
            "avg_tokens_per_rollout": round(
                sum(t.tokens for t in curated) / len(curated), 1
            ),
        }

        await self.send_message("task_result", {
            "message": (
                f"Curated {result['curated']}/{result['sampled']} rollouts "
                f"(success {result['curated_success_rate']:.0%})"
            ),
            "metrics": result,
        }, target="Coordinator")

        return result

    async def step(self, **kwargs) -> dict[str, Any]:
        return await self.run(**kwargs)
