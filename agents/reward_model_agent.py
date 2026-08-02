"""Reward-model agent — trains and serves the reward signal.

Two flavors matter for agentic post-training:

  • Outcome Reward Model (ORM) — one scalar for the whole trajectory.
    Cheap, sparse. Used by DeepSeek-R1's rule-based verifier.
  • Process Reward Model (PRM) — one scalar per step.
    Denser signal, needed for long-horizon tool use (Let's Verify Step-by-Step,
    Math-Shepherd, and Kimi K2's process feedback).

This agent owns the reward pipeline so the trainer never sees raw returns.
"""

from __future__ import annotations

import random
from typing import Any

from agents.base_agent import BaseAgent


class RewardModelAgent(BaseAgent):
    """Reward-model trainer & scorer.

    Fits an outcome or process reward model on curated trajectories, then
    scores held-out rollouts. This is the piece that decides *what the
    trainer chases*, so calibration matters more than raw accuracy.
    """

    def __init__(self, name: str = "RewardModel"):
        super().__init__(name, role="reward_model")
        self.kind: str = "outcome"
        self.accuracy: float = 0.0
        self.calibration_ece: float = 0.0
        self.register_capability("reward_modeling", "Fit an ORM or PRM")
        self.register_capability("scoring", "Score held-out trajectories")

    async def run(self, **kwargs) -> dict[str, Any]:
        self.kind = kwargs.get("reward_kind", "outcome")
        pairs = kwargs.get("preference_pairs", 4096)
        held_out = kwargs.get("held_out", 512)

        await self.send_message("status_update", {
            "message": f"Fitting {self.kind.upper()} reward model on {pairs} pairs",
        }, target="broadcast")

        # Simulated fit — outcome RMs are easier to train than process RMs.
        if self.kind == "outcome":
            self.accuracy = 0.78 + random.uniform(0, 0.08)
            self.calibration_ece = 0.06 + random.uniform(0, 0.03)
        else:
            self.accuracy = 0.71 + random.uniform(0, 0.06)
            self.calibration_ece = 0.09 + random.uniform(0, 0.04)

        result = {
            "reward_kind": self.kind,
            "pairs_seen": pairs,
            "held_out_accuracy": round(self.accuracy, 3),
            "expected_calibration_error": round(self.calibration_ece, 3),
            "scored_rollouts": held_out,
        }

        await self.send_message("task_result", {
            "message": (
                f"{self.kind.upper()} RM ready — "
                f"acc {result['held_out_accuracy']:.2f}, "
                f"ECE {result['expected_calibration_error']:.2f}"
            ),
            "metrics": result,
        }, target="Coordinator")

        return result

    async def step(self, **kwargs) -> dict[str, Any]:
        return await self.run(**kwargs)

    def score(self, trajectory_reward_signal: list[float]) -> float:
        """Score a single trajectory. Placeholder — real path calls the RM head."""
        if not trajectory_reward_signal:
            return 0.0
        return sum(trajectory_reward_signal) / len(trajectory_reward_signal)
