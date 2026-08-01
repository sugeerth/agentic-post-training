"""Trajectory agent — samples, filters, and curates multi-turn agent rollouts.

Post-training an *agent* (not a base LLM) hinges on trajectory quality:

  • long-horizon tool-use rollouts are sparse and noisy
  • most rollouts fail — the reward signal is heavily imbalanced
  • credit assignment across turns is the hard problem

This agent owns rollout collection + rejection sampling (STaR / RFT-style)
so downstream trainers see a curated dataset instead of raw noise.

It rolls out against an actual `ToolEnv` — no more fabricated trajectory
dicts. A policy-mixing schedule (`p_expert`) lets the agent simulate a
policy improving from random baseline toward expert play across iterations.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from agents.base_agent import BaseAgent
from environments.tool_env import ToolEnv, Task, sample_tasks
from environments.policies import POLICIES


@dataclass
class Turn:
    tool: str
    args: dict[str, Any]
    obs: str
    reward: float


@dataclass
class Trajectory:
    task_id: str
    turns: list[Turn]
    outcome_reward: float             # terminal: task succeeded?
    process_rewards: list[float]      # per-turn shaped reward
    success: bool
    tokens: int                       # rough proxy = sum of turn text lengths
    goal: str = ""


class TrajectoryAgent(BaseAgent):
    """Rollout generator + rejection sampler for agent trajectories."""

    def __init__(self, name: str = "TrajectoryCurator"):
        super().__init__(name, role="trajectory")
        self.pool: list[Trajectory] = []
        self._env: ToolEnv | None = None
        self.register_capability("rollout", "Sample multi-turn agent trajectories")
        self.register_capability("rejection_sampling", "Keep only high-quality rollouts")
        self.register_capability("curation", "Balance and dedupe the training set")

    def _rollout(self, task: Task, p_expert: float, rng: random.Random) -> Trajectory:
        """Roll one trajectory. Policy is a stochastic mix of random ↔ expert."""
        env = self._env or ToolEnv()
        goal = env.reset(task)
        turns: list[Turn] = []
        process: list[float] = []
        done = False
        while not done:
            policy = POLICIES["expert" if rng.random() < p_expert else "random"]
            tool, args = policy(goal, [t.__dict__ for t in turns], rng)
            r = env.step(tool, args)
            turns.append(Turn(tool=tool, args=dict(args), obs=r.obs, reward=r.reward))
            process.append(r.reward)
            done = r.done

        success = any(t.tool == "finish" and t.reward >= 1.0 for t in turns)
        tokens = sum(len(t.obs) + sum(len(str(v)) for v in t.args.values()) for t in turns)
        return Trajectory(
            task_id=task.task_id,
            turns=turns,
            outcome_reward=1.0 if success else 0.0,
            process_rewards=process,
            success=success,
            tokens=tokens,
            goal=goal,
        )

    async def run(self, **kwargs) -> dict[str, Any]:
        n = kwargs.get("num_rollouts", 64)
        p_expert = kwargs.get("p_expert", 0.35)
        keep_top_frac = kwargs.get("keep_top_frac", 0.5)
        seed = kwargs.get("seed", 0)

        rng = random.Random(seed)
        self._env = ToolEnv(sample_tasks(n=max(16, n), seed=seed))
        tasks = [rng.choice(self._env.tasks) for _ in range(n)]

        await self.send_message("status_update", {
            "message": f"Rolling {n} trajectories against ToolEnv (p_expert={p_expert:.2f})",
        }, target="broadcast")

        self.pool = [self._rollout(t, p_expert, rng) for t in tasks]
        successes = [t for t in self.pool if t.success]

        # Rejection sampling: keep successful rollouts, then top-up with
        # the best failed ones (highest process-reward sum) so the curated
        # set doesn't collapse to zero on early iterations.
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
            "p_expert": p_expert,
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
