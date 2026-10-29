"""Bake-off agent — runs multiple recipes concurrently, picks a winner.

Post-training rarely has one obviously-right recipe. A supervised bake-off
(DPO vs GRPO vs RFT, or three GRPO configs) is often the cheapest way to
find out which one works for *your* data.

Bake-offs are structured, not free-for-all:

  1. Same task pool, same eval, same wall-clock budget.
  2. Report the pareto frontier (success rate vs KL divergence) — a
     winner isn't just "best success", it's "best success at acceptable
     drift".
  3. Publish one row per plan so a human can eyeball the trade.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from agents.base_agent import BaseAgent, BOLD, RESET


@dataclass
class BakeoffEntry:
    plan: str
    success_rate: float
    reward: float
    kl_divergence: float
    elapsed_s: float
    winner: bool = False


class BakeoffAgent(BaseAgent):
    """Runs N pipelines concurrently and reports the pareto set + a winner.

    The `pipeline_factory` argument is a zero-arg callable that returns a
    fresh `AgenticPipeline`. Kept as an injected dep so tests can stub it.
    """

    def __init__(self, name: str = "Bakeoff"):
        super().__init__(name, role="bakeoff")
        self.register_capability("multi_plan", "Run several plans in parallel")
        self.register_capability("pareto", "Report success-vs-KL trade-offs")

    def pick_winner(self, entries: list[BakeoffEntry], kl_ceiling: float = 0.2) -> BakeoffEntry:
        """Highest success rate among plans that stay under kl_ceiling.
        Falls back to overall best if all plans blew past the ceiling."""
        safe = [e for e in entries if e.kl_divergence <= kl_ceiling]
        pool = safe or entries
        return max(pool, key=lambda e: e.success_rate)

    def pareto_frontier(self, entries: list[BakeoffEntry]) -> list[BakeoffEntry]:
        """Non-dominated by (success ↑, kl ↓)."""
        frontier: list[BakeoffEntry] = []
        for e in entries:
            dominated = any(
                (other.success_rate >= e.success_rate and other.kl_divergence <= e.kl_divergence)
                and (other.success_rate > e.success_rate or other.kl_divergence < e.kl_divergence)
                for other in entries if other is not e
            )
            if not dominated:
                frontier.append(e)
        return sorted(frontier, key=lambda e: -e.success_rate)

    def print_table(self, entries: list[BakeoffEntry]) -> None:
        print(f"\n{BOLD}{'═' * 70}")
        print(f"  🥊 Bake-off results")
        print(f"{'═' * 70}{RESET}")
        print(f"  {'':2s} {'plan':<28s} {'success':>8s} {'reward':>7s} {'kl':>6s} {'sec':>6s}")
        print(f"  {'─' * 66}")
        for e in entries:
            marker = "🏆" if e.winner else "  "
            print(f"  {marker} {e.plan:<28s} {e.success_rate:>7.0%} {e.reward:>7.2f} "
                  f"{e.kl_divergence:>6.2f} {e.elapsed_s:>6.1f}")
        print(f"{BOLD}{'═' * 70}{RESET}\n")

    async def run(self, **kwargs) -> dict[str, Any]:
        pipeline_factory = kwargs["pipeline_factory"]         # required
        plans: list[str] = kwargs.get("plans", ["agentic-tool-use", "rejection-sampling-loop"])
        config: dict[str, Any] = kwargs.get("config", {})
        kl_ceiling: float = kwargs.get("kl_ceiling", 0.2)

        await self.send_message("coordination", {
            "message": f"Bake-off: {len(plans)} plans in parallel — {', '.join(plans)}",
        }, target="broadcast")

        async def _run_one(plan_name: str) -> BakeoffEntry:
            import time as _time
            t0 = _time.time()
            pipe = pipeline_factory()
            run = await pipe.run(goal=_plan_to_goal(plan_name), config=config)
            r = _pick_train_result(run.get("results", {}))
            return BakeoffEntry(
                plan=plan_name,
                success_rate=float(r.get("task_success_rate", 0.0)),
                reward=float(r.get("final_reward", 0.0)),
                kl_divergence=float(r.get("kl_divergence", 0.0)),
                elapsed_s=round(_time.time() - t0, 2),
            )

        entries = await asyncio.gather(*[_run_one(p) for p in plans])
        entries = list(entries)
        winner = self.pick_winner(entries, kl_ceiling=kl_ceiling)
        winner.winner = True

        entries_sorted = sorted(entries, key=lambda e: -e.success_rate)
        self.print_table(entries_sorted)

        frontier = self.pareto_frontier(entries)
        report = {
            "plans": [e.__dict__ for e in entries_sorted],
            "winner": winner.plan,
            "pareto_frontier": [e.plan for e in frontier],
            "kl_ceiling": kl_ceiling,
        }

        await self.send_message("task_result", {
            "message": f"Winner: {winner.plan} (success {winner.success_rate:.0%}, KL {winner.kl_divergence:.2f})",
            "metrics": report,
        }, target="Coordinator")
        return report

    async def step(self, **kwargs) -> dict[str, Any]:
        return await self.run(**kwargs)


def _plan_to_goal(plan_name: str) -> str:
    return {
        "agentic-tool-use":       "agentic tool use",
        "reasoning-r1":           "reasoning math",
        "preference-alignment":   "preference alignment",
        "rejection-sampling-loop": "rejection sampling",
    }.get(plan_name, plan_name)


def _pick_train_result(results: dict[str, Any]) -> dict[str, Any]:
    for stage in ("multi_turn_grpo", "grpo", "dpo", "rft", "training"):
        r = results.get(stage)
        if isinstance(r, dict) and "final_reward" in r:
            return r
    for r in results.values():
        if isinstance(r, dict) and "final_reward" in r:
            return r
    return {}
