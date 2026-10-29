"""Bake-off pipeline — run N recipes in parallel, pick the winner.

Thin wrapper around `BakeoffAgent`: creates fresh `AgenticPipeline`s per
plan so histories don't cross-contaminate and each plan gets its own
`.runs/` record. Winner + pareto frontier are printed and returned.
"""

from __future__ import annotations

import time
from typing import Any

from agents.base_agent import BOLD, RESET
from agents.bakeoff_agent import BakeoffAgent
from pipeline.agentic_pipeline import AgenticPipeline


class BakeoffPipeline:
    def __init__(self, verbose: bool = False, out_dir: str = "./output/bakeoff",
                 history_dir: str = ".runs") -> None:
        self.verbose = verbose
        self.out_dir = out_dir
        self.history_dir = history_dir
        self.bakeoff = BakeoffAgent()

    def _pipeline_factory(self, plan_slug: str):
        def _make() -> AgenticPipeline:
            return AgenticPipeline(
                verbose=self.verbose,
                out_dir=f"{self.out_dir}/{plan_slug}",
                history_dir=self.history_dir,
            )
        return _make

    async def run(
        self,
        plans: list[str] | None = None,
        config: dict[str, Any] | None = None,
        kl_ceiling: float = 0.2,
    ) -> dict[str, Any]:
        plans = plans or ["agentic-tool-use", "reasoning-r1", "rejection-sampling-loop"]

        print(f"\n{BOLD}{'╔' + '═' * 68 + '╗'}")
        print(f"║{'BAKE-OFF · N recipes, one winner':^68s}║")
        print(f"{'╚' + '═' * 68 + '╝'}{RESET}")
        for p in plans:
            print(f"  · {p}")
        print()

        t0 = time.time()

        # Sequential runs (each plan is one AgenticPipeline). Genuinely
        # parallel plan runs would need per-pipeline stdout isolation,
        # which is out of scope for this file — the bake-off's value is
        # in the head-to-head comparison, not the wall-clock speedup.
        results_by_plan: dict[str, Any] = {}
        entries = []
        for plan in plans:
            pipe = self._pipeline_factory(plan)()
            run = await pipe.run(goal=_plan_to_goal(plan), config=config or {})
            results_by_plan[plan] = run
            entries.append(_entry_from(run, plan))

        winner = self.bakeoff.pick_winner(entries, kl_ceiling=kl_ceiling)
        winner.winner = True
        entries.sort(key=lambda e: -e.success_rate)
        self.bakeoff.print_table(entries)

        pareto = self.bakeoff.pareto_frontier(entries)
        summary = {
            "plans": [e.__dict__ for e in entries],
            "winner": winner.plan,
            "pareto_frontier": [e.plan for e in pareto],
            "runs": results_by_plan,
            "elapsed_s": round(time.time() - t0, 2),
        }
        return summary


def _plan_to_goal(plan: str) -> str:
    return {
        "agentic-tool-use":        "agentic tool use",
        "reasoning-r1":            "reasoning math",
        "preference-alignment":    "preference alignment",
        "rejection-sampling-loop": "rejection sampling",
    }.get(plan, plan)


def _entry_from(run: dict[str, Any], plan: str):
    from agents.bakeoff_agent import BakeoffEntry
    results = run.get("results", {})
    for stage in ("multi_turn_grpo", "grpo", "dpo", "rft", "training"):
        r = results.get(stage)
        if isinstance(r, dict) and "final_reward" in r:
            return BakeoffEntry(
                plan=plan,
                success_rate=float(r.get("task_success_rate", 0.0)),
                reward=float(r.get("final_reward", 0.0)),
                kl_divergence=float(r.get("kl_divergence", 0.0)),
                elapsed_s=float(run.get("elapsed_s", 0.0)),
            )
    return BakeoffEntry(plan=plan, success_rate=0.0, reward=0.0, kl_divergence=0.0,
                        elapsed_s=float(run.get("elapsed_s", 0.0)))
