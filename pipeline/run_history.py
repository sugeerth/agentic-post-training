"""Run history — persist each pipeline run to disk so the reporter can
compare against past runs of the same plan.

Design decisions:

  • File-per-run under `.runs/` (one JSON file). Cheap, greppable, no DB.
  • Filename includes plan + a monotonic counter (not a timestamp — the
    workflow sandbox blocks `Date.now()` in some tools, and tests want
    deterministic paths).
  • `latest_for(plan)` reads the newest completed run of that plan so
    the reporter can compute deltas against it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class RunRecord:
    plan: str
    n: int                                     # monotonic per-plan counter
    goal: str
    task_success_rate: float
    reward: float
    kl_divergence: float
    eval_overall: str                          # e.g. "+21.1%"
    elapsed_s: float
    interventions_needing_action: int
    path: Path

    def to_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        d["path"] = str(self.path)
        return d


class RunHistory:
    """Append-only per-plan history in a directory."""

    def __init__(self, root: str | Path = ".runs"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _list_for(self, plan: str) -> list[Path]:
        return sorted(self.root.glob(f"{plan}-*.json"))

    def latest_for(self, plan: str) -> RunRecord | None:
        files = self._list_for(plan)
        if not files:
            return None
        raw = json.loads(files[-1].read_text())
        raw["path"] = Path(raw["path"])
        return RunRecord(**raw)

    def append(self, run: dict[str, Any]) -> RunRecord:
        """Persist a run summary. Returns the RunRecord that was written."""
        plan = run.get("plan", "unknown")
        existing = self._list_for(plan)
        n = len(existing) + 1
        path = self.root / f"{plan}-{n:04d}.json"

        train_result = _find_train_result(run.get("results", {}))
        eval_result = run.get("results", {}).get("eval") or run.get("results", {}).get("evaluation") or {}
        needing = sum(
            1 for i in run.get("interventions", [])
            if isinstance(i, dict) and i.get("action") not in (None, "continue")
        )

        record = RunRecord(
            plan=plan,
            n=n,
            goal=run.get("goal", ""),
            task_success_rate=float(train_result.get("task_success_rate", 0.0)),
            reward=float(train_result.get("final_reward", 0.0)),
            kl_divergence=float(train_result.get("kl_divergence", 0.0)),
            eval_overall=str(eval_result.get("overall_improvement", "n/a")),
            elapsed_s=float(run.get("elapsed_s", 0.0)),
            interventions_needing_action=needing,
            path=path,
        )
        path.write_text(json.dumps(record.to_dict(), indent=2))
        return record

    def deltas(self, current: RunRecord, previous: RunRecord | None) -> dict[str, Any] | None:
        """Return {metric: (prev, curr, delta)} or None if no baseline."""
        if previous is None:
            return None
        return {
            "task_success_rate": (previous.task_success_rate, current.task_success_rate,
                                  round(current.task_success_rate - previous.task_success_rate, 3)),
            "reward":            (previous.reward, current.reward,
                                  round(current.reward - previous.reward, 3)),
            "kl_divergence":     (previous.kl_divergence, current.kl_divergence,
                                  round(current.kl_divergence - previous.kl_divergence, 3)),
            "elapsed_s":         (previous.elapsed_s, current.elapsed_s,
                                  round(current.elapsed_s - previous.elapsed_s, 2)),
        }


def _find_train_result(results: dict[str, Any]) -> dict[str, Any]:
    for stage in ("multi_turn_grpo", "grpo", "dpo", "rft", "training"):
        r = results.get(stage)
        if isinstance(r, dict) and ("task_success_rate" in r or "final_reward" in r):
            return r
    for r in results.values():
        if isinstance(r, dict) and ("task_success_rate" in r or "final_reward" in r):
            return r
    return {}
