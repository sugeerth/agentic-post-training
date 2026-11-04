"""JSON summary of experiments, for dashboards and the docs site. Presentation only.

python -m evals run experiments/*.yaml --summary docs/data/experiments.json
"""

from __future__ import annotations

import sys
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from evals.envs import ENVIRONMENTS
from evals.failures import failure_counts
from evals.metrics import bootstrap_ci, brier, ece, mean
from evals.types import Grade

if TYPE_CHECKING:
    from evals.experiment import ExperimentResult


def _arm(grades: list[Grade]) -> dict[str, Any]:
    scores = [g.score for g in grades]
    lo, hi = bootstrap_ci(scores)
    out: dict[str, Any] = {
        "mean": mean(scores),
        "ci": [lo, hi],
        "failures": dict(failure_counts(grades).most_common()),
    }
    scored = [g for g in grades if "confidence" in g.details]
    if scored:
        conf = [float(g.details["confidence"]) for g in scored]
        ok = [g.passed for g in scored]
        out["calibration"] = {"ece": ece(conf, ok), "brier": brier(conf, ok)}
    slices: dict[str, list[float]] = defaultdict(list)
    for g in grades:
        for tag in g.tags:
            slices[tag].append(g.score)
    out["slices"] = {tag: {"mean": mean(v), "n": len(v)} for tag, v in sorted(slices.items())}
    return out


def summarize(result: ExperimentResult) -> dict[str, Any]:
    cfg, dec = result.config, result.decision
    envs = []
    for d in dec.deltas:
        envs.append(
            {
                "env": d.env,
                "target": d.env in cfg.targets,
                "delta": d.delta,
                "delta_ci": [d.ci_low, d.ci_high],
                **{
                    arm: _arm([g for g in result.grades[arm] if g.env == d.env])
                    for arm in ("baseline", "candidate")
                },
            }
        )
    return {
        "name": cfg.name,
        "hypothesis": cfg.hypothesis,
        "targets": cfg.targets,
        "seeds": cfg.seeds,
        "gate": cfg.gate.model_dump(),
        "verdict": dec.verdict,
        "reasons": list(dec.reasons),
        "episodes_per_arm": len(result.grades["baseline"]),
        "envs": envs,
    }


def env_catalog() -> list[dict[str, Any]]:
    out = []
    for name, cls in sorted(ENVIRONMENTS.items()):
        tasks = cls().tasks()
        doc = (sys.modules[cls.__module__].__doc__ or "").strip()
        out.append(
            {
                "name": name,
                "summary": doc.splitlines()[0] if doc else "",
                "n_tasks": len(tasks),
                "example": tasks[0].prompt,
                "tools": [t.name for t in tasks[0].tools],
            }
        )
    return out
