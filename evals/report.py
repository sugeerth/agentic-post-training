"""Render an ExperimentResult as a Markdown report. Presentation only."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from evals.failures import failure_counts
from evals.metrics import bootstrap_ci, brier, ece, mean
from evals.types import Grade

if TYPE_CHECKING:
    from evals.experiment import ExperimentResult


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _by_env(grades: list[Grade]) -> dict[str, list[Grade]]:
    out: dict[str, list[Grade]] = defaultdict(list)
    for g in grades:
        out[g.env].append(g)
    return out


def _results_table(result: ExperimentResult) -> list[str]:
    base, cand = _by_env(result.grades["baseline"]), _by_env(result.grades["candidate"])
    rows = [
        "| env | baseline | candidate | Δ (95% CI) | target |",
        "|---|---|---|---|---|",
    ]
    for d in result.decision.deltas:
        b = [g.score for g in base[d.env]]
        c = [g.score for g in cand[d.env]]
        b_lo, b_hi = bootstrap_ci(b)
        c_lo, c_hi = bootstrap_ci(c)
        rows.append(
            f"| {d.env} | {_pct(mean(b))} [{_pct(b_lo)}, {_pct(b_hi)}] "
            f"| {_pct(mean(c))} [{_pct(c_lo)}, {_pct(c_hi)}] "
            f"| {d.delta:+.3f} [{d.ci_low:+.3f}, {d.ci_high:+.3f}] "
            f"| {'✓' if d.env in result.config.targets else ''} |"
        )
    return rows


def _failure_table(result: ExperimentResult) -> list[str]:
    base, cand = _by_env(result.grades["baseline"]), _by_env(result.grades["candidate"])
    rows = ["| env | failure | baseline | candidate |", "|---|---|---|---|"]
    for env in result.config.envs:
        b, c = failure_counts(base[env]), failure_counts(cand[env])
        for tag in sorted(set(b) | set(c), key=lambda t: -(b[t] + c[t])):
            rows.append(f"| {env} | `{tag}` | {b[tag]} | {c[tag]} |")
    return rows if len(rows) > 2 else ["No failures in either arm."]


def _calibration_table(result: ExperimentResult) -> list[str]:
    rows = ["| arm | env | ECE ↓ | Brier ↓ |", "|---|---|---|---|"]
    for arm, grades in result.grades.items():
        for env, gs in _by_env(grades).items():
            scored = [g for g in gs if "confidence" in g.details]
            if scored:
                conf = [float(g.details["confidence"]) for g in scored]
                ok = [g.passed for g in scored]
                rows.append(f"| {arm} | {env} | {ece(conf, ok):.3f} | {brier(conf, ok):.3f} |")
    return rows if len(rows) > 2 else []


def _slice_table(result: ExperimentResult) -> list[str]:
    def by_slice(grades: list[Grade]) -> dict[tuple[str, str], list[float]]:
        out: dict[tuple[str, str], list[float]] = defaultdict(list)
        for g in grades:
            for tag in g.tags:
                out[(g.env, tag)].append(g.score)
        return out

    base, cand = by_slice(result.grades["baseline"]), by_slice(result.grades["candidate"])
    rows = ["| env | slice | n | baseline | candidate |", "|---|---|---|---|---|"]
    for key in sorted(base):
        rows.append(
            f"| {key[0]} | `{key[1]}` | {len(base[key])} "
            f"| {_pct(mean(base[key]))} | {_pct(mean(cand[key]))} |"
        )
    return rows


def render_report(result: ExperimentResult, *, n_pairs: int) -> str:
    cfg, dec = result.config, result.decision
    lines = [
        f"# Experiment: {cfg.name}",
        "",
        f"**Hypothesis.** {cfg.hypothesis}",
        "",
        f"**Verdict: {dec.verdict}**",
        "",
        *[f"- {r}" for r in dec.reasons],
        "",
        f"Seeds: {cfg.seeds} · episodes per arm: {len(result.grades['baseline'])} · "
        f"gate: min_gain={cfg.gate.min_gain}, max_regression={cfg.gate.max_regression}",
        "",
        "## Results by environment",
        "",
        *_results_table(result),
        "",
        "## Failure modes (counts)",
        "",
        *_failure_table(result),
        "",
    ]
    calibration = _calibration_table(result)
    if calibration:
        lines += ["## Calibration", "", *calibration, ""]
    lines += [
        "## Slices",
        "",
        *_slice_table(result),
        "",
        "## Next round of training data",
        "",
        f"{n_pairs} preference pairs mined from candidate failures → `preference_pairs.jsonl` "
        "(chosen = reference, rejected = candidate output). Feed to DPO/ORPO/KTO.",
        "",
    ]
    return "\n".join(lines)
