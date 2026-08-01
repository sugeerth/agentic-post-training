"""Reporter agent — human-friendly, attention-budgeted TL;DR.

Design bias: humans skim before they read.

  • ≤ 6 lines above the fold
  • one number per line
  • lead with the delta, not the value
  • hide the log unless something anomalous happened
  • sparklines for curves so humans see the shape, not the numbers
  • compare against last successful run of the same plan
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agents.base_agent import BaseAgent, BOLD, RESET, DIM
from pipeline.run_history import RunHistory


# 8-level block characters — the whole spectrum for a sparkline.
_SPARK_CHARS = " ▁▂▃▄▅▆▇█"


def sparkline(values: list[float]) -> str:
    """One-line unicode sparkline. Auto-scales to the value range.

    Handles: empty input (returns ''), constant input (returns a flat mid line).
    Non-finite values are treated as the local minimum.
    """
    xs = [v for v in values if isinstance(v, (int, float))]
    if not xs:
        return ""
    lo, hi = min(xs), max(xs)
    if hi - lo < 1e-9:
        return _SPARK_CHARS[4] * len(xs)
    n = len(_SPARK_CHARS) - 1
    return "".join(_SPARK_CHARS[int((v - lo) / (hi - lo) * n)] for v in xs)


class ReporterAgent(BaseAgent):
    """Turns a raw pipeline result dict into a scannable summary."""

    def __init__(self, name: str = "Reporter", history: RunHistory | None = None):
        super().__init__(name, role="reporter")
        self.history = history or RunHistory()
        self.register_capability("summarize", "Condense results into a TL;DR")
        self.register_capability("surface_anomalies", "Show only what needs attention")
        self.register_capability("regression_check", "Compare against last run of the plan")

    async def run(self, **kwargs) -> dict[str, Any]:
        run = kwargs.get("run", {})
        out_dir = Path(kwargs.get("out_dir", "./output"))
        persist = kwargs.get("persist", True)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Persist first so the delta on THIS report is against the
        # previous run, not against itself.
        previous = self.history.latest_for(run.get("plan", "unknown")) if persist else None
        current = self.history.append(run) if persist else None
        deltas = self.history.deltas(current, previous) if (current and previous) else None

        tldr = self._compose_tldr(run, deltas)
        json_path = out_dir / "report.json"
        md_path = out_dir / "report.md"
        json_path.write_text(json.dumps(_slim(run), indent=2))
        md_path.write_text(tldr)

        self._print(tldr)

        await self.send_message("task_result", {
            "message": f"Report ready: {md_path}",
            "path": str(md_path),
        }, target="Coordinator")

        return {
            "report_md": str(md_path),
            "report_json": str(json_path),
            "tldr": tldr,
            "history_record": current.to_dict() if current else None,
            "deltas": deltas,
        }

    async def step(self, **kwargs) -> dict[str, Any]:
        return await self.run(**kwargs)

    def _compose_tldr(self, run: dict[str, Any], deltas: dict[str, Any] | None) -> str:
        plan = run.get("plan", "unknown")
        interventions = run.get("interventions", [])
        results = run.get("results", {})

        headline = _headline(results)
        anomalies = [i for i in interventions if i.get("action") != "continue"]
        sparks = _sparklines(results)

        lines: list[str] = []
        lines.append(f"# TL;DR — {plan}")
        lines.append("")
        lines.append(f"**{headline}**")
        lines.append("")
        lines.append(_key_metrics_line(results))
        lines.append("")

        if sparks:
            lines.append("## Curves")
            for label, spark in sparks:
                lines.append(f"- `{label:8s}` {spark}")
            lines.append("")

        if deltas:
            lines.append("## vs last run")
            lines.append(_deltas_line(deltas))
            lines.append("")

        if anomalies:
            lines.append("## Needs attention")
            for a in anomalies[:3]:
                lines.append(f"- **{a['stage']}**: {a['reason']} → `{a['action']}`")
            lines.append("")
        else:
            lines.append("_All stages within expected envelope. No manual review needed._")
            lines.append("")

        lines.append("## Next action")
        lines.append(f"- {_next_action(results, anomalies, deltas)}")
        lines.append("")
        lines.append("<details><summary>Full stage metrics</summary>")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(_slim(run), indent=2))
        lines.append("```")
        lines.append("</details>")
        return "\n".join(lines)

    def _print(self, tldr: str) -> None:
        print(f"\n{BOLD}{'═' * 70}")
        print(f"  📝 TL;DR (human-friendly)")
        print(f"{'═' * 70}{RESET}")
        for line in tldr.splitlines():
            if line.startswith("#"):
                print(f"{BOLD}{line}{RESET}")
            elif line.startswith("<details") or line.startswith("```") or line.startswith("</details"):
                print(f"{DIM}{line}{RESET}")
            else:
                print(line)
        print(f"{BOLD}{'═' * 70}{RESET}\n")


def _headline(results: dict[str, Any]) -> str:
    """One sentence, delta-first."""
    for name in ("multi_turn_grpo", "grpo", "training", "dpo", "rft"):
        r = results.get(name)
        if isinstance(r, dict) and "task_success_rate" in r:
            lift = r.get("success_rate_lift", 0.0)
            return (
                f"Task success {r['task_success_rate']:.0%} "
                f"(Δ +{lift:.0%}) via {name}."
            )
    eval_r = results.get("eval") or results.get("evaluation")
    if isinstance(eval_r, dict):
        return f"Eval delta: {eval_r.get('overall_improvement', 'n/a')}."
    return "Pipeline finished."


def _key_metrics_line(results: dict[str, Any]) -> str:
    """One line of the ~5 most decision-relevant scalars, deduped."""
    seen: set[str] = set()
    parts: list[str] = []

    def add(key: str, formatted: str) -> None:
        if key in seen:
            return
        seen.add(key)
        parts.append(formatted)

    for _stage, r in results.items():
        if not isinstance(r, dict):
            continue
        if "curated" in r and "sampled" in r:
            add("curated", f"curated `{r['curated']}/{r['sampled']}`")
        if "held_out_accuracy" in r:
            add("rm", f"rm-acc `{r['held_out_accuracy']:.2f}`")
        if "final_reward" in r:
            add("reward", f"reward `{r['final_reward']:.2f}`")
        if "task_success_rate" in r:
            add("success", f"success `{r['task_success_rate']:.0%}`")
        if "overall_improvement" in r:
            add("eval", f"eval `{r['overall_improvement']}`")
    return " · ".join(parts[:5]) if parts else "_no scalar metrics reported_"


def _sparklines(results: dict[str, Any]) -> list[tuple[str, str]]:
    """Extract iteration_metrics from the training stage and render sparklines."""
    for stage in ("multi_turn_grpo", "grpo", "dpo", "rft", "training"):
        r = results.get(stage)
        if not isinstance(r, dict):
            continue
        iters = r.get("iteration_metrics")
        if not isinstance(iters, list) or not iters:
            continue
        cols = [
            ("reward", [it.get("reward", 0.0) for it in iters]),
            ("loss",   [it.get("loss",   0.0) for it in iters]),
            ("success", [it.get("success_rate", 0.0) for it in iters]),
            ("kl",     [it.get("kl_divergence", 0.0) for it in iters]),
        ]
        return [(label, sparkline(vals)) for label, vals in cols if vals and any(vals)]
    return []


def _deltas_line(deltas: dict[str, Any]) -> str:
    def fmt(name: str, unit: str = "", pct: bool = False) -> str:
        prev, curr, d = deltas[name]
        if pct:
            return f"{name.split('_')[0]} `{curr:.0%}` (Δ {d:+.1%})"
        return f"{name.split('_')[0]} `{curr:.2f}{unit}` (Δ {d:+.2f})"
    parts = []
    if "task_success_rate" in deltas:
        parts.append(fmt("task_success_rate", pct=True))
    if "reward" in deltas:
        parts.append(fmt("reward"))
    if "kl_divergence" in deltas:
        parts.append(fmt("kl_divergence"))
    if "elapsed_s" in deltas:
        parts.append(fmt("elapsed_s", unit="s"))
    return " · ".join(parts)


def _next_action(results: dict[str, Any], anomalies: list[dict], deltas: dict[str, Any] | None) -> str:
    if anomalies:
        first = anomalies[0]
        return f"Address `{first['stage']}` — {first['reason']}."
    # Regression check on success_rate — highest-signal negative delta.
    if deltas and "task_success_rate" in deltas:
        _, _, d = deltas["task_success_rate"]
        if d < -0.05:
            return f"Success rate regressed by {abs(d):.0%} vs last run — revert or investigate."
    for _stage, r in results.items():
        if isinstance(r, dict) and r.get("task_success_rate", 1.0) < 0.6:
            return "Success rate under 60% — schedule another RL iteration."
    return "Ship the checkpoint. Nothing else to tune."


def _slim(obj: Any, depth: int = 0) -> Any:
    """Recursively drop long lists so the JSON report stays scannable."""
    if depth > 6:
        return "…"
    if isinstance(obj, dict):
        return {k: _slim(v, depth + 1) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_slim(x, depth + 1) for x in obj[:5]] + (
            [f"…(+{len(obj) - 5} more)"] if len(obj) > 5 else []
        )
    if isinstance(obj, float):
        return round(obj, 4)
    return obj
