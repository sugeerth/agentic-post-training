"""MimicWriter — generates a stub technique file per scouted paper.

The generated file has the same shape as `techniques/agentic/*.py`:

    class MimicName:
        name = "slug"
        paper_reference = "https://arxiv.org/abs/…"
        def step(iteration: int) -> dict[str, float]: ...

The stub emits a deterministic decay curve tuned by two knobs pulled
from the paper's abstract:

  • `gain` — success-rate slope per iteration, biased by any
    "outperforms X by N%" claim we could parse. Bigger claim → higher gain.
  • `cap`  — asymptote. Set from the largest N% "X higher on TASK" claim
    when present, defaults to a middle-of-the-pack 0.80.

The mimic is NOT a working implementation of the paper. It's a fair
stand-in for bake-off comparisons and API testing until someone reads
the paper and writes a real one.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from agents.base_agent import BaseAgent


class MimicWriter(BaseAgent):
    def __init__(self, name: str = "MimicWriter", root: str = "techniques/mimics"):
        super().__init__(name, role="mimic_writer")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.register_capability("codegen", "Emit a technique stub file")

    def _class_name(self, technique_name: str) -> str:
        """Best-effort CamelCase from a possibly-hyphenated acronym.

        Preserves acronyms that are already all-caps (GRPO stays GRPO)
        but capitalizes mixed-case chunks (mt-agrpo → MtAgrpo).
        """
        parts = re.split(r"[-_]", technique_name)
        return "".join(p if p.isupper() else p.capitalize() for p in parts if p)

    def _tune(self, technique: dict[str, Any]) -> tuple[float, float]:
        """Pick (gain, cap) from parsed outperform / gain claims.

        Heuristic (deliberately simple):
          • largest gain %/points in the abstract sets the gain slope
            (clipped to a sane range so a paper claiming +99% doesn't
            get instantly ranked #1 in the bake-off).
          • largest "N% higher on X" bumps the cap toward 0.90.
        """
        outperforms = technique.get("outperforms", []) or []
        gains = technique.get("gains", []) or []

        biggest_perf = max((v for _, v in outperforms if v is not None), default=None)
        biggest_gain = max((v for v, _ in gains), default=None)

        # Baseline: match multi_turn_grpo's schedule so mimics are competitive
        # but not overpowered by unverified claims.
        gain = 0.10
        cap = 0.80
        if biggest_perf is not None:
            # +8 points → +0.03 slope; +20 points → +0.06 slope.
            gain = min(0.16, 0.10 + biggest_perf / 300.0)
        if biggest_gain is not None:
            cap = min(0.92, 0.80 + biggest_gain / 200.0)
        return round(gain, 3), round(cap, 3)

    def _render(self, technique: dict[str, Any]) -> str:
        cls = self._class_name(technique["name"])
        slug = technique["slug"]
        gain, cap = self._tune(technique)
        notes = []
        if technique.get("mentions_kl"):
            notes.append("KL-aware")
        if technique.get("mentions_process_reward"):
            notes.append("uses PRM")
        if technique.get("mentions_reward_hacking"):
            notes.append("addresses reward hacking")
        notes_str = " · ".join(notes) if notes else "no auto-detected KL/PRM/hack flags"

        return (
            '"""Auto-generated mimic of {name}.\n\n'
            'Paper: {title}\n'
            'Source: {url}\n'
            'Flags: {notes}\n\n'
            'This is a STUB — deterministic decay curve, not a real training loop.\n'
            'Replace with the actual algorithm when you have time to read the paper.\n'
            '"""\n\n'
            'from __future__ import annotations\n\n'
            'import math\n\n\n'
            'class {cls}:\n'
            '    name = "{slug}"\n'
            '    paper_reference = "{url}"\n'
            '    is_mimic = True\n\n'
            '    def __init__(self) -> None:\n'
            '        self._it = 0\n'
            '        self.gain = {gain}\n'
            '        self.cap = {cap}\n\n'
            '    def step(self, iteration: int | None = None) -> dict[str, float]:\n'
            '        self._it = iteration if iteration is not None else self._it + 1\n'
            '        loss = (1.7 - self.gain) * math.exp(-0.3 * self._it) + 0.28\n'
            '        reward = min(self.cap + 0.05, 0.30 + self.gain * 2 * self._it)\n'
            '        kl = max(0.0, (0.20 - self.gain) - 0.02 * self._it)\n'
            '        return {{\n'
            '            "loss": round(loss, 4),\n'
            '            "reward": round(reward, 4),\n'
            '            "kl_divergence": round(kl, 4),\n'
            '        }}\n'
        ).format(
            name=technique["name"],
            title=technique.get("paper_title", ""),
            url=technique.get("paper_url", ""),
            notes=notes_str,
            cls=cls,
            slug=slug,
            gain=gain,
            cap=cap,
        )

    def write(self, technique: dict[str, Any]) -> Path:
        path = self.root / f"{technique['slug']}.py"
        path.write_text(self._render(technique))
        return path

    async def run(self, **kwargs) -> dict[str, Any]:
        techniques: list[dict] = kwargs.get("techniques", [])
        overwrite: bool = kwargs.get("overwrite", False)

        written: list[str] = []
        skipped: list[str] = []
        for t in techniques:
            path = self.root / f"{t['slug']}.py"
            if path.exists() and not overwrite:
                skipped.append(t["slug"])
                continue
            self.write(t)
            written.append(t["slug"])

        await self.send_message("task_result", {
            "message": f"Wrote {len(written)} mimics (skipped {len(skipped)} existing)",
            "written": written,
            "skipped": skipped,
        }, target="Coordinator")

        return {
            "written": written,
            "skipped": skipped,
            "root": str(self.root),
        }

    async def step(self, **kwargs) -> dict[str, Any]:
        return await self.run(**kwargs)
