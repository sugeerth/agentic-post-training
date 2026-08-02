"""Paper-scan pipeline — daily automation for tracking agent-PT research.

Chain:
  PaperScanner  → fresh abstracts (arXiv live, or canned)
  TechniqueScout → algorithm names + claims per paper
  MimicWriter    → one stub file per new technique
  Digest         → human-friendly Markdown for the workflow / a PR body

The pipeline is *idempotent*: running it twice on the same day writes
the same file names, and MimicWriter skips files that already exist
unless `overwrite=True`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from agents.paper_scanner import PaperScanner
from agents.technique_scout import TechniqueScout
from agents.mimic_writer import MimicWriter


class PaperScanPipeline:
    def __init__(
        self,
        out_dir: str = "./output/paper_scan",
        mimic_root: str = "techniques/mimics",
    ):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.mimic_root = mimic_root
        self.scanner = PaperScanner()
        self.scout = TechniqueScout()
        self.writer = MimicWriter(root=mimic_root)

    async def run(
        self,
        max_results: int = 12,
        allow_network: bool = True,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        t0 = time.time()

        scan = await self.scanner.run(max_results=max_results, allow_network=allow_network)
        scout = await self.scout.run(papers=scan["papers"])
        write = await self.writer.run(techniques=scout["techniques"], overwrite=overwrite)

        digest_md = _render_digest(scan, scout, write)
        digest_path = self.out_dir / "digest.md"
        digest_path.write_text(digest_md)
        (self.out_dir / "digest.json").write_text(json.dumps({
            "scan": scan, "scout": scout, "write": write,
        }, indent=2))

        summary = {
            "papers": scan["count"],
            "source": scan["source"],
            "techniques_found": scout["techniques_found"],
            "mimics_written": len(write["written"]),
            "mimics_skipped": len(write["skipped"]),
            "digest_path": str(digest_path),
            "elapsed_s": round(time.time() - t0, 2),
        }
        print(digest_md)
        return summary


def _render_digest(scan: dict, scout: dict, write: dict) -> str:
    """Attention-budgeted digest: 3 lines above the fold, per-paper detail below."""
    lines: list[str] = []
    lines.append(f"# 📚 Paper scan · {scan['count']} papers · "
                 f"{scout['techniques_found']} candidate techniques · "
                 f"{len(write['written'])} new mimics")
    lines.append("")
    lines.append(f"**Source:** {scan['source']} · "
                 f"**Written:** {', '.join(f'`{s}`' for s in write['written']) or '_none_'} · "
                 f"**Skipped (already present):** {len(write['skipped'])}")
    lines.append("")

    # New mimics — most-important first (papers whose scout produced a stub).
    written_slugs = set(write["written"])
    by_slug: dict[str, dict] = {t["slug"]: t for t in scout["techniques"]}

    if written_slugs:
        lines.append("## 🆕 New mimics generated")
        for slug in write["written"]:
            t = by_slug.get(slug)
            if not t:
                continue
            claims = _claims_summary(t)
            lines.append(f"- **`{slug}`** — {t['paper_title']}")
            lines.append(f"    - Paper: [{t['paper_id']}]({t['paper_url']})")
            lines.append(f"    - Novelty: {t['novelty']}")
            if claims:
                lines.append(f"    - Claims: {claims}")
        lines.append("")

    # Papers where we found a technique but the file already existed.
    if scout["techniques_found"] and not written_slugs:
        lines.append("_All extracted techniques already had mimic files — nothing new to write._")
        lines.append("")

    # Full paper list — collapsed.
    lines.append("<details><summary>All papers scanned</summary>")
    lines.append("")
    for p in scan["papers"]:
        authors = ", ".join(p["authors"][:3])
        more = f" (+{len(p['authors']) - 3} more)" if len(p["authors"]) > 3 else ""
        lines.append(f"- {p['published']} — [{p['title']}]({p['url']}) · "
                     f"{authors}{more} · `{', '.join(p['categories'][:2])}`")
    lines.append("")
    lines.append("</details>")
    return "\n".join(lines)


def _claims_summary(technique: dict) -> str:
    parts = []
    for target, pct in technique.get("outperforms", [])[:2]:
        parts.append(f"beats {target}" + (f" by {pct:g}%" if pct is not None else ""))
    for pct, task in technique.get("gains", [])[:2]:
        parts.append(f"+{pct:g}% on {task}")
    flags = []
    if technique.get("mentions_kl"):
        flags.append("KL")
    if technique.get("mentions_process_reward"):
        flags.append("PRM")
    if technique.get("mentions_reward_hacking"):
        flags.append("hack-aware")
    if flags:
        parts.append(f"[{', '.join(flags)}]")
    return " · ".join(parts)
