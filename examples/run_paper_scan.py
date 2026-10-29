#!/usr/bin/env python3
"""Run the daily paper-scan pipeline manually.

    python3 examples/run_paper_scan.py                    # tries arXiv, falls back to canned
    python3 examples/run_paper_scan.py --no-network       # canned only (deterministic)
    python3 examples/run_paper_scan.py --overwrite        # regenerate existing mimics
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.paper_scan_pipeline import PaperScanPipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scan arXiv for agent-PT papers and write mimic stubs")
    p.add_argument("--max-results", type=int, default=12)
    p.add_argument("--no-network", action="store_true",
                   help="Skip the arXiv fetch — use canned abstract list")
    p.add_argument("--overwrite", action="store_true",
                   help="Regenerate mimic files even when they already exist")
    p.add_argument("--out-dir", default="./output/paper_scan")
    p.add_argument("--mimic-root", default="techniques/mimics")
    return p.parse_args()


async def main() -> int:
    args = parse_args()
    pipeline = PaperScanPipeline(out_dir=args.out_dir, mimic_root=args.mimic_root)
    summary = await pipeline.run(
        max_results=args.max_results,
        allow_network=not args.no_network,
        overwrite=args.overwrite,
    )
    print(f"\n📊 {summary['papers']} papers · "
          f"{summary['mimics_written']} new mimics · "
          f"{summary['elapsed_s']}s · digest → {summary['digest_path']}\n")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
