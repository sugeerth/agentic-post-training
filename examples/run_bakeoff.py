#!/usr/bin/env python3
"""Run several recipes in parallel and pick a winner.

    python3 examples/run_bakeoff.py
    python3 examples/run_bakeoff.py --plans agentic-tool-use reasoning-r1
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.bakeoff_pipeline import BakeoffPipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a recipe bake-off")
    p.add_argument("--plans", nargs="+",
                   default=["agentic-tool-use", "reasoning-r1", "rejection-sampling-loop"])
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("--rollouts", type=int, default=32)
    p.add_argument("--kl-ceiling", type=float, default=0.2)
    p.add_argument("--out-dir", default="./output/bakeoff")
    p.add_argument("--history-dir", default=".runs")
    return p.parse_args()


async def main() -> int:
    args = parse_args()
    pipe = BakeoffPipeline(verbose=False, out_dir=args.out_dir, history_dir=args.history_dir)
    summary = await pipe.run(
        plans=args.plans,
        config={"iterations": args.iterations, "num_rollouts": args.rollouts,
                "benchmarks": ["mmlu", "gsm8k"]},
        kl_ceiling=args.kl_ceiling,
    )
    print(f"\nWinner: {summary['winner']}")
    print(f"Pareto: {' · '.join(summary['pareto_frontier'])}")
    print(f"Elapsed: {summary['elapsed_s']}s\n")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
