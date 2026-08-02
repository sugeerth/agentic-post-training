#!/usr/bin/env python3
"""Run the agentic post-training pipeline.

    python3 examples/run_agentic_training.py --goal "agentic tool use"
    python3 examples/run_agentic_training.py --goal reasoning --iterations 5
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.agentic_pipeline import AgenticPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Agentic post-training pipeline")
    parser.add_argument("--goal", default="agentic tool use",
                        help="Goal string — supervisor picks a plan from this.")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--rollouts", type=int, default=64)
    parser.add_argument("--out-dir", default="./output")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-message bus chatter; only print the TL;DR.")
    return parser.parse_args()


async def main() -> int:
    args = parse_args()
    pipeline = AgenticPipeline(verbose=not args.quiet, out_dir=args.out_dir)
    config = {
        "iterations": args.iterations,
        "group_size": args.group_size,
        "num_rollouts": args.rollouts,
        "benchmarks": ["mmlu", "gsm8k", "humaneval"],
    }
    await pipeline.run(goal=args.goal, config=config)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
