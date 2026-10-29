#!/usr/bin/env python3
"""Run the autonomous closed-loop trainer.

    python3 examples/run_autonomous.py --target-success 0.85 --budget 2000
    python3 examples/run_autonomous.py --target-success 0.95 --budget 800   # will budget-out
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.autonomous_loop import AutonomousLoop


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Goal-driven autonomous training loop")
    p.add_argument("--target-success", type=float, default=0.85)
    p.add_argument("--budget", type=int, default=2000, help="Total rollout budget")
    p.add_argument("--initial-skill", type=float, default=0.15)
    p.add_argument("--max-rounds", type=int, default=20)
    p.add_argument("--kl-coef", type=float, default=0.05)
    p.add_argument("--rollouts-per-round", type=int, default=64)
    p.add_argument("--out-dir", default="./output/autonomous")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


async def main() -> int:
    args = parse_args()
    loop = AutonomousLoop(verbose=args.verbose, out_dir=args.out_dir)
    summary = await loop.run(
        target_success=args.target_success,
        budget_rollouts=args.budget,
        initial_skill=args.initial_skill,
        max_rounds=args.max_rounds,
        config={"kl_coef": args.kl_coef, "num_rollouts": args.rollouts_per_round},
    )
    result = summary["results"]["autonomous_loop"]
    return 0 if result["target_achieved"] else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
