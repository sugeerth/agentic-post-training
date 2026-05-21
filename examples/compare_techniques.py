#!/usr/bin/env python3
"""Compare multiple post-training techniques side by side.

Usage:
    python3 examples/compare_techniques.py
    python3 examples/compare_techniques.py --techniques ppo dpo grpo spo
"""

import argparse
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.config import PipelineConfig
from pipeline.pipeline import AgenticPipeline


async def main():
    parser = argparse.ArgumentParser(description="Compare post-training techniques")
    parser.add_argument("--techniques", nargs="+", default=["ppo", "dpo", "grpo"],
                        help="Techniques to compare")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    config = PipelineConfig(model_name=args.model, epochs=args.epochs)
    pipeline = AgenticPipeline(config)
    results = await pipeline.compare_techniques(args.techniques)

    return results


if __name__ == "__main__":
    asyncio.run(main())
