#!/usr/bin/env python3
"""Run the full agentic post-training pipeline.

Usage:
    python3 examples/run_pipeline.py --technique grpo --model gpt2 --epochs 3
    python3 examples/run_pipeline.py --preset production
    python3 examples/run_pipeline.py --technique dpo --quantize gptq
"""

import argparse
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.config import PipelineConfig
from pipeline.pipeline import AgenticPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Agentic Post-Training Pipeline")
    parser.add_argument("--technique", type=str, default="grpo",
                        choices=["ppo", "grpo", "dpo", "spo", "rlhf", "rlaif", "kto", "orpo", "spin", "simpo", "ipo"],
                        help="Post-training technique")
    parser.add_argument("--model", type=str, default="gpt2", help="Model name or path")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--quantize", type=str, default=None,
                        choices=["gptq", "awq", "gguf", "nf4", "int8"],
                        help="Quantization method")
    parser.add_argument("--preset", type=str, default=None,
                        choices=["quick_dpo", "full_rlhf", "efficient_grpo", "research_spo", "production"],
                        help="Use a preset configuration")
    parser.add_argument("--benchmarks", type=str, nargs="+",
                        default=["mmlu", "mt_bench", "humaneval"],
                        help="Evaluation benchmarks")
    parser.add_argument("--output-dir", type=str, default="./output")
    return parser.parse_args()


async def main():
    args = parse_args()

    if args.preset:
        config = PipelineConfig.preset(args.preset)
    else:
        config = PipelineConfig(
            technique=args.technique,
            model_name=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            quantization=args.quantize,
            benchmarks=args.benchmarks,
            output_dir=args.output_dir,
        )

    errors = config.validate()
    if errors:
        print(f"Configuration errors: {errors}")
        sys.exit(1)

    pipeline = AgenticPipeline(config)
    results = await pipeline.run()

    print("Pipeline finished successfully!")
    return results


if __name__ == "__main__":
    asyncio.run(main())
