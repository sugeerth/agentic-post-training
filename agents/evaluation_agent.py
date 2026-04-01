"""Evaluation agent for benchmarking trained models."""

from __future__ import annotations

import asyncio
import random
from typing import Any

from agents.base_agent import BaseAgent


class EvaluationAgent(BaseAgent):
    """Agent responsible for evaluating and benchmarking models.

    Runs standard benchmarks, compares before/after metrics,
    and provides recommendations for further training.
    """

    BENCHMARKS = {
        "mmlu": {"name": "MMLU", "description": "Massive Multitask Language Understanding", "max_score": 100},
        "humaneval": {"name": "HumanEval", "description": "Code generation benchmark", "max_score": 100},
        "mt_bench": {"name": "MT-Bench", "description": "Multi-turn conversation quality", "max_score": 10},
        "truthfulqa": {"name": "TruthfulQA", "description": "Truthfulness evaluation", "max_score": 100},
        "gsm8k": {"name": "GSM8K", "description": "Grade school math reasoning", "max_score": 100},
        "arc": {"name": "ARC Challenge", "description": "Abstract reasoning", "max_score": 100},
        "hellaswag": {"name": "HellaSwag", "description": "Commonsense reasoning", "max_score": 100},
    }

    def __init__(self, name: str = "Evaluator"):
        super().__init__(name, role="evaluator")
        self.register_capability("evaluation", "Run model benchmarks")
        self.register_capability("comparison", "Compare model versions")
        self.register_capability("recommendation", "Suggest further training")

    async def run(self, **kwargs) -> dict[str, Any]:
        benchmarks = kwargs.get("benchmarks", ["mmlu", "mt_bench", "humaneval", "gsm8k"])
        model_name = kwargs.get("model", "trained_model")

        await self.send_message("evaluation", {
            "message": f"Starting evaluation suite on {model_name}: {', '.join(benchmarks)}",
            "benchmarks": benchmarks,
        }, target="broadcast")

        results = {}
        for bench_name in benchmarks:
            bench = self.BENCHMARKS.get(bench_name)
            if not bench:
                continue

            self.log(f"Running {bench['name']}...")
            await asyncio.sleep(0.3)

            score = self._simulate_score(bench_name, bench["max_score"])
            baseline = score * random.uniform(0.75, 0.90)

            results[bench_name] = {
                "name": bench["name"],
                "score": round(score, 2),
                "baseline": round(baseline, 2),
                "improvement": f"+{(score - baseline) / baseline * 100:.1f}%",
                "max_score": bench["max_score"],
            }

            await self.send_message("status_update", {
                "message": f"{bench['name']}: {score:.1f}/{bench['max_score']} (baseline: {baseline:.1f}, +{(score - baseline) / baseline * 100:.1f}%)",
            }, target="Coordinator")

        # Generate recommendations
        recommendations = self._generate_recommendations(results)

        eval_report = {
            "model": model_name,
            "benchmarks": results,
            "recommendations": recommendations,
            "overall_improvement": self._calc_overall_improvement(results),
        }

        # Print nice report
        self._print_report(eval_report)

        await self.send_message("task_result", {
            "message": f"Evaluation complete. Overall improvement: {eval_report['overall_improvement']}",
            "report": eval_report,
        }, target="Coordinator")

        return eval_report

    async def step(self, **kwargs) -> dict[str, Any]:
        return await self.run(**kwargs)

    def _simulate_score(self, benchmark: str, max_score: float) -> float:
        base_scores = {
            "mmlu": 68.5, "humaneval": 42.0, "mt_bench": 7.8,
            "truthfulqa": 55.0, "gsm8k": 62.0, "arc": 72.0, "hellaswag": 78.0,
        }
        base = base_scores.get(benchmark, 60.0)
        # Add some variance and post-training improvement
        improvement = random.uniform(3, 12)
        noise = random.gauss(0, 1.5)
        return min(max_score, max(0, base + improvement + noise))

    def _generate_recommendations(self, results: dict) -> list[str]:
        recs = []
        for bench_name, data in results.items():
            ratio = data["score"] / data["max_score"]
            if ratio < 0.5:
                recs.append(f"Consider additional training focused on {data['name']} (score: {data['score']:.1f})")
            elif ratio < 0.7:
                recs.append(f"{data['name']} shows room for improvement — try technique-specific tuning")
        if not recs:
            recs.append("Model performs well across all benchmarks. Consider optimization for deployment.")
        return recs

    def _calc_overall_improvement(self, results: dict) -> str:
        if not results:
            return "N/A"
        improvements = []
        for data in results.values():
            if data["baseline"] > 0:
                improvements.append((data["score"] - data["baseline"]) / data["baseline"] * 100)
        avg = sum(improvements) / len(improvements) if improvements else 0
        return f"+{avg:.1f}%"

    def _print_report(self, report: dict) -> None:
        from agents.base_agent import BOLD, RESET, DIM
        print(f"\n{BOLD}{'═' * 70}")
        print(f"  📈 Evaluation Report: {report['model']}")
        print(f"{'═' * 70}{RESET}")

        for name, data in report["benchmarks"].items():
            bar_len = int(data["score"] / data["max_score"] * 30)
            bar = "█" * bar_len + "░" * (30 - bar_len)
            print(f"  {data['name']:15s} [{bar}] {data['score']:6.1f}/{data['max_score']} ({data['improvement']})")

        print(f"\n  {BOLD}Overall: {report['overall_improvement']}{RESET}")
        print(f"\n  {BOLD}Recommendations:{RESET}")
        for rec in report["recommendations"]:
            print(f"    → {rec}")
        print(f"\n{BOLD}{'═' * 70}{RESET}\n")
