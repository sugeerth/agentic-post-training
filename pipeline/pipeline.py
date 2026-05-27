"""Main pipeline orchestrator that ties agents, techniques, and optimization together."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from agents.base_agent import BOLD, RESET
from agents.communication import MessageBus
from agents.coordinator import CoordinatorAgent
from agents.evaluation_agent import EvaluationAgent
from agents.optimization_agent import OptimizationAgent
from agents.training_agent import TrainingAgent
from pipeline.config import PipelineConfig


class AgenticPipeline:
    """The main entry point for running agentic post-training.

    Creates and coordinates all agents, executes the pipeline,
    and provides a beautiful terminal interface.

    Usage:
        config = PipelineConfig(technique="grpo", model_name="gpt2", epochs=3)
        pipeline = AgenticPipeline(config)
        results = await pipeline.run()
    """

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self.bus = MessageBus(verbose=True)

        # Create agents
        self.coordinator = CoordinatorAgent()
        self.trainer = TrainingAgent()
        self.optimizer = OptimizationAgent()
        self.evaluator = EvaluationAgent()

        # Wire up
        self.coordinator.setup_bus(self.bus)
        self.coordinator.register_worker(self.trainer)
        self.coordinator.register_worker(self.optimizer)
        self.coordinator.register_worker(self.evaluator)

    def _print_header(self) -> None:
        print(f"\n{BOLD}{'╔' + '═' * 68 + '╗'}")
        print(f"║{'AGENTIC POST-TRAINING FRAMEWORK':^68s}║")
        print(f"║{'Orchestrating LLM alignment through agent collaboration':^68s}║")
        print(f"{'╚' + '═' * 68 + '╝'}{RESET}\n")

        print(f"  Model:     {self.config.model_name}")
        print(f"  Technique: {self.config.technique.upper()}")
        print(f"  Epochs:    {self.config.epochs}")
        if self.config.quantization:
            print(f"  Quant:     {self.config.quantization.upper()}")
        print(f"  Benchmarks: {', '.join(self.config.benchmarks)}")
        print()

    async def run(self) -> dict[str, Any]:
        """Execute the full pipeline."""
        self._print_header()
        start_time = time.time()

        # Build coordinator config from pipeline config
        coordinator_config = {
            "technique": self.config.technique,
            "model": self.config.model_name,
            "epochs": self.config.epochs,
            "data_prep": {},
            "technique_selection": {"technique": self.config.technique},
            "training": {
                "technique": self.config.technique,
                "model": self.config.model_name,
                "epochs": self.config.epochs,
            },
            "optimization": {
                "method": "quantization" if self.config.quantization else "skip",
                "quant_type": self.config.quantization or "gptq",
            },
            "evaluation": {
                "benchmarks": self.config.benchmarks,
                "model": self.config.model_name,
            },
        }

        results = await self.coordinator.execute(config=coordinator_config)

        elapsed = time.time() - start_time

        # Print communication log
        self.bus.print_conversation()

        # Final summary
        print(f"\n{BOLD}{'═' * 70}")
        print(f"  🎉 Pipeline Complete! ({elapsed:.1f}s)")
        print(f"{'═' * 70}{RESET}")
        msg_summary = self.bus.summary()
        print(f"  Messages exchanged: {sum(msg_summary.values())}")
        for msg_type, count in msg_summary.items():
            print(f"    {msg_type}: {count}")
        print()

        return results

    async def compare_techniques(self, techniques: list[str]) -> dict[str, Any]:
        """Run multiple techniques and compare results (parallel)."""
        print(f"\n{BOLD}Comparing techniques (parallel): {', '.join(t.upper() for t in techniques)}{RESET}\n")

        trainers: dict[str, TrainingAgent] = {}
        for tech in techniques:
            t = TrainingAgent(name=f"Trainer-{tech.upper()}")
            self.coordinator.register_worker(t)
            trainers[tech] = t

        async def _run(tech: str) -> tuple[str, dict[str, Any]]:
            print(f"  Launching {tech.upper()}...")
            result = await trainers[tech].execute(
                technique=tech,
                model=self.config.model_name,
                epochs=self.config.epochs,
            )
            return tech, result

        gathered = await asyncio.gather(*(_run(t) for t in techniques))
        all_results = dict(gathered)

        # Comparison table
        print(f"\n{BOLD}{'═' * 70}")
        print("  📊 Technique Comparison")
        print(f"{'═' * 70}{RESET}")
        print(f"  {'Technique':<12s} {'Final Loss':>12s} {'Reward':>10s} {'Epochs':>8s}")
        print(f"  {'─' * 44}")

        for tech, result in all_results.items():
            loss = result.get("final_loss", 0)
            reward = result.get("final_reward", 0)
            epochs = result.get("epochs_completed", 0)
            print(f"  {tech.upper():<12s} {loss:>12.4f} {reward:>10.4f} {epochs:>8d}")

        print(f"\n{BOLD}{'═' * 70}{RESET}\n")
        return all_results


def run_pipeline(config: PipelineConfig | None = None) -> dict[str, Any]:
    """Convenience function to run the pipeline synchronously."""
    pipeline = AgenticPipeline(config)
    return asyncio.run(pipeline.run())
