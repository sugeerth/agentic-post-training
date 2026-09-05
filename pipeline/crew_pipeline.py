"""CrewAI-backed pipeline (optional layer).

CrewAI lives outside this repo's hard dependencies. When it's not
installed we fall back to the native AgenticPipeline so `--crew` is a
no-op rather than an error.

The shape of the CrewAI mapping:

    CoordinatorAgent          → Crew (the orchestrator)
    TrainingAgent             → Agent(role="trainer",   goal=..., tools=[python_exec])
    OptimizationAgent         → Agent(role="optimizer", goal=..., tools=[shell_run])
    EvaluationAgent           → Agent(role="evaluator", goal=..., tools=[file_read])

Each pipeline stage becomes a CrewAI Task. We keep the existing async
agents reachable for the actual heavy lifting — the CrewAI layer is the
orchestrator, not the worker.

If you want to wire a real LLM in CrewAI, set OPENAI_API_KEY or
ANTHROPIC_API_KEY and pass `llm=` when constructing the Agents. By
default we don't, because this framework's pipeline is mock-driven.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from agents.tools import ToolRegistry
from pipeline.config import PipelineConfig
from pipeline.pipeline import AgenticPipeline


def _has_llm_key() -> bool:
    return any(
        os.environ.get(k)
        for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GROQ_API_KEY")
    )

try:
    from crewai import Agent as CrewAgent
    from crewai import Crew, Task
    _CREWAI_AVAILABLE = True
except ImportError:
    _CREWAI_AVAILABLE = False


def crewai_available() -> bool:
    return _CREWAI_AVAILABLE


class CrewPipeline:
    """Crew-orchestrated pipeline. Falls back to AgenticPipeline if crewai missing."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        self.tools = ToolRegistry()
        self._native = AgenticPipeline(self.config)

    async def run(self) -> dict[str, Any]:
        if not _CREWAI_AVAILABLE:
            print("[CrewPipeline] crewai not installed — running native pipeline.")
            print("[CrewPipeline] Install with: pip install crewai")
            return await self._native.run()

        # CrewAI is sync; run the orchestration in a thread so we don't block
        # the event loop that the native agents use.
        return await asyncio.to_thread(self._run_crew_sync)

    def _build_agents(self) -> dict[str, CrewAgent]:
        trainer = CrewAgent(
            role="Post-training engineer",
            goal=f"Train the model with {self.config.technique.upper()} for {self.config.epochs} epochs",
            backstory="Specialist in RLHF, DPO, GRPO and related alignment methods.",
            allow_delegation=False,
            verbose=True,
        )
        optimizer = CrewAgent(
            role="Model optimizer",
            goal="Compress the trained model via quantization/pruning while preserving quality",
            backstory="Inference-cost specialist familiar with GPTQ, AWQ, GGUF.",
            allow_delegation=False,
            verbose=True,
        )
        evaluator = CrewAgent(
            role="Benchmark analyst",
            goal=f"Evaluate the model on {', '.join(self.config.benchmarks)} and recommend next steps",
            backstory="Maintains the eval harness across MMLU, MT-Bench, HumanEval.",
            allow_delegation=False,
            verbose=True,
        )
        return {"trainer": trainer, "optimizer": optimizer, "evaluator": evaluator}

    def _run_crew_sync(self) -> dict[str, Any]:
        agents = self._build_agents()
        tasks = [
            Task(
                description=(
                    f"Run {self.config.technique.upper()} training on {self.config.model_name} "
                    f"for {self.config.epochs} epochs."
                ),
                expected_output="A dict with final_loss, final_reward, epochs_completed.",
                agent=agents["trainer"],
            ),
            Task(
                description=(
                    f"Apply optimization ({self.config.quantization or 'skip'}) to the trained model."
                ),
                expected_output="A dict describing compression and quality retention.",
                agent=agents["optimizer"],
            ),
            Task(
                description=(
                    f"Evaluate on {', '.join(self.config.benchmarks)} and write a recommendation."
                ),
                expected_output="A report with per-benchmark scores and overall improvement.",
                agent=agents["evaluator"],
            ),
        ]

        crew = Crew(agents=list(agents.values()), tasks=tasks, verbose=True)

        crew_output: str
        if _has_llm_key():
            crew_output = str(crew.kickoff())
        else:
            print("\n[CrewPipeline] No LLM API key found (OPENAI_API_KEY / ANTHROPIC_API_KEY).")
            print("[CrewPipeline] Crew constructed with these roles + tasks:")
            for a in agents.values():
                print(f"  - role={a.role!r}  goal={a.goal!r}")
            for i, t in enumerate(tasks, 1):
                print(f"  task {i}: {t.description}")
            print("[CrewPipeline] Skipping crew.kickoff(); running native pipeline for metrics.\n")
            crew_output = "skipped (no LLM key)"

        # Run the native pipeline either way so we always get the canonical
        # metrics dict back. CrewAI orchestrates; native agents execute.
        native_results = asyncio.run(self._native.run())
        return {"crew_output": crew_output, "native_results": native_results}
