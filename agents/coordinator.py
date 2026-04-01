"""Coordinator agent that orchestrates the full post-training pipeline."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from agents.base_agent import BaseAgent, AgentStatus, BOLD, RESET, DIM
from agents.communication import MessageBus, Message, MessageType


@dataclass
class PipelineStage:
    name: str
    description: str
    agent_role: str
    config: dict[str, Any] = field(default_factory=dict)
    status: str = "pending"
    result: dict[str, Any] = field(default_factory=dict)
    duration: float = 0.0


class CoordinatorAgent(BaseAgent):
    """The master orchestrator that manages the full post-training pipeline.

    Responsibilities:
    - Register and manage worker agents
    - Create and execute pipeline plans
    - Assign tasks to appropriate agents
    - Monitor progress and handle failures
    - Provide beautiful status dashboards
    """

    DEFAULT_STAGES = [
        PipelineStage("data_prep", "Prepare and validate training data", "trainer"),
        PipelineStage("technique_selection", "Select optimal training technique", "coordinator"),
        PipelineStage("training", "Execute post-training with selected technique", "trainer"),
        PipelineStage("optimization", "Quantize, prune, or distill the model", "optimizer"),
        PipelineStage("evaluation", "Benchmark and evaluate the trained model", "evaluator"),
    ]

    def __init__(self, name: str = "Coordinator"):
        super().__init__(name, role="coordinator")
        self.workers: dict[str, BaseAgent] = {}
        self.stages: list[PipelineStage] = []
        self.bus: MessageBus | None = None
        self.pipeline_config: dict[str, Any] = {}
        self.register_capability("orchestration", "Coordinate multi-agent pipelines")
        self.register_capability("scheduling", "Schedule and assign tasks to agents")

    def setup_bus(self, bus: MessageBus) -> None:
        self.bus = bus
        bus.register_agent(self)

    def register_worker(self, agent: BaseAgent) -> None:
        self.workers[agent.name] = agent
        if self.bus:
            self.bus.register_agent(agent)
        self.log(f"Registered worker: {agent.name} (role: {agent.role})")

    def create_pipeline(self, config: dict[str, Any] | None = None) -> list[PipelineStage]:
        self.pipeline_config = config or {}
        self.stages = [
            PipelineStage(s.name, s.description, s.agent_role, dict(s.config))
            for s in self.DEFAULT_STAGES
        ]
        # Merge config into stages
        for stage in self.stages:
            stage.config.update(self.pipeline_config.get(stage.name, {}))
        return self.stages

    def get_worker_for_role(self, role: str) -> BaseAgent | None:
        for worker in self.workers.values():
            if worker.role == role and worker.status in (AgentStatus.IDLE, AgentStatus.COMPLETED):
                return worker
        return None

    def print_dashboard(self) -> None:
        print(f"\n{BOLD}{'═' * 70}")
        print(f"  📊 Pipeline Dashboard")
        print(f"{'═' * 70}{RESET}")

        # Agent status
        print(f"\n  {BOLD}Agents:{RESET}")
        all_agents = {"Coordinator": self, **self.workers}
        for name, agent in all_agents.items():
            status_icon = {
                AgentStatus.IDLE: "⚪",
                AgentStatus.RUNNING: "🔵",
                AgentStatus.WAITING: "🟡",
                AgentStatus.COMPLETED: "🟢",
                AgentStatus.FAILED: "🔴",
            }.get(agent.status, "⚪")
            caps = ", ".join(c.name for c in agent.capabilities[:3])
            print(f"    {status_icon} {agent._color}{name}{RESET} [{agent.role}] - {caps}")

        # Pipeline stages
        if self.stages:
            print(f"\n  {BOLD}Pipeline Stages:{RESET}")
            for i, stage in enumerate(self.stages):
                icon = {"pending": "⬜", "running": "🔄", "completed": "✅", "failed": "❌"}.get(stage.status, "⬜")
                dur = f" ({stage.duration:.1f}s)" if stage.duration > 0 else ""
                print(f"    {icon} {i+1}. {stage.name}: {stage.description}{dur}")

        print(f"\n{BOLD}{'═' * 70}{RESET}\n")

    async def run_stage(self, stage: PipelineStage) -> dict[str, Any]:
        stage.status = "running"
        start = time.time()

        await self.send_message(
            "coordination",
            {"message": f"Starting stage: {stage.name} - {stage.description}",
             "stage": stage.name},
            target="broadcast"
        )

        if stage.agent_role == "coordinator":
            result = await self._handle_coordinator_stage(stage)
        else:
            worker = self.get_worker_for_role(stage.agent_role)
            if not worker:
                stage.status = "failed"
                self.log(f"No available worker for role: {stage.agent_role}")
                return {"error": f"No worker for role {stage.agent_role}"}

            result = await worker.execute(**stage.config)

        stage.duration = time.time() - start
        stage.status = "completed"
        stage.result = result

        await self.send_message(
            "task_result",
            {"message": f"Stage {stage.name} completed in {stage.duration:.1f}s",
             "stage": stage.name, "result_keys": list(result.keys())},
            target="broadcast"
        )
        return result

    async def _handle_coordinator_stage(self, stage: PipelineStage) -> dict[str, Any]:
        if stage.name == "technique_selection":
            technique = self.pipeline_config.get("technique", "grpo")
            self.log(f"Selected technique: {technique}")
            # Store in shared memory
            self.memory.set("selected_technique", technique)
            for worker in self.workers.values():
                worker.memory.set("selected_technique", technique)
            return {"technique": technique}
        return {}

    async def run(self, **kwargs) -> dict[str, Any]:
        self.log(f"🚀 Initiating post-training pipeline")
        config = kwargs.get("config", {})
        self.create_pipeline(config)
        self.print_dashboard()

        results = {}
        for stage in self.stages:
            self.log(f"Executing stage: {stage.name}")
            try:
                result = await self.run_stage(stage)
                results[stage.name] = result
                self.memory.set(f"stage_{stage.name}", result)
            except Exception as e:
                self.log(f"Stage {stage.name} failed: {e}", level="error")
                stage.status = "failed"
                results[stage.name] = {"error": str(e)}
                break

        self.print_dashboard()

        await self.send_message(
            "coordination",
            {"message": "Pipeline complete! All stages finished.",
             "stages_completed": sum(1 for s in self.stages if s.status == "completed"),
             "total_stages": len(self.stages)},
            target="broadcast"
        )
        return results

    async def step(self, **kwargs) -> dict[str, Any]:
        if not self.stages:
            return {"status": "no_pipeline"}
        for stage in self.stages:
            if stage.status == "pending":
                return await self.run_stage(stage)
        return {"status": "all_stages_complete"}
