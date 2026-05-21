"""Coordinator agent — composes Planner + Workers + Dashboard.

Phase 2 refactor: the original `CoordinatorAgent` was 190 LOC mixing
planning, execution, and dashboard printing. Each concern now lives in its
own module:

  - `agents.planner`     — pure stage construction from config
  - `agents.dashboard`   — pure dashboard rendering
  - `agents.coordinator` — asyncio dispatch + result aggregation (this file)

Public surface preserved: `CoordinatorAgent`, `PipelineStage`,
`create_pipeline`, `register_worker`, `setup_bus`, `print_dashboard`,
`run_stage`, `run`, `step` all keep their signatures so the existing
`tests/test_agents.py` and notebooks don't break.
"""

from __future__ import annotations

import time
from typing import Any

from agents.base_agent import AgentStatus, BaseAgent
from agents.communication import MessageBus
from agents.dashboard import print_dashboard as _print_dashboard
from agents.planner import PipelineStage, plan

# Re-export so `from agents.coordinator import PipelineStage` (legacy import) still works.
__all__ = ["CoordinatorAgent", "PipelineStage"]


class CoordinatorAgent(BaseAgent):
    """The master orchestrator. Dispatch only — planning + rendering are split out."""

    def __init__(self, name: str = "Coordinator") -> None:
        super().__init__(name, role="coordinator")
        self.workers: dict[str, BaseAgent] = {}
        self.stages: list[PipelineStage] = []
        self.bus: MessageBus | None = None
        self.pipeline_config: dict[str, Any] = {}
        self.register_capability("orchestration", "Coordinate multi-agent pipelines")
        self.register_capability("scheduling", "Schedule and assign tasks to agents")

    # ---- bus / worker registration --------------------------------------- #

    def setup_bus(self, bus: MessageBus) -> None:
        self.bus = bus
        bus.register_agent(self)

    def register_worker(self, agent: BaseAgent) -> None:
        self.workers[agent.name] = agent
        if self.bus is not None:
            self.bus.register_agent(agent)
        self.log(f"Registered worker: {agent.name} (role: {agent.role})")

    def get_worker_for_role(self, role: str) -> BaseAgent | None:
        for worker in self.workers.values():
            if worker.role == role and worker.status in (AgentStatus.IDLE, AgentStatus.COMPLETED):
                return worker
        return None

    # ---- planning (delegates to agents.planner) -------------------------- #

    def create_pipeline(self, config: dict[str, Any] | None = None) -> list[PipelineStage]:
        self.pipeline_config = config or {}
        self.stages = plan(self.pipeline_config)
        return self.stages

    # ---- presentation (delegates to agents.dashboard) -------------------- #

    def print_dashboard(self) -> None:
        all_agents: dict[str, BaseAgent] = {"Coordinator": self, **self.workers}
        _print_dashboard(all_agents, self.stages)

    # ---- execution ------------------------------------------------------- #

    async def run_stage(self, stage: PipelineStage) -> dict[str, Any]:
        stage.status = "running"
        start = time.time()

        await self.send_message(
            "coordination",
            {"message": f"Starting stage: {stage.name} - {stage.description}",
             "stage": stage.name},
            target="broadcast",
        )

        if stage.agent_role == "coordinator":
            result = await self._handle_coordinator_stage(stage)
        else:
            worker = self.get_worker_for_role(stage.agent_role)
            if worker is None:
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
            target="broadcast",
        )
        return result

    async def _handle_coordinator_stage(self, stage: PipelineStage) -> dict[str, Any]:
        if stage.name == "technique_selection":
            technique = self.pipeline_config.get("technique", "grpo")
            self.log(f"Selected technique: {technique}")
            self.memory.set("selected_technique", technique)
            for worker in self.workers.values():
                worker.memory.set("selected_technique", technique)
            return {"technique": technique}
        return {}

    async def run(self, **kwargs: Any) -> dict[str, Any]:
        self.log("🚀 Initiating post-training pipeline")
        self.create_pipeline(kwargs.get("config", {}))
        self.print_dashboard()

        results: dict[str, Any] = {}
        for stage in self.stages:
            self.log(f"Executing stage: {stage.name}")
            try:
                result = await self.run_stage(stage)
                results[stage.name] = result
                self.memory.set(f"stage_{stage.name}", result)
            except Exception as e:  # broad: surface any worker failure
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
            target="broadcast",
        )
        return results

    async def step(self, **kwargs: Any) -> dict[str, Any]:
        if not self.stages:
            return {"status": "no_pipeline"}
        for stage in self.stages:
            if stage.status == "pending":
                return await self.run_stage(stage)
        return {"status": "all_stages_complete"}
