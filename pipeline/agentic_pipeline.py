"""Agentic post-training pipeline — supervisor + coordinator + specialists.

Two-layer orchestration:

  Supervisor         picks the recipe, reviews outcomes, decides to
                    continue / adjust / rollback / abort
  Coordinator       owns per-stage plumbing (dispatch, timing, dashboard)
  Specialists       Trajectory, RewardModel, AgenticTrainer, Evaluator, Reporter

The pipeline exists so that a user runs one command
(`python3 examples/run_agentic_training.py --goal "tool use"`) and gets a
scannable TL;DR, not a 500-line log.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from agents.base_agent import BOLD, RESET
from agents.communication import MessageBus
from agents.coordinator import CoordinatorAgent, PipelineStage
from agents.supervisor_agent import SupervisorAgent
from agents.trajectory_agent import TrajectoryAgent
from agents.reward_model_agent import RewardModelAgent
from agents.agentic_training_agent import AgenticTrainingAgent
from agents.evaluation_agent import EvaluationAgent
from agents.reporter_agent import ReporterAgent


# Mapping from a plan's stage name → (agent role, human description, extra config keys)
STAGE_SPEC: dict[str, tuple[str, str]] = {
    "curate": ("trajectory", "Sample & curate agent trajectories"),
    "trajectory": ("trajectory", "Sample & curate agent trajectories"),
    "sft": ("agentic_trainer", "Supervised fine-tuning cold-start"),
    "rm": ("reward_model", "Fit outcome / process reward model"),
    "grpo": ("agentic_trainer", "GRPO update on curated rollouts"),
    "multi_turn_grpo": ("agentic_trainer", "Multi-turn GRPO on tool-use trajectories"),
    "dpo": ("agentic_trainer", "Trajectory DPO on preference pairs"),
    "rft": ("agentic_trainer", "Rejection-sampling fine-tuning outer loop"),
    "distill": ("agentic_trainer", "Distill reasoning traces into a smaller policy"),
    "eval": ("evaluator", "Benchmark suite"),
}


class AgenticPipeline:
    """End-to-end agentic post-training run.

        pipeline = AgenticPipeline()
        results = await pipeline.run(goal="agentic tool use", config={...})
    """

    def __init__(self, verbose: bool = True, out_dir: str = "./output"):
        self.bus = MessageBus(verbose=verbose)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.supervisor = SupervisorAgent()
        self.coordinator = CoordinatorAgent()
        self.trajectory = TrajectoryAgent()
        self.rm = RewardModelAgent()
        self.trainer = AgenticTrainingAgent()
        self.evaluator = EvaluationAgent()
        self.reporter = ReporterAgent()

        self.bus.register_agent(self.supervisor)
        self.coordinator.setup_bus(self.bus)
        for w in (self.trajectory, self.rm, self.trainer, self.evaluator, self.reporter):
            self.coordinator.register_worker(w)
        self.supervisor.attach_coordinator(self.coordinator)

    def _header(self, goal: str, plan_name: str) -> None:
        print(f"\n{BOLD}{'╔' + '═' * 68 + '╗'}")
        print(f"║{'AGENTIC POST-TRAINING · Supervisor + Specialists':^68s}║")
        print(f"{'╚' + '═' * 68 + '╝'}{RESET}")
        print(f"  Goal:  {goal}")
        print(f"  Plan:  {plan_name}\n")

    async def run(self, goal: str = "agentic tool use", config: dict[str, Any] | None = None) -> dict[str, Any]:
        config = config or {}
        start = time.time()

        plan = self.supervisor.pick_plan(goal, config)
        self._header(goal, plan.name)
        self.supervisor.print_plan()

        # Build a stage list from the plan, using the coordinator's dispatch.
        merged_cfg = {**config, **plan.config_overrides}
        stages: list[PipelineStage] = []
        for name in plan.stages:
            role, desc = STAGE_SPEC.get(name, ("agentic_trainer", name))
            stage_cfg = {**merged_cfg, "technique": merged_cfg.get("technique", name)}
            stages.append(PipelineStage(name=name, description=desc, agent_role=role, config=stage_cfg))
        self.coordinator.stages = stages
        self.coordinator.pipeline_config = merged_cfg
        self.coordinator.print_dashboard()

        results: dict[str, Any] = {}
        for stage in stages:
            self.coordinator.log(f"Executing stage: {stage.name}")
            result = await self.coordinator.run_stage(stage)
            results[stage.name] = result
            # Supervisor peeks at each stage as it completes.
            decision = self.supervisor.review_stage(stage.name, result if isinstance(result, dict) else {})
            if decision.action == "abort":
                self.coordinator.log(f"Supervisor aborted after {stage.name}: {decision.reason}", level="error")
                break

        self.coordinator.print_dashboard()

        run_summary = {
            "goal": goal,
            "plan": plan.name,
            "stages": plan.stages,
            "results": results,
            "interventions": [i.__dict__ for i in self.supervisor.interventions],
            "elapsed_s": round(time.time() - start, 2),
        }

        # Reporter runs last — turns the summary into a human-friendly TL;DR.
        report = await self.reporter.execute(run=run_summary, out_dir=str(self.out_dir))
        run_summary["report"] = report

        print(f"\n{BOLD}{'═' * 70}")
        print(f"  🎉 Agentic pipeline complete ({run_summary['elapsed_s']}s)")
        print(f"     Report: {report['report_md']}")
        print(f"{'═' * 70}{RESET}\n")

        return run_summary
