"""Supervisor agent — hierarchical layer above the Coordinator.

The Coordinator schedules stages. The Supervisor decides *what pipeline to
run* in the first place, watches the run, and intervenes when metrics go
sideways (reward collapse, KL blow-up, benchmark regression).

Modeled on the "manager–worker" pattern used in agentic post-training stacks
(Kimi K2's supervisor loop, DeepSeek-R1's iterative RL + rejection-sampling
outer loop): a thin, always-on layer that owns strategy while the coordinator
owns execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agents.base_agent import BaseAgent, BOLD, RESET
from agents.coordinator import CoordinatorAgent


@dataclass
class Intervention:
    """A supervisor decision recorded on the run."""
    stage: str
    reason: str
    action: str  # one of: continue, adjust, rollback, abort


@dataclass
class RunPlan:
    """A named recipe the supervisor can pick from."""
    name: str
    stages: list[str]
    rationale: str
    config_overrides: dict[str, Any] = field(default_factory=dict)


class SupervisorAgent(BaseAgent):
    """Hierarchical supervisor above the CoordinatorAgent.

    Responsibilities:
      • plan     — pick a RunPlan based on config + goal
      • delegate — hand execution to the coordinator
      • monitor  — inspect stage results as they land
      • intervene — abort / adjust / rollback on anomalies
    """

    PLANS: list[RunPlan] = [
        RunPlan(
            name="reasoning-r1",
            stages=["curate", "sft", "rm", "grpo", "distill", "eval"],
            rationale="DeepSeek-R1 recipe: SFT cold-start → outcome-reward GRPO → distill.",
            config_overrides={"technique": "grpo", "reward_kind": "outcome"},
        ),
        RunPlan(
            name="agentic-tool-use",
            stages=["curate", "trajectory", "rm", "multi_turn_grpo", "eval"],
            rationale="Kimi K2 / tool-agent recipe: sample multi-turn tool-use trajectories, "
                      "reward on task success, PPO-style clipped update.",
            config_overrides={"technique": "multi_turn_grpo", "reward_kind": "process"},
        ),
        RunPlan(
            name="preference-alignment",
            stages=["curate", "sft", "dpo", "eval"],
            rationale="Cheap DPO alignment when preference pairs already exist.",
            config_overrides={"technique": "trajectory_dpo"},
        ),
        RunPlan(
            name="rejection-sampling-loop",
            stages=["trajectory", "rft", "eval"],
            rationale="STaR / RFT: sample, keep the correct rollouts, SFT on them, repeat.",
            config_overrides={"technique": "rejection_sampling_ft"},
        ),
    ]

    def __init__(self, name: str = "Supervisor"):
        super().__init__(name, role="supervisor")
        self.coordinator: CoordinatorAgent | None = None
        self.selected_plan: RunPlan | None = None
        self.interventions: list[Intervention] = []
        self.register_capability("planning", "Choose a training recipe from the goal")
        self.register_capability("monitoring", "Detect anomalies during a run")
        self.register_capability("intervention", "Abort / adjust / rollback the pipeline")

    def attach_coordinator(self, coordinator: CoordinatorAgent) -> None:
        self.coordinator = coordinator

    def pick_plan(self, goal: str, config: dict[str, Any]) -> RunPlan:
        goal = (goal or "").lower()
        if "tool" in goal or "agent" in goal:
            plan = self._by_name("agentic-tool-use")
        elif "reason" in goal or "math" in goal or "code" in goal:
            plan = self._by_name("reasoning-r1")
        elif "prefer" in goal or "align" in goal:
            plan = self._by_name("preference-alignment")
        elif "reject" in goal or "star" in goal or "rft" in goal:
            plan = self._by_name("rejection-sampling-loop")
        else:
            plan = self._by_name("agentic-tool-use")
        plan.config_overrides = {**plan.config_overrides, **config}
        self.selected_plan = plan
        return plan

    def _by_name(self, name: str) -> RunPlan:
        for p in self.PLANS:
            if p.name == name:
                return RunPlan(p.name, list(p.stages), p.rationale, dict(p.config_overrides))
        raise KeyError(name)

    def review_stage(self, stage: str, result: dict[str, Any]) -> Intervention:
        """Inspect a stage result and decide what to do next.

        Anomaly rules are intentionally simple — they are guardrails, not
        oracles. Real production checks live in the eval agent.
        """
        kl = float(result.get("kl_divergence", 0.0) or 0.0)
        reward = float(result.get("final_reward", result.get("reward", 0.0)) or 0.0)
        loss = float(result.get("final_loss", result.get("loss", 0.0)) or 0.0)

        if kl > 0.5:
            action, reason = "adjust", f"KL={kl:.2f} exceeds 0.5 — tighten kl_coef"
        elif reward < 0.1 and stage in {"grpo", "multi_turn_grpo", "training"}:
            action, reason = "rollback", f"reward collapsed to {reward:.2f}"
        elif loss > 5.0:
            action, reason = "abort", f"loss={loss:.2f} diverged"
        else:
            action, reason = "continue", "metrics within expected envelope"

        decision = Intervention(stage=stage, reason=reason, action=action)
        self.interventions.append(decision)
        return decision

    def print_plan(self) -> None:
        p = self.selected_plan
        if not p:
            return
        print(f"\n{BOLD}Supervisor plan: {p.name}{RESET}")
        print(f"  Why:    {p.rationale}")
        print(f"  Stages: {' → '.join(p.stages)}")
        if p.config_overrides:
            keys = ", ".join(f"{k}={v}" for k, v in p.config_overrides.items())
            print(f"  Config: {keys}")
        print()

    async def run(self, **kwargs) -> dict[str, Any]:
        goal = kwargs.get("goal", "agentic tool use")
        config = kwargs.get("config", {})

        plan = self.pick_plan(goal, config)
        await self.send_message(
            "coordination",
            {"message": f"Plan selected: {plan.name} ({plan.rationale})", "plan": plan.name},
            target="broadcast",
        )
        self.print_plan()

        if not self.coordinator:
            return {"error": "no coordinator attached", "plan": plan.name}

        results = await self.coordinator.execute(config={**config, **plan.config_overrides})

        # Post-run review — one intervention per stage that has a result dict.
        for stage_name, stage_result in results.items():
            if isinstance(stage_result, dict):
                self.review_stage(stage_name, stage_result)

        return {
            "plan": plan.name,
            "stages": plan.stages,
            "results": results,
            "interventions": [i.__dict__ for i in self.interventions],
        }

    async def step(self, **kwargs) -> dict[str, Any]:
        return await self.run(**kwargs)
