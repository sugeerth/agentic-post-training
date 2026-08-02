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
    """A supervisor decision recorded on the run.

    `config_patch` is what makes the decision *real*: the loop merges it
    into the next round's config. An `adjust` with an empty patch is
    just commentary — the supervisor should never emit one.
    """
    stage: str
    reason: str
    action: str  # one of: continue, adjust, rollback, abort, stop
    config_patch: dict[str, Any] = field(default_factory=dict)


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

    def review_stage(self, stage: str, result: dict[str, Any],
                     config: dict[str, Any] | None = None) -> Intervention:
        """Inspect a stage result and decide what to do next.

        Anomaly rules are intentionally simple — they are guardrails, not
        oracles. Every non-continue decision ships a config_patch so the
        caller can actually act on it.
        """
        config = config or {}
        kl = float(result.get("kl_divergence", 0.0) or 0.0)
        reward = float(result.get("final_reward", result.get("reward", 0.0)) or 0.0)
        loss = float(result.get("final_loss", result.get("loss", 0.0)) or 0.0)
        patch: dict[str, Any] = {}

        if kl > 0.5:
            action, reason = "adjust", f"KL={kl:.2f} exceeds 0.5 — tighten kl_coef"
            patch = {"kl_coef": round(float(config.get("kl_coef", 0.05)) * 2, 4)}
        elif reward < 0.1 and stage in {"grpo", "multi_turn_grpo", "training"}:
            action, reason = "rollback", f"reward collapsed to {reward:.2f}"
            patch = {"restore_best": True}
        elif loss > 5.0:
            action, reason = "abort", f"loss={loss:.2f} diverged"
        else:
            action, reason = "continue", "metrics within expected envelope"

        decision = Intervention(stage=stage, reason=reason, action=action, config_patch=patch)
        self.interventions.append(decision)
        return decision

    KL_COEF_CAP = 0.4          # beyond this the update is so conservative it can't learn
    ESCALATION_BUDGET = 2      # distinct interventions to try per no-gain streak before stopping

    def decide_round(
        self,
        round_metrics: dict[str, Any],
        best_success: float,
        target_success: float,
        rounds_without_gain: int,
        budget_remaining: int,
        config: dict[str, Any],
        patience: int = 3,
    ) -> Intervention:
        """Loop-mode decision: what should the NEXT round do?

        Priority order (first match wins):
          1. target met       → stop
          2. budget exhausted → stop
          3. no gain for ≥ patience rounds → escalate through interventions
             (rollback-to-best on hacking evidence, then exploration boost),
             at most ESCALATION_BUDGET per streak — then stop. A supervisor
             that adjusts forever isn't supervising.
          4. hacking (high) with gains still coming → rollback + tighten KL
          5. drift (medium)  → tighten kl_coef, hard-capped at KL_COEF_CAP;
             at the cap it stops adjusting (spamming the same knob is noise)
          6. otherwise → continue

        Escalation state lives in config under underscore keys; the loop
        clears them when a new best is found. Every non-continue decision
        carries a config_patch — decisions are never advisory.
        """
        success = float(round_metrics.get("task_success_rate", 0.0))
        audit = round_metrics.get("audit") or {}
        severity = audit.get("severity", "low")
        kl_coef = float(config.get("kl_coef", 0.05))
        escalations = int(config.get("_escalations", 0))

        if success >= target_success:
            d = Intervention("loop", f"target met: {success:.0%} ≥ {target_success:.0%}", "stop")
        elif budget_remaining <= 0:
            d = Intervention("loop", "rollout budget exhausted", "stop")
        elif rounds_without_gain >= patience:
            if escalations >= self.ESCALATION_BUDGET:
                d = Intervention(
                    "loop",
                    f"no improvement in {rounds_without_gain} rounds despite "
                    f"{escalations} interventions — converged at {best_success:.0%}",
                    "stop",
                )
            elif severity in ("medium", "high") and not config.get("_rolled_back"):
                d = Intervention(
                    "loop",
                    f"plateaued with reward/eval misalignment (ρ={audit.get('spearman_rho')}) "
                    f"— rollback to best ({best_success:.0%}) and tighten KL",
                    "rollback",
                    {"restore_best": True,
                     "kl_coef": min(self.KL_COEF_CAP, round(kl_coef * 2, 4)),
                     "_rolled_back": True, "_escalations": escalations + 1},
                )
            else:
                d = Intervention(
                    "loop", f"plateaued {rounds_without_gain} rounds — boost exploration",
                    "adjust",
                    {"num_rollouts": int(config.get("num_rollouts", 64) * 2),
                     "keep_top_frac": max(0.25, float(config.get("keep_top_frac", 0.5)) - 0.15),
                     "_escalations": escalations + 1},
                )
        elif severity == "high":
            d = Intervention(
                "loop", f"reward hacking suspected (ρ={audit.get('spearman_rho')}) — "
                        f"rollback to best ({best_success:.0%}) and tighten KL",
                "rollback",
                {"restore_best": True,
                 "kl_coef": min(self.KL_COEF_CAP, round(kl_coef * 2, 4))},
            )
        elif severity == "medium" and kl_coef < self.KL_COEF_CAP:
            d = Intervention(
                "loop", "reward/eval drift — tighten kl_coef",
                "adjust", {"kl_coef": min(self.KL_COEF_CAP, round(kl_coef * 2, 4))},
            )
        else:
            d = Intervention("loop", "improving — continue", "continue")

        self.interventions.append(d)
        return d

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
