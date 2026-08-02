"""Autonomous closed-loop trainer — the actual core of agent post-training.

Everything else in this repo is staging for this loop:

    ┌────────────────────────────────────────────────────────┐
    │                                                        │
    │   collect(policy skill s) ──▶ curate ──▶ train ──▶ s'  │
    │        ▲                                          │    │
    │        └───────────── next round uses s' ─────────┘    │
    │                                                        │
    │   measure s' on fresh env rollouts (probe)             │
    │   audit reward vs measured success (hacking check)     │
    │   supervisor decides: continue / adjust / rollback /   │
    │     stop — and its config_patch MUTATES the next round │
    └────────────────────────────────────────────────────────┘

This is expert iteration / iterated RFT (STaR, DeepSeek-R1's multi-round
recipe) with a supervisor in the loop. The three properties that make it
"agentic" rather than a script:

  1. Closed feedback — the policy that collects round r+1's data is the
     policy improved in round r. Skill is state, not a schedule.
  2. Goal-driven termination — the loop runs until the target success
     rate is met, the rollout budget is spent, or improvement genuinely
     plateaus. Not a fixed stage list.
  3. Real interventions — supervisor decisions carry config patches the
     loop applies: KL tightening changes the next update's conservatism,
     rollback restores the best-known policy, exploration boosts change
     collection itself.

Policy skill is modeled as the TrajectoryAgent's `p_expert` mixing
probability — measured success comes from *fresh environment rollouts*,
never from the training simulation, so the loop's decisions are grounded
in the same env the policy acts in.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

from agents.base_agent import BOLD, RESET
from agents.registry import build
from agents.communication import MessageBus


@dataclass
class RoundRecord:
    iteration: int
    skill_before: float
    skill_after: float
    reward: float                 # what the trainer chased (curated quality)
    success_rate: float           # measured on fresh env rollouts (probe)
    kl_divergence: float
    loss: float
    rollouts_used: int
    decision: str
    decision_reason: str

    def to_iteration_metrics(self) -> dict[str, float]:
        """Shape one round as an iteration_metrics entry so the reporter's
        sparklines and the hacking audit consume the loop unchanged."""
        return {
            "iteration": self.iteration,
            "reward": self.reward,
            "loss": self.loss,
            "success_rate": self.success_rate,
            "kl_divergence": self.kl_divergence,
        }


@dataclass
class LoopState:
    skill: float
    best_skill: float
    best_success: float = 0.0
    rounds_without_gain: int = 0
    budget_used: int = 0
    rounds: list[RoundRecord] = field(default_factory=list)


class AutonomousLoop:
    """Goal-driven training loop. Constructed from the agent registry —
    swap any specialist by registering a replacement under its role."""

    def __init__(self, verbose: bool = False, out_dir: str = "./output/autonomous",
                 history_dir: str = ".runs"):
        self.bus = MessageBus(verbose=verbose)
        self.supervisor = build("supervisor")
        self.trajectory = build("trajectory")
        self.detector = build("reward_hacking_detector")
        from pipeline.run_history import RunHistory
        self.reporter = build("reporter", history=RunHistory(history_dir))
        for a in (self.supervisor, self.trajectory, self.detector, self.reporter):
            self.bus.register_agent(a)
        self.out_dir = out_dir

    # ---- round primitives -------------------------------------------------- #

    async def _collect(self, state: LoopState, config: dict[str, Any]) -> dict[str, Any]:
        n = int(config.get("num_rollouts", 64))
        result = await self.trajectory.run(
            num_rollouts=n,
            p_expert=state.skill,
            keep_top_frac=float(config.get("keep_top_frac", 0.5)),
            seed=int(config.get("seed", 0)) + len(state.rounds),
        )
        state.budget_used += n
        return result

    def _train(self, state: LoopState, curated_quality: float, config: dict[str, Any]) -> float:
        """One policy update. Returns the new skill.

        Improvement is logistic: gain ∝ curated quality × headroom (1 - s).
        A higher kl_coef makes the update more conservative — smaller gain,
        smaller KL. This is the actual trade RL practitioners tune.
        """
        gain = float(config.get("learning_gain", 0.35))
        kl_coef = float(config.get("kl_coef", 0.05))
        conservatism = 1.0 / (1.0 + 8.0 * kl_coef)      # kl_coef 0.05 → ~0.71
        delta = gain * conservatism * curated_quality * (1.0 - state.skill)
        return min(0.99, state.skill + delta)

    async def _probe(self, state: LoopState, config: dict[str, Any]) -> float:
        """Measure the updated policy on fresh env rollouts. This is the
        loop's ground truth — decisions never trust the training curve."""
        n = int(config.get("probe_rollouts", 24))
        result = await self.trajectory.run(
            num_rollouts=n,
            p_expert=state.skill,
            keep_top_frac=1.0,                          # probe keeps everything
            seed=int(config.get("seed", 0)) + 10_000 + len(state.rounds),
        )
        state.budget_used += n
        return float(result["success_rate"])

    # ---- the loop ---------------------------------------------------------- #

    async def run(
        self,
        target_success: float = 0.85,
        budget_rollouts: int = 2000,
        initial_skill: float = 0.15,
        max_rounds: int = 20,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        config = {"num_rollouts": 64, "probe_rollouts": 24, "keep_top_frac": 0.5,
                  "kl_coef": 0.05, "learning_gain": 0.35, "seed": 0,
                  **(config or {})}
        state = LoopState(skill=initial_skill, best_skill=initial_skill)
        t0 = time.time()
        stop_reason = f"max_rounds ({max_rounds}) reached"

        print(f"\n{BOLD}Autonomous loop · target {target_success:.0%} · "
              f"budget {budget_rollouts} rollouts{RESET}\n")

        for it in range(1, max_rounds + 1):
            # Pre-round budget guard: never start a round we can't afford.
            # (Post-round checking overspends by one round — caught in testing
            # when a 500-rollout budget burned 528.)
            round_cost = int(config["num_rollouts"]) + int(config["probe_rollouts"])
            if budget_rollouts - state.budget_used < round_cost:
                stop_reason = (f"rollout budget exhausted "
                               f"({state.budget_used}/{budget_rollouts} used, "
                               f"next round needs {round_cost})")
                break

            skill_before = state.skill

            # 1. Collect with the CURRENT policy; curate.
            collected = await self._collect(state, config)
            curated_quality = float(collected["curated_success_rate"])

            # 2. Train — skill is state; this round's update feeds next
            #    round's collection.
            state.skill = self._train(state, curated_quality, config)

            # 3. Probe the updated policy on fresh rollouts (ground truth).
            measured = await self._probe(state, config)

            kl = round(0.30 * (state.skill - skill_before) / (0.01 + float(config["kl_coef"]) * 4), 4)
            record = RoundRecord(
                iteration=it,
                skill_before=round(skill_before, 4),
                skill_after=round(state.skill, 4),
                reward=curated_quality,
                success_rate=round(measured, 4),
                kl_divergence=max(0.0, kl),
                loss=round(1.5 * math.exp(-3.0 * state.skill) + 0.30, 4),
                rollouts_used=state.budget_used,
                decision="", decision_reason="",
            )
            state.rounds.append(record)

            # 4. Track best / plateau. A new best resets the supervisor's
            #    escalation state — earlier interventions evidently worked.
            if measured > state.best_success + 0.01:
                state.best_success = measured
                state.best_skill = state.skill
                state.rounds_without_gain = 0
                for k in ("_escalations", "_rolled_back"):
                    config.pop(k, None)
            else:
                state.rounds_without_gain += 1

            # 5. Audit reward-vs-measured-success alignment.
            audit = self.detector.audit(
                rewards=[r.reward for r in state.rounds],
                evals=[r.success_rate for r in state.rounds],
            ) if len(state.rounds) >= 2 else {"severity": "low"}

            # 6. Supervisor decides the NEXT round — and its patch is applied.
            decision = self.supervisor.decide_round(
                round_metrics={"task_success_rate": measured, "audit": audit},
                best_success=state.best_success,
                target_success=target_success,
                rounds_without_gain=state.rounds_without_gain,
                budget_remaining=budget_rollouts - state.budget_used,
                config=config,
            )
            record.decision = decision.action
            record.decision_reason = decision.reason

            patch = dict(decision.config_patch)
            if patch.pop("restore_best", False):
                state.skill = state.best_skill        # rollback is real
            config.update(patch)

            print(f"  round {it:2d} | skill {skill_before:.2f}→{state.skill:.2f} "
                  f"| success {measured:.0%} (best {state.best_success:.0%}) "
                  f"| reward {curated_quality:.2f} | kl_coef {config['kl_coef']} "
                  f"| budget {state.budget_used}/{budget_rollouts} "
                  f"| {decision.action}: {decision.reason}")

            if decision.action == "stop":
                stop_reason = decision.reason
                break

        achieved = state.best_success >= target_success
        final = state.rounds[-1] if state.rounds else None
        run_summary = {
            "goal": f"autonomous: reach {target_success:.0%} task success",
            "plan": "autonomous-loop",
            "stages": ["collect", "train", "probe", "decide"] ,
            "results": {
                "autonomous_loop": {
                    "technique": "expert_iteration",
                    "iterations": len(state.rounds),
                    "final_loss": final.loss if final else None,
                    "final_reward": final.reward if final else None,
                    "kl_divergence": final.kl_divergence if final else None,
                    "task_success_rate": state.best_success,
                    "success_rate_lift": round(state.best_success - (state.rounds[0].success_rate if state.rounds else 0), 3),
                    "target_success": target_success,
                    "target_achieved": achieved,
                    "stop_reason": stop_reason,
                    "budget_used": state.budget_used,
                    "budget_total": budget_rollouts,
                    "final_skill": round(state.skill, 4),
                    "iteration_metrics": [r.to_iteration_metrics() for r in state.rounds],
                    "rounds": [r.__dict__ for r in state.rounds],
                },
            },
            "interventions": [i.__dict__ for i in self.supervisor.interventions],
            "elapsed_s": round(time.time() - t0, 2),
        }

        report = await self.reporter.execute(run=run_summary, out_dir=self.out_dir)
        run_summary["report"] = report

        icon = "🎯" if achieved else "⏹"
        print(f"\n{BOLD}{icon} {stop_reason} — best success {state.best_success:.0%} "
              f"in {len(state.rounds)} rounds, {state.budget_used} rollouts{RESET}\n")
        return run_summary
