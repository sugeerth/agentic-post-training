"""Tests for the autonomous closed loop, agent registry, and real interventions.

The properties under test are the ones that make the loop agentic:
  • closed feedback — skill state carries between rounds
  • goal-driven termination — target / budget / convergence, never max_rounds
    on a well-formed run
  • real interventions — supervisor patches actually mutate the next round
  • bounded escalation — a broken run stops early instead of thrashing
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from agents.base_agent import BaseAgent
from agents.registry import AGENTS, build, register_agent
from agents.supervisor_agent import SupervisorAgent
from pipeline.autonomous_loop import AutonomousLoop


# --------------------------------------------------------------------------- #
# Agent registry
# --------------------------------------------------------------------------- #

def test_registry_has_all_builtin_roles():
    for role in ("supervisor", "coordinator", "trajectory", "reward_model",
                 "agentic_trainer", "evaluator", "reporter",
                 "reward_hacking_detector", "bakeoff",
                 "paper_scanner", "technique_scout", "mimic_writer"):
        assert role in AGENTS, f"missing role: {role}"


def test_registry_build_returns_instance():
    agent = build("supervisor")
    assert isinstance(agent, BaseAgent)
    assert agent.role == "supervisor"


def test_registry_replace_swaps_specialist():
    class FakeTrainer(BaseAgent):
        def __init__(self):
            super().__init__("FakeTrainer", role="agentic_trainer")
        async def run(self, **kwargs):
            return {"fake": True}
        async def step(self, **kwargs):
            return {"fake": True}

    original = AGENTS.get("agentic_trainer")
    try:
        register_agent("agentic_trainer", replace=True)(FakeTrainer)
        assert isinstance(build("agentic_trainer"), FakeTrainer)
    finally:
        AGENTS.register("agentic_trainer", original, replace=True)


# --------------------------------------------------------------------------- #
# Supervisor decide_round — decisions carry real patches
# --------------------------------------------------------------------------- #

def _decide(sup: SupervisorAgent, **overrides):
    defaults = dict(
        round_metrics={"task_success_rate": 0.5, "audit": {"severity": "low"}},
        best_success=0.5, target_success=0.85,
        rounds_without_gain=0, budget_remaining=1000,
        config={"kl_coef": 0.05, "num_rollouts": 64, "keep_top_frac": 0.5},
    )
    defaults.update(overrides)
    return sup.decide_round(**defaults)


def test_decide_stops_on_target():
    d = _decide(SupervisorAgent(),
                round_metrics={"task_success_rate": 0.9, "audit": {"severity": "low"}})
    assert d.action == "stop" and "target met" in d.reason


def test_decide_stops_on_budget():
    d = _decide(SupervisorAgent(), budget_remaining=0)
    assert d.action == "stop" and "budget" in d.reason


def test_decide_rollback_on_hacking_carries_restore_and_capped_kl():
    d = _decide(SupervisorAgent(),
                round_metrics={"task_success_rate": 0.5,
                               "audit": {"severity": "high", "spearman_rho": -0.8}},
                config={"kl_coef": 0.3, "num_rollouts": 64, "keep_top_frac": 0.5})
    assert d.action == "rollback"
    assert d.config_patch["restore_best"] is True
    assert d.config_patch["kl_coef"] == SupervisorAgent.KL_COEF_CAP  # 0.6 capped to 0.4


def test_decide_medium_at_kl_cap_does_not_spam_adjust():
    d = _decide(SupervisorAgent(),
                round_metrics={"task_success_rate": 0.5, "audit": {"severity": "medium"}},
                config={"kl_coef": SupervisorAgent.KL_COEF_CAP,
                        "num_rollouts": 64, "keep_top_frac": 0.5})
    assert d.action == "continue"


def test_decide_escalation_budget_exhausted_stops():
    d = _decide(SupervisorAgent(),
                rounds_without_gain=5,
                config={"kl_coef": 0.05, "num_rollouts": 64, "keep_top_frac": 0.5,
                        "_escalations": SupervisorAgent.ESCALATION_BUDGET})
    assert d.action == "stop" and "despite" in d.reason


def test_decide_plateau_boosts_exploration_with_real_patch():
    d = _decide(SupervisorAgent(), rounds_without_gain=3)
    assert d.action == "adjust"
    assert d.config_patch["num_rollouts"] == 128       # doubled
    assert d.config_patch["keep_top_frac"] == 0.35     # loosened


# --------------------------------------------------------------------------- #
# The loop itself
# --------------------------------------------------------------------------- #

def _run_loop(tmp: Path, **kwargs) -> dict:
    loop = AutonomousLoop(out_dir=str(tmp / "out"), history_dir=str(tmp / "runs"))
    summary = asyncio.run(loop.run(**kwargs))
    return summary["results"]["autonomous_loop"]


def test_loop_reaches_target_under_budget(tmp_path):
    r = _run_loop(tmp_path, target_success=0.8, budget_rollouts=2000, max_rounds=20)
    assert r["target_achieved"] is True
    assert "target met" in r["stop_reason"]
    assert r["budget_used"] <= 2000


def test_loop_skill_state_carries_between_rounds(tmp_path):
    """Closed feedback: every round's skill_before equals the previous
    round's skill_after (modulo rollback, absent on a healthy run)."""
    r = _run_loop(tmp_path, target_success=0.8, budget_rollouts=2000, max_rounds=20)
    rounds = r["rounds"]
    assert len(rounds) >= 2
    for prev, curr in zip(rounds, rounds[1:]):
        assert curr["skill_before"] == prev["skill_after"]
    # And skill actually improved.
    assert rounds[-1]["skill_after"] > rounds[0]["skill_before"]


def test_loop_never_overspends_budget(tmp_path):
    r = _run_loop(tmp_path, target_success=0.999, budget_rollouts=300, max_rounds=20)
    assert r["budget_used"] <= 300
    assert "budget" in r["stop_reason"]


def test_loop_broken_config_converges_early_not_max_rounds(tmp_path):
    """learning_gain=0 → the policy can't improve. The supervisor must
    conclude 'converged' well before max_rounds instead of thrashing."""
    r = _run_loop(tmp_path, target_success=0.9, budget_rollouts=50_000, max_rounds=30,
                  config={"learning_gain": 0.0, "seed": 5})
    assert r["target_achieved"] is False
    assert r["iterations"] < 30
    assert "converged" in r["stop_reason"] or "no improvement" in r["stop_reason"]


def test_loop_report_headline_uses_autonomous_stage(tmp_path):
    loop = AutonomousLoop(out_dir=str(tmp_path / "out"), history_dir=str(tmp_path / "runs"))
    summary = asyncio.run(loop.run(target_success=0.8, budget_rollouts=2000))
    tldr = summary["report"]["tldr"]
    assert "via autonomous_loop" in tldr        # reporter recognizes the stage
    assert "Curves" in tldr                     # sparklines rendered from rounds


def test_loop_target_met_is_not_an_anomaly(tmp_path):
    """A stop-on-target run must NOT render 'Needs attention' or
    'Address loop — target met' — reaching the goal is success. Caught in
    the first CI run of the autonomous job."""
    loop = AutonomousLoop(out_dir=str(tmp_path / "out"), history_dir=str(tmp_path / "runs"))
    summary = asyncio.run(loop.run(target_success=0.8, budget_rollouts=2000))
    tldr = summary["report"]["tldr"]
    assert "Needs attention" not in tldr
    assert "Address `loop`" not in tldr
    assert "Target met" in tldr and "ship the checkpoint" in tldr
