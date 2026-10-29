"""Tests for the agentic post-training layer.

Covers:
  - ToolEnv semantics + task types
  - Reference policies (random ~ 0%, expert ~ 100%)
  - TrajectoryAgent rejection sampling
  - Reporter sparklines + regression detection
  - Reward-hacking detector (Spearman ρ + gap)
  - Bake-off winner + pareto frontier
  - RunHistory round-trip
  - Supervisor plan routing
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

import pytest

from agents.bakeoff_agent import BakeoffAgent, BakeoffEntry
from agents.reporter_agent import ReporterAgent, sparkline
from agents.reward_hacking_detector import RewardHackingDetector, spearman_rho
from agents.supervisor_agent import SupervisorAgent
from agents.trajectory_agent import TrajectoryAgent
from environments.policies import POLICIES
from environments.tool_env import ToolEnv, sample_tasks
from pipeline.run_history import RunHistory


# --------------------------------------------------------------------------- #
# ToolEnv + policies
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("kind_index", range(4))
def test_toolenv_task_kinds_all_have_answer(kind_index):
    tasks = sample_tasks(n=8, seed=1)
    assert tasks[kind_index].hidden.get("answer") is not None


def test_finish_correct_answer_gets_reward_one():
    tasks = sample_tasks(n=4, seed=0)
    env = ToolEnv(tasks)
    task = tasks[0]
    env.reset(task)
    r = env.step("finish", {"answer": str(task.hidden["answer"])})
    assert r.done is True
    assert r.reward == pytest.approx(1.0)


def test_finish_wrong_answer_gets_zero():
    tasks = sample_tasks(n=4, seed=0)
    env = ToolEnv(tasks)
    env.reset(tasks[0])
    r = env.step("finish", {"answer": "definitely wrong"})
    assert r.done is True
    assert r.reward == pytest.approx(0.0)


def test_expert_policy_is_much_better_than_random():
    """Baseline vs skilled play: policies must produce a real skill gap."""
    def _success(policy_name: str) -> float:
        env = ToolEnv(sample_tasks(n=40, seed=7))
        wins = 0
        for task in env.tasks:
            goal = env.reset(task)
            done, hist = False, []
            while not done:
                tool, args = POLICIES[policy_name](goal, hist)
                r = env.step(tool, args)
                hist.append({"tool": tool, "args": args, "obs": r.obs, "reward": r.reward})
                done = r.done
            if any(h["tool"] == "finish" and h["reward"] >= 1.0 for h in hist):
                wins += 1
        return wins / len(env.tasks)

    expert = _success("expert")
    rand = _success("random")
    assert expert >= 0.9, f"expert should nail ≥90%, got {expert:.0%}"
    assert rand <= 0.1, f"random should be near-zero, got {rand:.0%}"


# --------------------------------------------------------------------------- #
# TrajectoryAgent
# --------------------------------------------------------------------------- #

def test_trajectory_agent_curates_high_quality_first():
    agent = TrajectoryAgent()
    result = asyncio.run(agent.run(num_rollouts=32, p_expert=0.7, seed=3))
    # Curated success rate should be at least as high as raw sampled rate.
    assert result["curated_success_rate"] >= result["success_rate"]
    assert 0 < result["curated"] <= result["sampled"]


# --------------------------------------------------------------------------- #
# Sparklines
# --------------------------------------------------------------------------- #

def test_sparkline_empty_and_constant():
    assert sparkline([]) == ""
    line = sparkline([0.5, 0.5, 0.5])
    assert len(line) == 3
    assert len(set(line)) == 1  # all the same char


def test_sparkline_ascending_uses_full_range():
    line = sparkline([0.0, 0.25, 0.5, 0.75, 1.0])
    assert line[0] == " "     # low end of spectrum
    assert line[-1] == "█"    # high end


# --------------------------------------------------------------------------- #
# Reward-hacking detector
# --------------------------------------------------------------------------- #

def test_spearman_rho_monotone_up_is_one():
    assert spearman_rho([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)


def test_spearman_rho_monotone_down_is_minus_one():
    assert spearman_rho([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)


def test_hacking_detector_flags_reward_up_eval_down():
    det = RewardHackingDetector()
    audit = det.audit(
        rewards=[0.2, 0.5, 0.8, 0.95],   # goes up
        evals  =[0.6, 0.55, 0.4, 0.35],  # goes down
    )
    assert audit["severity"] == "high"
    assert audit["spearman_rho"] < 0


def test_hacking_detector_says_ok_when_aligned():
    det = RewardHackingDetector()
    audit = det.audit(
        rewards=[0.2, 0.4, 0.6, 0.8],
        evals  =[0.3, 0.5, 0.7, 0.9],
    )
    assert audit["severity"] == "low"
    assert audit["spearman_rho"] > 0.9


# --------------------------------------------------------------------------- #
# Bake-off
# --------------------------------------------------------------------------- #

def test_bakeoff_picks_highest_success_under_ceiling():
    b = BakeoffAgent()
    entries = [
        BakeoffEntry(plan="A", success_rate=0.9, reward=0.8, kl_divergence=0.30, elapsed_s=1),
        BakeoffEntry(plan="B", success_rate=0.7, reward=0.6, kl_divergence=0.10, elapsed_s=1),
        BakeoffEntry(plan="C", success_rate=0.8, reward=0.7, kl_divergence=0.15, elapsed_s=1),
    ]
    winner = b.pick_winner(entries, kl_ceiling=0.2)
    # A blows past the ceiling, so C wins over B.
    assert winner.plan == "C"


def test_bakeoff_falls_back_to_overall_best_if_all_over_ceiling():
    b = BakeoffAgent()
    entries = [
        BakeoffEntry(plan="A", success_rate=0.9, reward=0.8, kl_divergence=0.4, elapsed_s=1),
        BakeoffEntry(plan="B", success_rate=0.7, reward=0.6, kl_divergence=0.3, elapsed_s=1),
    ]
    winner = b.pick_winner(entries, kl_ceiling=0.2)
    assert winner.plan == "A"


def test_pareto_frontier_drops_dominated():
    b = BakeoffAgent()
    entries = [
        BakeoffEntry(plan="A", success_rate=0.9, reward=0.8, kl_divergence=0.10, elapsed_s=1),  # frontier
        BakeoffEntry(plan="B", success_rate=0.7, reward=0.6, kl_divergence=0.05, elapsed_s=1),  # frontier
        BakeoffEntry(plan="C", success_rate=0.8, reward=0.7, kl_divergence=0.15, elapsed_s=1),  # dominated by A
    ]
    front = {e.plan for e in b.pareto_frontier(entries)}
    assert front == {"A", "B"}


# --------------------------------------------------------------------------- #
# RunHistory
# --------------------------------------------------------------------------- #

def test_run_history_round_trip_and_deltas():
    with tempfile.TemporaryDirectory() as tmp:
        h = RunHistory(tmp)
        run1 = {
            "plan": "agentic-tool-use", "goal": "tool use",
            "results": {"multi_turn_grpo": {
                "task_success_rate": 0.6, "final_reward": 0.7, "kl_divergence": 0.10,
            }},
            "interventions": [], "elapsed_s": 2.0,
        }
        run2 = {**run1, "results": {"multi_turn_grpo": {
            "task_success_rate": 0.75, "final_reward": 0.85, "kl_divergence": 0.08,
        }}, "elapsed_s": 2.5}

        rec1 = h.append(run1)
        rec2 = h.append(run2)

        # Filenames are per-plan monotonic.
        assert rec1.n == 1 and rec2.n == 2
        assert h.latest_for("agentic-tool-use").n == 2

        # Deltas point in the right direction.
        deltas = h.deltas(rec2, rec1)
        assert deltas["task_success_rate"][2] == pytest.approx(0.15)
        assert deltas["kl_divergence"][2] == pytest.approx(-0.02)


# --------------------------------------------------------------------------- #
# Supervisor plan routing
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("goal,plan_name", [
    ("agentic tool use",   "agentic-tool-use"),
    ("reasoning math",     "reasoning-r1"),
    ("preference align",   "preference-alignment"),
    ("rft rejection",      "rejection-sampling-loop"),
])
def test_supervisor_routes_goals_to_plans(goal, plan_name):
    s = SupervisorAgent()
    plan = s.pick_plan(goal, {})
    assert plan.name == plan_name


def test_supervisor_intervenes_on_high_kl():
    s = SupervisorAgent()
    decision = s.review_stage("multi_turn_grpo", {"kl_divergence": 0.9, "final_reward": 0.7})
    assert decision.action == "adjust"


def test_supervisor_intervenes_on_reward_collapse():
    s = SupervisorAgent()
    decision = s.review_stage("multi_turn_grpo", {"kl_divergence": 0.05, "final_reward": 0.02})
    assert decision.action == "rollback"


# --------------------------------------------------------------------------- #
# Reporter — smoke test that the TL;DR renders
# --------------------------------------------------------------------------- #

def test_reporter_headline_metrics_and_next_action_agree_on_strongest_stage():
    """After the stage-routing fix, a plan running sft AND grpo AND distill
    all emit task_success_rate. Headline / metrics / next-action must all
    read from GRPO (highest priority), not from whichever stage is first
    in dict order (sft)."""
    with tempfile.TemporaryDirectory() as tmp:
        reporter = ReporterAgent(history=RunHistory(Path(tmp) / ".runs"))
        run = {
            "plan": "reasoning-r1", "goal": "reasoning math",
            "results": {
                # Order matches the pipeline's plan order — sft appears before grpo.
                "sft":     {"task_success_rate": 0.59, "final_reward": 0.78, "kl_divergence": 0.0,
                            "success_rate_lift": 0.24,
                            "iteration_metrics": [{"reward": 0.5, "loss": 1.0, "success_rate": 0.5, "kl_divergence": 0.0}]},
                "grpo":    {"task_success_rate": 0.68, "final_reward": 0.95, "kl_divergence": 0.03,
                            "success_rate_lift": 0.33,
                            "iteration_metrics": [{"reward": 0.5, "loss": 1.0, "success_rate": 0.5, "kl_divergence": 0.05}]},
                "distill": {"task_success_rate": 0.53, "final_reward": 0.66, "kl_divergence": 0.0,
                            "success_rate_lift": 0.18,
                            "iteration_metrics": [{"reward": 0.4, "loss": 1.1, "success_rate": 0.4, "kl_divergence": 0.0}]},
            },
            "interventions": [], "elapsed_s": 1.0,
        }
        result = asyncio.run(reporter.run(run=run, out_dir=tmp, persist=False))
        tldr = result["tldr"]
        # Headline picks grpo's 68% — not sft's 59% or distill's 53%.
        assert "68%" in tldr
        # Metrics line: reward = grpo's 0.95, success = grpo's 68%.
        assert "reward `0.95`" in tldr
        assert "success `68%`" in tldr
        assert "reward `0.78`" not in tldr  # sft's reward would leak here
        # Next-action: grpo passes 60% → "Ship the checkpoint" (not the
        # "under 60%" branch that sft alone would trigger).
        assert "Ship the checkpoint" in tldr
        assert "under 60%" not in tldr


def test_reporter_renders_tldr_with_deltas_and_sparklines():
    with tempfile.TemporaryDirectory() as tmp:
        history = RunHistory(Path(tmp) / ".runs")
        reporter = ReporterAgent(history=history)

        # Seed one prior run so the second one has a baseline.
        run1 = {
            "plan": "agentic-tool-use", "goal": "tool use",
            "results": {"multi_turn_grpo": {
                "task_success_rate": 0.5, "final_reward": 0.5, "kl_divergence": 0.15,
                "iteration_metrics": [
                    {"reward": 0.3, "loss": 1.5, "success_rate": 0.3, "kl_divergence": 0.20},
                    {"reward": 0.5, "loss": 1.0, "success_rate": 0.5, "kl_divergence": 0.15},
                ],
            }},
            "interventions": [], "elapsed_s": 1.0,
        }
        run2 = {
            "plan": "agentic-tool-use", "goal": "tool use",
            "results": {"multi_turn_grpo": {
                "task_success_rate": 0.8, "final_reward": 0.85, "kl_divergence": 0.10,
                "success_rate_lift": 0.3,
                "iteration_metrics": [
                    {"reward": 0.5, "loss": 1.0, "success_rate": 0.5, "kl_divergence": 0.15},
                    {"reward": 0.7, "loss": 0.7, "success_rate": 0.7, "kl_divergence": 0.12},
                    {"reward": 0.85, "loss": 0.5, "success_rate": 0.8, "kl_divergence": 0.10},
                ],
            }},
            "interventions": [], "elapsed_s": 1.5,
        }

        asyncio.run(reporter.run(run=run1, out_dir=tmp))
        result = asyncio.run(reporter.run(run=run2, out_dir=tmp))

        tldr = result["tldr"]
        assert "# TL;DR — agentic-tool-use" in tldr
        assert "vs last run" in tldr          # delta section appeared
        assert "Curves" in tldr               # sparkline section appeared
        assert result["deltas"] is not None
        assert result["deltas"]["task_success_rate"][2] == pytest.approx(0.3)
