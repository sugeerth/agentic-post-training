"""Tests for the behavior-eval stack: envs, graders, metrics, gate, experiments."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.__main__ import main as cli
from evals.environment import run_episode
from evals.envs import ENVIRONMENTS, get_env
from evals.envs._schema import schema_failure
from evals.envs.coding import run_tests
from evals.experiment import ExperimentConfig, run_experiment, write_artifacts
from evals.failures import to_preference_pairs
from evals.gate import decide
from evals.metrics import bootstrap_ci, brier, ece, paired_delta
from evals.policies import CallablePolicy, SimulatedPolicy
from evals.types import Action, ToolCall, ToolSpec


def _grades(env_name: str, defects: dict[str, float], seed: int = 0):
    env = get_env(env_name)
    tasks = env.tasks()
    policy = SimulatedPolicy(tasks, defects, seed)
    return [env.grade(t, run_episode(env, t, policy)) for t in tasks]


# ---------------------------------------------------------------- envs/graders


@pytest.mark.parametrize("env_name", sorted(ENVIRONMENTS))
def test_reference_solutions_pass_every_grader(env_name: str) -> None:
    """If this fails, a task or a grader is wrong — not the model."""
    grades = _grades(env_name, {})
    assert grades and all(g.passed for g in grades), [g for g in grades if not g.passed]


@pytest.mark.parametrize(
    ("env_name", "defect", "expected"),
    [
        ("function_calling", "drop_arg", {"missing_arg"}),
        ("function_calling", "stringify_arg", {"bad_arg_type"}),
        ("function_calling", "hallucinate_tool", {"hallucinated_tool"}),
        ("multistep_tools", "answer_early", {"answered_early"}),
        ("multistep_tools", "drop_arg", {"exceeded_budget"}),
        ("coding", "buggy_code", {"test_failure", "runtime_error"}),
        ("factuality", "overconfident_guess", {"hallucination"}),
        ("factuality", "over_abstain", {"over_abstain"}),
    ],
)
def test_graders_catch_each_defect(env_name: str, defect: str, expected: set[str]) -> None:
    failures = {g.failure for g in _grades(env_name, {defect: 1.0}) if not g.passed}
    assert failures and failures <= expected


def test_schema_rejects_bool_for_integer() -> None:
    spec = (ToolSpec("f", "", {"n": "integer"}),)
    assert schema_failure(ToolCall("f", {"n": True}), spec) == "bad_arg_type"
    assert schema_failure(ToolCall("f", {"n": 3}), spec) is None
    assert schema_failure(ToolCall("f", {"n": 3, "x": 1}), spec) == "unknown_arg"


def test_coding_sandbox_tags() -> None:
    assert run_tests("def f(:", "") == "syntax_error"
    assert run_tests("while True: pass", "") == "timeout"
    assert run_tests("x = 1", "assert x == 2") == "test_failure"
    assert run_tests("x = 1/0", "") == "runtime_error"


def test_simulated_policy_is_deterministic() -> None:
    defects = {"drop_arg": 0.3, "answer_early": 0.2}
    a = _grades("multistep_tools", defects, seed=7)
    b = _grades("multistep_tools", defects, seed=7)
    assert a == b


def test_unknown_defect_rejected() -> None:
    with pytest.raises(KeyError):
        SimulatedPolicy([], {"typo": 1.0})


def test_callable_policy_from_path() -> None:
    policy = CallablePolicy.from_path("tests.test_evals:_always_abstain")
    env = get_env("factuality")
    task = env.tasks()[-1]  # unanswerable
    assert env.grade(task, run_episode(env, task, policy)).passed


def _always_abstain(_obs: object) -> Action:
    return Action(text="I don't know", confidence=0.8)


# ---------------------------------------------------------------- metrics/gate


def test_calibration_metrics() -> None:
    assert ece([1.0, 0.0], [True, False]) == 0.0
    assert brier([1.0, 0.0], [True, False]) == 0.0
    assert ece([0.9] * 10, [True] * 5 + [False] * 5) == pytest.approx(0.4)


def test_bootstrap_and_paired_delta() -> None:
    lo, hi = bootstrap_ci([1.0] * 10)
    assert (lo, hi) == (1.0, 1.0)
    delta, lo, hi = paired_delta([0.0] * 20, [1.0] * 20)
    assert delta == lo == hi == 1.0
    with pytest.raises(ValueError):
        paired_delta([0.0], [0.0, 1.0])


@pytest.mark.parametrize(
    ("candidate", "verdict"),
    [
        ({"t": [1.0] * 20, "o": [1.0] * 20}, "SHIP"),
        ({"t": [0.0] * 20, "o": [1.0] * 20}, "HOLD"),
        ({"t": [1.0] * 20, "o": [0.0] * 20}, "BLOCK"),
    ],
)
def test_gate_verdicts(candidate: dict[str, list[float]], verdict: str) -> None:
    baseline = {"t": [0.0] * 20, "o": [1.0] * 20}
    decision = decide(baseline, candidate, targets=["t"], min_gain=0.05, max_regression=0.02)
    assert decision.verdict == verdict


# ---------------------------------------------------------------- experiments


def _config(**overrides: object) -> ExperimentConfig:
    base: dict[str, object] = {
        "name": "t",
        "hypothesis": "h",
        "envs": ["factuality", "function_calling"],
        "targets": ["factuality"],
        "seeds": [0],
        "baseline": {"defects": {"overconfident_guess": 1.0}},
        "candidate": {"defects": {}},
    }
    return ExperimentConfig.from_dict({**base, **overrides})


def test_experiment_end_to_end(tmp_path: Path) -> None:
    result = run_experiment(_config())
    assert result.decision.verdict == "SHIP"
    out = write_artifacts(result, tmp_path)
    for name in ("manifest.json", "report.md", "baseline.grades.jsonl", "preference_pairs.jsonl"):
        assert (out / name).exists()
    assert json.loads((out / "manifest.json").read_text())["verdict"] == "SHIP"
    assert "Verdict: SHIP" in (out / "report.md").read_text()


def test_failures_become_preference_pairs() -> None:
    result = run_experiment(_config())
    tasks = [t for ts in result.tasks.values() for t in ts]
    pairs = to_preference_pairs(tasks, result.trajectories["baseline"], result.grades["baseline"])
    assert len(pairs) == 4  # the 4 unanswerable questions
    assert all(p.metadata["failure"] == "hallucination" for p in pairs)
    assert all("I don't know" in p.chosen for p in pairs)


def test_config_validation() -> None:
    with pytest.raises(ValueError, match="not in envs"):
        run_experiment(_config(targets=["coding"]))
    with pytest.raises(ValueError):  # extra="forbid": typos are errors
        _config(typo_field=1)


def test_shipped_configs_parse() -> None:
    for path in sorted(Path("experiments").glob("*.yaml")):
        cfg = ExperimentConfig.from_yaml(path)
        assert set(cfg.targets) <= set(cfg.envs)


def test_cli_exit_code_is_gate(tmp_path: Path) -> None:
    cfg_path = tmp_path / "exp.yaml"
    cfg_path.write_text(_config().to_yaml())
    assert cli(["run", str(cfg_path), "--out", str(tmp_path / "runs")]) == 0


def test_cli_writes_site_summary(tmp_path: Path) -> None:
    cfg_path = tmp_path / "exp.yaml"
    cfg_path.write_text(_config().to_yaml())
    summary = tmp_path / "site" / "experiments.json"
    cli(["run", str(cfg_path), "--out", str(tmp_path / "runs"), "--summary", str(summary)])
    data = json.loads(summary.read_text())
    assert {e["name"] for e in data["environments"]} == set(ENVIRONMENTS)
    exp = data["experiments"][0]
    assert exp["verdict"] == "SHIP"
    assert {e["env"] for e in exp["envs"]} == {"factuality", "function_calling"}
    assert "calibration" in next(e for e in exp["envs"] if e["env"] == "factuality")["baseline"]
