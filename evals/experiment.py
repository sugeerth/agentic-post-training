"""An experiment = hypothesis + baseline vs candidate + envs + a ship gate.

    cfg = ExperimentConfig.from_yaml("experiments/abstention.yaml")
    result = run_experiment(cfg)
    out_dir = write_artifacts(result)        # runs/<name>/{report.md, ...}

Every run writes the same artifacts, so any result can be audited and
reproduced from its manifest alone.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from core.config import BaseConfig
from evals.environment import run_episode
from evals.envs import ENVIRONMENTS, get_env
from evals.failures import to_preference_pairs
from evals.gate import Decision, decide
from evals.policies import build_policy
from evals.report import render_report
from evals.types import Grade, Task, Trajectory

ARMS = ("baseline", "candidate")


class PolicySpec(BaseConfig):
    kind: Literal["simulated", "callable"] = Field(default="simulated", description="Policy type.")
    defects: dict[str, float] = Field(default_factory=dict, description="Simulated defect rates.")
    target: str | None = Field(default=None, description="`module:function` for callable policies.")


class GateSpec(BaseConfig):
    min_gain: float = Field(default=0.05, ge=0, description="Required improvement on target envs.")
    max_regression: float = Field(default=0.02, ge=0, description="Tolerated drop on any env.")


class ExperimentConfig(BaseConfig):
    name: str = Field(..., description="Run name; also the output folder.")
    hypothesis: str = Field(..., description="What we expect to change, and why.")
    envs: list[str] = Field(default_factory=lambda: sorted(ENVIRONMENTS))
    targets: list[str] = Field(..., description="Envs the intervention should improve.")
    baseline: PolicySpec
    candidate: PolicySpec
    seeds: list[int] = Field(default_factory=lambda: [0, 1, 2])
    gate: GateSpec = Field(default_factory=GateSpec)
    output_dir: str = "runs"


@dataclass(frozen=True)
class ExperimentResult:
    config: ExperimentConfig
    tasks: dict[str, list[Task]]  # env -> tasks
    trajectories: dict[str, list[Trajectory]]  # arm -> aligned across arms
    grades: dict[str, list[Grade]]  # arm -> aligned across arms
    decision: Decision


def run_experiment(cfg: ExperimentConfig) -> ExperimentResult:
    unknown = set(cfg.targets) - set(cfg.envs)
    if unknown:
        raise ValueError(f"targets {sorted(unknown)} are not in envs")
    envs = {name: get_env(name) for name in cfg.envs}
    tasks = {name: env.tasks() for name, env in envs.items()}
    trajs: dict[str, list[Trajectory]] = {arm: [] for arm in ARMS}
    grades: dict[str, list[Grade]] = {arm: [] for arm in ARMS}
    for arm, spec in zip(ARMS, (cfg.baseline, cfg.candidate), strict=True):
        for seed in cfg.seeds:
            for name, env in envs.items():
                policy = build_policy(
                    spec.kind, tasks[name], seed=seed, defects=spec.defects, target=spec.target
                )
                for task in tasks[name]:
                    traj = run_episode(env, task, policy)
                    trajs[arm].append(traj)
                    grades[arm].append(env.grade(task, traj))
    scores = {
        arm: {env: [g.score for g in grades[arm] if g.env == env] for env in cfg.envs}
        for arm in ARMS
    }
    decision = decide(
        scores["baseline"],
        scores["candidate"],
        targets=cfg.targets,
        min_gain=cfg.gate.min_gain,
        max_regression=cfg.gate.max_regression,
    )
    return ExperimentResult(cfg, tasks, trajs, grades, decision)


def _git_sha() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
        return out.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def _write_jsonl(path: Path, rows: Sequence[Any]) -> None:
    """Write dataclass rows as JSON lines."""
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(asdict(row)) + "\n")


def write_artifacts(result: ExperimentResult, out_dir: str | Path | None = None) -> Path:
    cfg = result.config
    out = Path(out_dir or Path(cfg.output_dir) / cfg.name)
    out.mkdir(parents=True, exist_ok=True)
    cfg_json = cfg.model_dump_json()
    manifest = {
        "name": cfg.name,
        "config": json.loads(cfg_json),
        "config_sha256": hashlib.sha256(cfg_json.encode()).hexdigest(),
        "git_sha": _git_sha(),
        "python": platform.python_version(),
        "episodes_per_arm": len(result.grades["baseline"]),
        "verdict": result.decision.verdict,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for arm in ARMS:
        _write_jsonl(out / f"{arm}.grades.jsonl", result.grades[arm])
        _write_jsonl(out / f"{arm}.trajectories.jsonl", result.trajectories[arm])
    all_tasks = [t for ts in result.tasks.values() for t in ts]
    pairs = to_preference_pairs(
        all_tasks, result.trajectories["candidate"], result.grades["candidate"]
    )
    _write_jsonl(out / "preference_pairs.jsonl", pairs)
    (out / "report.md").write_text(render_report(result, n_pairs=len(pairs)))
    return out
