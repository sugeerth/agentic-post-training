"""Behavior evals for agentic models.

    envs/          one module per behavior (tasks + world + grader)
    environment    the Environment contract and the shared episode loop
    policies       models under test (real via CallablePolicy, or simulated)
    metrics        bootstrap CIs, paired deltas, ECE, Brier
    gate           SHIP / HOLD / BLOCK readiness decision
    failures       failure counts and failure -> preference-pair mining
    experiment     config + runner + artifacts
    report         Markdown report rendering

Dependency direction: types <- environment <- envs <- policies <- experiment.
"""

from evals.environment import Environment, Policy, run_episode
from evals.types import Action, Grade, Observation, Task, ToolCall, ToolSpec, Trajectory, Turn

__all__ = [
    "Action",
    "Environment",
    "Grade",
    "Observation",
    "Policy",
    "Task",
    "ToolCall",
    "ToolSpec",
    "Trajectory",
    "Turn",
    "run_episode",
]
