"""Agent registry — construct agents by role name, not hard-wired imports.

Built on `core.registry.Registry` (same safety properties: unique names,
loud KeyError with alternatives). Every agent module self-registers at
import time via `@register_agent("role")`; pipelines then do

    from agents.registry import build
    trainer = build("agentic_trainer")

which means a downstream user can swap any specialist by registering a
replacement under the same role — no pipeline edits, no forks:

    @register_agent("agentic_trainer", replace=True)
    class MyTrainer(BaseAgent): ...

`build_all(roles)` constructs a full crew in one call and is what
`AgenticPipeline` / `AutonomousLoop` use.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from core.registry import Registry
from agents.base_agent import BaseAgent

T = TypeVar("T", bound=type)

AGENTS: Registry = Registry("agent")


def register_agent(role: str, *, replace: bool = False) -> Callable[[T], T]:
    def _decorator(cls: T) -> T:
        AGENTS.register(role, cls, replace=replace)
        return cls
    return _decorator


def build(role: str, **kwargs) -> BaseAgent:
    """Instantiate the registered agent class for `role`."""
    cls = AGENTS.get(role)
    return cls(**kwargs)


def build_all(roles: list[str], **shared_kwargs) -> dict[str, BaseAgent]:
    return {role: build(role) for role in roles}


def _register_builtins() -> None:
    """Import the built-in agents so their classes land in the registry.

    Imports live here (not module top-level) to avoid import cycles:
    agent modules import `register_agent` from this module.
    """
    from agents.supervisor_agent import SupervisorAgent
    from agents.coordinator import CoordinatorAgent
    from agents.trajectory_agent import TrajectoryAgent
    from agents.reward_model_agent import RewardModelAgent
    from agents.agentic_training_agent import AgenticTrainingAgent
    from agents.evaluation_agent import EvaluationAgent
    from agents.reporter_agent import ReporterAgent
    from agents.reward_hacking_detector import RewardHackingDetector
    from agents.bakeoff_agent import BakeoffAgent
    from agents.paper_scanner import PaperScanner
    from agents.technique_scout import TechniqueScout
    from agents.mimic_writer import MimicWriter

    builtins = {
        "supervisor": SupervisorAgent,
        "coordinator": CoordinatorAgent,
        "trajectory": TrajectoryAgent,
        "reward_model": RewardModelAgent,
        "agentic_trainer": AgenticTrainingAgent,
        "evaluator": EvaluationAgent,
        "reporter": ReporterAgent,
        "reward_hacking_detector": RewardHackingDetector,
        "bakeoff": BakeoffAgent,
        "paper_scanner": PaperScanner,
        "technique_scout": TechniqueScout,
        "mimic_writer": MimicWriter,
    }
    for role, cls in builtins.items():
        if role not in AGENTS:
            AGENTS.register(role, cls)


_register_builtins()

__all__ = ["AGENTS", "build", "build_all", "register_agent"]
