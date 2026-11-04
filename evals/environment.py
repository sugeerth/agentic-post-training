"""The one environment contract, plus the one episode loop.

An environment owns three things and nothing else:
  1. its tasks,
  2. how it responds to a tool call (the "world"),
  3. how it grades a finished trajectory.

The loop is shared by every environment, so single-turn function calling
and 8-step tool chains run through identical code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Protocol

from evals.types import Action, Grade, Observation, Task, Trajectory, Turn


class Policy(Protocol):
    """Anything that maps an observation to an action: an LLM, a script, a human."""

    def act(self, obs: Observation) -> Action: ...


class Environment(ABC):
    name: ClassVar[str]
    max_steps: ClassVar[int] = 1

    @abstractmethod
    def tasks(self) -> list[Task]: ...

    def respond(self, task: Task, action: Action) -> str | None:
        """Result of a tool call, or None to end the episode. Default: single-turn."""
        return None

    @abstractmethod
    def grade(self, task: Task, trajectory: Trajectory) -> Grade: ...

    def max_steps_for(self, task: Task) -> int:
        return self.max_steps


def run_episode(env: Environment, task: Task, policy: Policy) -> Trajectory:
    history: list[Turn] = []
    for _ in range(env.max_steps_for(task)):
        obs = Observation(task.id, task.prompt, task.tools, tuple(history))
        action = policy.act(obs)
        result = env.respond(task, action) if action.tool_call else None
        history.append(Turn(action, result))
        if result is None:
            break
    return Trajectory(task.id, env.name, tuple(history))
