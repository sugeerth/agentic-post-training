"""Environments the agent trains against.

An environment is *step semantics*: `env.reset(task) → obs`, `env.step(action)
→ (obs, reward, done, info)`. Trajectories are what you get out of running
a policy against one.

This layer exists so `TrajectoryAgent` doesn't have to hand-fabricate
rollout dicts. Real deployments swap in a live tool environment (a Python
sandbox, a browser, a shell) without changing the trainer.
"""

from environments.tool_env import ToolEnv, ToolSpec, Task, StepResult

__all__ = ["StepResult", "Task", "ToolEnv", "ToolSpec"]
