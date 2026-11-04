"""Environment registry: name -> class. Add an env by adding one line here."""

from __future__ import annotations

from evals.environment import Environment
from evals.envs.coding import CodingEnv
from evals.envs.factuality import FactualityEnv
from evals.envs.function_calling import FunctionCallingEnv
from evals.envs.instruction_following import InstructionFollowingEnv
from evals.envs.multistep_tools import MultiStepToolsEnv

ENVIRONMENTS: dict[str, type[Environment]] = {
    env.name: env
    for env in (
        FunctionCallingEnv,
        MultiStepToolsEnv,
        CodingEnv,
        InstructionFollowingEnv,
        FactualityEnv,
    )
}


def get_env(name: str) -> Environment:
    if name not in ENVIRONMENTS:
        raise KeyError(f"Unknown env '{name}'. Available: {', '.join(sorted(ENVIRONMENTS))}")
    return ENVIRONMENTS[name]()
