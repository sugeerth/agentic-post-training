"""Policies under test.

- `CallablePolicy`  wraps any function `Observation -> Action`. This is how
                    you plug in a real model (HF, vLLM, an API, a harness).
- `SimulatedPolicy` is a test double: it replays the reference solution and
                    injects named defects at configured rates. It exists so
                    the harness (envs, graders, gate, reports) can be tested
                    and demoed end-to-end without a GPU. Its numbers say
                    nothing about any real model.
"""

from __future__ import annotations

import importlib
import random
from collections.abc import Callable, Mapping

from evals.environment import Policy
from evals.envs.factuality import ABSTAIN
from evals.types import Action, Observation, Task, ToolCall

# --------------------------------------------------------------------------- #
# Defects: each is `Action -> Action`, returning the input unchanged when it
# doesn't apply. They model failure modes we see in real agents.
# --------------------------------------------------------------------------- #


def _drop_arg(a: Action) -> Action:
    if not a.tool_call or not a.tool_call.arguments:
        return a
    args = dict(a.tool_call.arguments)
    args.pop(next(reversed(args)))
    return Action(tool_call=ToolCall(a.tool_call.name, args))


def _stringify_arg(a: Action) -> Action:
    if not a.tool_call:
        return a
    args = dict(a.tool_call.arguments)
    for k, v in args.items():
        if isinstance(v, int | float) and not isinstance(v, bool):
            args[k] = str(v)
            return Action(tool_call=ToolCall(a.tool_call.name, args))
    return a


def _hallucinate_tool(a: Action) -> Action:
    if not a.tool_call:
        return a
    return Action(tool_call=ToolCall(f"{a.tool_call.name}_v2", a.tool_call.arguments))


def _answer_early(a: Action) -> Action:
    return Action(text="not sure, probably resolved") if a.tool_call else a


def _overconfident_guess(a: Action) -> Action:
    return Action(text="It was 42.", confidence=0.9) if a.text == ABSTAIN else a


def _over_abstain(a: Action) -> Action:
    if a.confidence is None or a.text == ABSTAIN:
        return a
    return Action(text=ABSTAIN, confidence=0.6)


def _miscalibrated(a: Action) -> Action:
    return Action(a.text, a.tool_call, 0.99) if a.confidence is not None else a


def _verbose_tail(a: Action) -> Action:
    if a.tool_call or a.confidence is not None or not a.text:
        return a
    return Action(text=a.text + "\nHope this helps, let me know if you need more!")


def _buggy_code(a: Action) -> Action:
    if "```python" not in a.text:
        return a
    lines = a.text.rstrip("`\n").splitlines()
    return Action(text="\n".join(lines[:-1]) + "\n```")


DEFECTS: dict[str, Callable[[Action], Action]] = {
    "drop_arg": _drop_arg,
    "stringify_arg": _stringify_arg,
    "hallucinate_tool": _hallucinate_tool,
    "answer_early": _answer_early,
    "overconfident_guess": _overconfident_guess,
    "over_abstain": _over_abstain,
    "miscalibrated": _miscalibrated,
    "verbose_tail": _verbose_tail,
    "buggy_code": _buggy_code,
}

# Defects that only make sense mid-trajectory (step > 0).
_MID_EPISODE_ONLY = {"answer_early"}


class SimulatedPolicy:
    """Replays `task.reference`, corrupting each step with independent defects.

    Deterministic in (seed, task_id, step). Retries a reference step after a
    tool error, so tool-error recovery is exercised too.
    """

    def __init__(self, tasks: list[Task], defects: Mapping[str, float], seed: int = 0) -> None:
        unknown = set(defects) - set(DEFECTS)
        if unknown:
            raise KeyError(f"Unknown defects {sorted(unknown)}. Available: {sorted(DEFECTS)}")
        self._refs = {t.id: t.reference for t in tasks}
        self._defects = dict(defects)
        self._seed = seed

    def act(self, obs: Observation) -> Action:
        ref = self._refs[obs.task_id]
        done = sum(1 for t in obs.history if not (t.result or "").startswith("error"))
        action = ref[min(done, len(ref) - 1)]
        step = len(obs.history)
        for name, rate in self._defects.items():
            if step == 0 and name in _MID_EPISODE_ONLY:
                continue
            if random.Random(f"{self._seed}:{obs.task_id}:{step}:{name}").random() < rate:
                action = DEFECTS[name](action)
        return action


class CallablePolicy:
    """Adapter: any `Observation -> Action` function becomes a Policy."""

    def __init__(self, fn: Callable[[Observation], Action]) -> None:
        self._fn = fn

    def act(self, obs: Observation) -> Action:
        return self._fn(obs)

    @classmethod
    def from_path(cls, target: str) -> CallablePolicy:
        """Load `package.module:function`."""
        module, _, attr = target.partition(":")
        return cls(getattr(importlib.import_module(module), attr))


def build_policy(
    kind: str,
    tasks: list[Task],
    *,
    seed: int,
    defects: Mapping[str, float] | None = None,
    target: str | None = None,
) -> Policy:
    if kind == "simulated":
        return SimulatedPolicy(tasks, defects or {}, seed)
    if kind == "callable":
        if not target:
            raise ValueError("callable policy needs `target: module:function`")
        return CallablePolicy.from_path(target)
    raise ValueError(f"Unknown policy kind '{kind}'")
