"""ToolEnv — a small, deterministic tool-use environment.

Structurally mirrors what a real agent env looks like (Kimi K2's agent
harness, OpenAI's function-calling loop, LangChain's AgentExecutor):

  1. Task: goal + hidden state
  2. Turn loop: agent picks a tool + args → env runs it → returns obs + shaped reward
  3. Termination: agent emits `finish(answer)` OR turn budget exhausted
  4. Reward: sparse terminal (correct? 1 : 0) + optional per-step shaping

No LLM required — the "policy" here is just a strategy function
`(obs) -> (tool_name, args)`. That lets the TrajectoryAgent produce
concrete trajectories in seconds, and lets us unit-test credit assignment.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ToolSpec:
    """One tool the agent can call. `run(args, state) → (obs, delta_reward)`."""
    name: str
    description: str
    run: Callable[[dict, dict], tuple[str, float]]


@dataclass
class Task:
    """A single task instance."""
    task_id: str
    goal: str
    hidden: dict[str, Any] = field(default_factory=dict)   # ground truth
    max_turns: int = 8


@dataclass
class StepResult:
    obs: str
    reward: float
    done: bool
    info: dict[str, Any] = field(default_factory=dict)


class ToolEnv:
    """A small deterministic env with 4 tools and 4 task types.

    Tools:
        - search(query)      → hint string
        - calculate(expr)    → numeric answer
        - lookup(key)        → value from hidden state
        - finish(answer)     → terminates; reward = 1 if answer matches

    Task types (see `sample_tasks`):
        - "sum"      : goal="What is A+B?"      hidden={"A": _, "B": _, "answer": _}
        - "lookup"   : goal="Find the city..."  hidden={"key": _, "answer": _}
        - "twohop"   : goal="Chain lookup+calc" hidden={"key": _, "delta": _, "answer": _}
        - "distract" : as 'sum' but with 2 red-herring keys in the state
    """

    def __init__(self, tasks: list[Task] | None = None, seed: int = 0):
        self._rng = random.Random(seed)
        self.tasks = tasks or sample_tasks(n=64, seed=seed)
        self._task: Task | None = None
        self._turn = 0
        self._trace: list[dict] = []
        self.tools: dict[str, ToolSpec] = _default_tools()

    def reset(self, task: Task | None = None) -> str:
        """Start a new task. Returns the initial observation."""
        self._task = task or self._rng.choice(self.tasks)
        self._turn = 0
        self._trace = []
        return f"[TASK {self._task.task_id}] {self._task.goal}"

    def step(self, action_name: str, args: dict[str, Any] | None = None) -> StepResult:
        if self._task is None:
            raise RuntimeError("ToolEnv.step called before reset()")
        args = args or {}
        self._turn += 1

        tool = self.tools.get(action_name)
        if tool is None:
            self._trace.append({"turn": self._turn, "tool": action_name, "error": "unknown_tool"})
            return StepResult(obs=f"unknown tool: {action_name}", reward=-0.1, done=False)

        obs, delta = tool.run(args, self._task.hidden)
        self._trace.append({"turn": self._turn, "tool": action_name, "args": args, "obs": obs, "reward": delta})

        done = action_name == "finish" or self._turn >= self._task.max_turns
        info = {"turn": self._turn, "trace_len": len(self._trace)}
        return StepResult(obs=obs, reward=delta, done=done, info=info)

    @property
    def trace(self) -> list[dict]:
        return list(self._trace)


# --------------------------------------------------------------------------- #
# Default tool implementations
# --------------------------------------------------------------------------- #

def _tool_search(args: dict, state: dict) -> tuple[str, float]:
    q = str(args.get("query", "")).lower()
    for k, v in state.items():
        if k in ("answer",):  # never leak the answer directly
            continue
        if k.lower() in q or (isinstance(v, str) and v.lower() in q):
            return f"hit: {k} = {v}", 0.05
    return "no results", -0.02


def _tool_calculate(args: dict, state: dict) -> tuple[str, float]:
    expr = str(args.get("expr", ""))
    try:
        # Very restricted "eval" — only digits, operators, dot, and spaces.
        allowed = set("0123456789+-*/(). ")
        if not expr or set(expr) - allowed:
            return "invalid expression", -0.05
        value = eval(expr, {"__builtins__": {}}, {})  # noqa: S307 (guarded)
        return f"= {value}", 0.05
    except Exception as e:
        return f"calc error: {e.__class__.__name__}", -0.05


def _tool_lookup(args: dict, state: dict) -> tuple[str, float]:
    key = str(args.get("key", ""))
    if key in state and key != "answer":
        return f"{key} = {state[key]}", 0.05
    return f"no such key: {key}", -0.02


def _tool_finish(args: dict, state: dict) -> tuple[str, float]:
    answer = args.get("answer")
    truth = state.get("answer")
    try:
        # Compare as strings after light normalization.
        got = str(answer).strip().lower()
        want = str(truth).strip().lower()
        # Numeric answers: allow small float slop.
        try:
            if abs(float(got) - float(want)) < 1e-6:
                return "correct", 1.0
        except ValueError:
            pass
        if got == want:
            return "correct", 1.0
    except Exception:
        pass
    return "wrong", 0.0


def _default_tools() -> dict[str, ToolSpec]:
    return {
        "search":    ToolSpec("search", "Search hidden state by keyword", _tool_search),
        "calculate": ToolSpec("calculate", "Evaluate an arithmetic expression", _tool_calculate),
        "lookup":    ToolSpec("lookup", "Fetch a value by exact key", _tool_lookup),
        "finish":    ToolSpec("finish", "Submit final answer; terminates episode", _tool_finish),
    }


# --------------------------------------------------------------------------- #
# Task generators
# --------------------------------------------------------------------------- #

def sample_tasks(n: int = 64, seed: int = 0) -> list[Task]:
    """Deterministic mix of the four task types."""
    rng = random.Random(seed)
    tasks: list[Task] = []
    kinds = ["sum", "lookup", "twohop", "distract"]
    for i in range(n):
        kind = kinds[i % len(kinds)]
        tasks.append(_make_task(f"{kind}-{i:03d}", kind, rng))
    return tasks


def _make_task(task_id: str, kind: str, rng: random.Random) -> Task:
    if kind == "sum":
        a, b = rng.randint(1, 50), rng.randint(1, 50)
        return Task(task_id, f"What is {a} + {b}?", {"A": a, "B": b, "answer": a + b})
    if kind == "lookup":
        cities = {"eiffel_tower": "paris", "colosseum": "rome", "big_ben": "london"}
        key, city = rng.choice(list(cities.items()))
        # Store the landmark → city directly under its own key so lookup(key=eiffel_tower)
        # actually finds something.
        return Task(task_id, f"Which city hosts the {key.replace('_', ' ')}?",
                    {key: city, "answer": city})
    if kind == "twohop":
        base, delta = rng.randint(10, 40), rng.randint(1, 9)
        return Task(task_id, f"Look up 'base' and add 'delta'. What's the result?",
                    {"base": base, "delta": delta, "answer": base + delta})
    if kind == "distract":
        a, b = rng.randint(1, 30), rng.randint(1, 30)
        red_a, red_b = rng.randint(100, 200), rng.randint(100, 200)
        return Task(task_id, f"What is {a} + {b}?",
                    {"A": a, "B": b, "red_A": red_a, "red_B": red_b, "answer": a + b})
    raise ValueError(f"unknown task kind: {kind}")
