"""Reference policies for the ToolEnv.

Not neural — just deterministic strategies used to seed rollouts. Real
training swaps in an LLM policy; these exist so trajectories are
concrete without a GPU, and so tests can assert exact behavior.

Three ability tiers so `TrajectoryAgent` can simulate a policy improving:
  - `random_policy`      — picks a random tool. Baseline.
  - `heuristic_policy`   — half the time uses the right tool, half random.
  - `expert_policy`      — always uses the right tool. Perfect play.
"""

from __future__ import annotations

import random
import re
from typing import Callable


Policy = Callable[..., tuple[str, dict]]
# Contract: policy(goal: str, history: list[dict]) -> (tool_name, args).
# `goal` is the initial task observation (stable, like a system prompt).
# `history` is the list of past (tool, args, obs, reward) dicts.


def _extract_ints(text: str) -> list[int]:
    return [int(m.group()) for m in re.finditer(r"-?\d+", text)]


def _parse_goal(obs: str) -> dict[str, str | list[int]]:
    """Very light goal parsing to help the heuristic + expert policies.

    We strip the `[TASK …] ` prefix before extracting ints so the task_id's
    digits don't leak into the goal (e.g. `sum-005` would inject `5` as
    the first integer and completely change the arithmetic).
    """
    body = re.sub(r"^\[TASK [^\]]+\]\s*", "", obs)
    lower = body.lower()
    ints = _extract_ints(body)
    kind = "unknown"
    if "+" in body and len(ints) >= 2:
        kind = "sum"
    elif "which city" in lower:
        kind = "lookup"
    elif "base" in lower and "delta" in lower:
        kind = "twohop"
    return {"kind": kind, "ints": ints, "obs": lower}


def random_policy(goal: str, history: list[dict], rng: random.Random | None = None) -> tuple[str, dict]:
    rng = rng or random.Random(0)
    tool = rng.choice(["search", "calculate", "lookup", "finish"])
    if tool == "finish":
        return tool, {"answer": str(rng.randint(0, 100))}
    if tool == "calculate":
        return tool, {"expr": f"{rng.randint(1, 9)}+{rng.randint(1, 9)}"}
    if tool == "search":
        return tool, {"query": "anything"}
    return tool, {"key": "unknown"}


def heuristic_policy(goal: str, history: list[dict], rng: random.Random | None = None) -> tuple[str, dict]:
    rng = rng or random.Random(0)
    if rng.random() < 0.5:
        return random_policy(goal, history, rng)
    return expert_policy(goal, history, rng)


def expert_policy(goal: str, history: list[dict], rng: random.Random | None = None) -> tuple[str, dict]:
    """Perfect play: parse the goal, use the right tools in the right order."""
    parsed = _parse_goal(goal)
    kind = parsed["kind"]
    ints = parsed["ints"]

    # Look at what the history already learned.
    seen_base = None
    seen_delta = None
    for h in history:
        for line in [h.get("obs", "")]:
            m = re.search(r"base = (-?\d+)", line)
            if m:
                seen_base = int(m.group(1))
            m = re.search(r"delta = (-?\d+)", line)
            if m:
                seen_delta = int(m.group(1))

    if kind == "sum" and len(ints) >= 2:
        a, b = ints[0], ints[1]
        return "finish", {"answer": str(a + b)}
    if kind == "lookup":
        for name in ("eiffel_tower", "colosseum", "big_ben"):
            if name.replace("_", " ") in parsed["obs"]:
                if not any(h.get("tool") == "lookup" and h.get("args", {}).get("key") == name for h in history):
                    return "lookup", {"key": name}
                for h in history:
                    if h.get("tool") == "lookup" and h.get("args", {}).get("key") == name:
                        m = re.search(rf"{name} = (\w+)", h.get("obs", ""))
                        if m:
                            return "finish", {"answer": m.group(1)}
    if kind == "twohop":
        if seen_base is None:
            return "lookup", {"key": "base"}
        if seen_delta is None:
            return "lookup", {"key": "delta"}
        return "finish", {"answer": str(seen_base + seen_delta)}

    # Fallback: try a search then finish with an arbitrary answer.
    if not any(h.get("tool") == "search" for h in history):
        return "search", {"query": "answer"}
    return "finish", {"answer": "unknown"}


POLICIES: dict[str, Policy] = {
    "random":    random_policy,
    "heuristic": heuristic_policy,
    "expert":    expert_policy,
}
