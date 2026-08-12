"""Task synthesis: stop writing benchmark tasks, search for them.

A hand-written GUI benchmark has three problems that no amount of care fixes.
The tasks don't scale — someone authors each one. The gold solutions rot, and a
rotted gold silently converts a benchmark into a measurement of its own
harness. And `optimal_steps`, which efficiency is scored against, is a human
guess about a number the environment actually knows.

All three come from the same root: the task is authored *outside* the
environment, so nothing keeps the two in agreement.

This module inverts that. Given an environment, it breadth-first searches the
reachable state space and reads tasks *out* of it:

    initial state ──BFS over abstract actions──▶ every reachable state
                                                        │
                          for each reachable state, the shortest path to it
                                                        │
        ┌───────────────────────────────────────────────┼──────────────────┐
        ▼                        ▼                      ▼                  ▼
    instruction            verifier                   gold            optimal_steps
  (from the delta)   (the goal state itself)    (the BFS path)     (its length — minimal)

What that buys, all of it structural rather than by discipline:

  **Gold can never rot.** It is derived from the environment, not written
  beside it. Change the layout and the next synthesis run produces correct
  solutions; there is no second artifact to fall out of sync.

  **`optimal_steps` is provably minimal.** BFS finds the shortest action
  sequence, so efficiency is scored against the true optimum rather than an
  estimate. A hand-written benchmark cannot make this claim.

  **Difficulty is measured, not labeled.** Search depth is ground truth for how
  hard a task is.

  **Tasks become unlimited, and holdable-out.** A benchmark that is a generator
  over a seed lets you hold out whole *environments*, not just tasks — the
  thing you actually need to detect memorization.

The same principle as the rest of the package, applied one level up: the
benchmark is verified by construction instead of by assertion.
"""

from __future__ import annotations

import random
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from computer_use.environments import MockComputer, Widget
from computer_use.rewards import StateVerifier
from computer_use.tasks import GUITask
from computer_use.types import Action, ActionKind

#: State keys that are bookkeeping rather than content. `steps` in particular
#: would make every state unique and defeat deduplication entirely.
_BOOKKEEPING = frozenset({"steps"})

#: Additionally excluded when deciding whether two states are the *same goal*.
#: Where the caret sits and how far the page is scrolled are means, not ends —
#: a task should be "the email is set", never "the email is set and you happen
#: to be scrolled 180px down".
_NON_GOAL = _BOOKKEEPING | {"focus", "scroll_y", "screen"}


@dataclass(frozen=True, slots=True)
class AbstractAction:
    """An action named by intent ("click SAVE") rather than by pixel.

    Search has to happen in a finite space, and the concrete action space is
    not finite — `left_click` alone has a million distinct coordinates. Naming
    the target instead makes the branching factor the number of visible
    widgets, and grounding back to coordinates is a lookup at emit time.
    """

    kind: str  # "click" | "type" | "scroll" | "key"
    target: str | None = None
    value: str | None = None
    direction: str | None = None

    def describe(self) -> str:
        if self.kind == "click":
            return f"click {self.target}"
        if self.kind == "type":
            return f"type {self.value!r}"
        if self.kind == "scroll":
            return f"scroll {self.direction}"
        return f"key {self.value}"


@dataclass
class SynthesisConfig:
    """Bounds and vocabulary for the search."""

    #: Text the agent may type. Typing is otherwise an infinite action space,
    #: so the caller supplies the values that matter for their domain.
    #:
    #: A flat sequence offers every token to every field, which generates
    #: type-inappropriate goals ("set EMAIL to 5"). A mapping from field id to
    #: tokens keeps values with the fields they belong to and is preferred for
    #: anything you intend to publish as a benchmark.
    vocabulary: Sequence[str] | Mapping[str, Sequence[str]] = ()
    max_depth: int = 6
    #: Hard cap on explored states, so a pathological environment cannot hang
    #: a synthesis run.
    max_states: int = 20_000
    scroll_amount: int = 3
    #: Whether the search may use a global submit key. Off by default: the mock
    #: treats Return as "press the primary button" from anywhere, which
    #: short-circuits scroll-then-click and yields shorter tasks that exercise
    #: less. It is a real affordance, so it is available — just not the default,
    #: because a benchmark should make the agent do the work.
    allow_submit_key: bool = False


@dataclass(frozen=True, slots=True)
class Discovery:
    """One reachable goal state and the cheapest way to reach it."""

    goal: Mapping[str, Any]
    delta: Mapping[str, Any]
    path: tuple[AbstractAction, ...]
    depth: int


# --------------------------------------------------------------------------- #
# Search
# --------------------------------------------------------------------------- #


def available_actions(
    env: MockComputer, config: SynthesisConfig
) -> list[AbstractAction]:
    """Every abstract action that could do something from the current state.

    Two prunings keep the space finite and the search honest:

    Only *visible* widgets are clickable, which is what makes the search
    discover that SAVE must be scrolled to before it can be clicked — the
    dependency is found, not encoded.

    Typing is offered only into an empty focused field. Fields append, so
    without this the state space grows without bound (`X`, `XX`, `XXX`, …) and
    BFS never terminates.
    """
    actions: list[AbstractAction] = []
    state = env.state()

    for widget in env._visible:
        if widget.kind == "label":
            continue
        y = widget.y - env._scroll_y + widget.height // 2
        if 0 <= y < env.height:
            actions.append(AbstractAction("click", target=widget.id))

    focused = env._focused_field()
    if focused is not None and not focused.value:
        actions.extend(
            AbstractAction("type", value=token)
            for token in _tokens_for(config.vocabulary, focused.id)
        )

    limit = max(0, env._content_height - env.height)
    if limit > 0:
        if state["scroll_y"] < limit:
            actions.append(AbstractAction("scroll", direction="down"))
        if state["scroll_y"] > 0:
            actions.append(AbstractAction("scroll", direction="up"))

    if config.allow_submit_key:
        actions.append(AbstractAction("key", value="Return"))

    return actions


def ground(env: MockComputer, abstract: AbstractAction, config: SynthesisConfig) -> Action:
    """Turn an abstract action into a concrete one, in the current viewport."""
    if abstract.kind == "click":
        widget = next(w for w in env._widgets if w.id == abstract.target)
        return Action(
            ActionKind.LEFT_CLICK,
            coordinate=(
                widget.x + widget.width // 2,
                widget.y + widget.height // 2 - env._scroll_y,
            ),
        )
    if abstract.kind == "type":
        return Action(ActionKind.TYPE, text=abstract.value)
    if abstract.kind == "scroll":
        return Action(
            ActionKind.SCROLL,
            coordinate=(env.width // 2, env.height // 2),
            scroll_direction=abstract.direction,
            scroll_amount=config.scroll_amount,
        )
    return Action(ActionKind.KEY, text=abstract.value)


def _apply(env: MockComputer, abstract: AbstractAction, config: SynthesisConfig) -> None:
    """Execute an abstract action synchronously.

    The environment's handlers are pure state transitions; only `execute` is
    async (it renders a frame afterwards). Search runs millions of transitions
    and needs none of those pixels, so it dispatches to the handler directly —
    which is also what makes exhaustive search tractable at all.
    """
    action = ground(env, abstract, config)
    handler = getattr(env, f"_do_{action.kind.value}")
    env.action_log.append(action.describe())
    handler(action)


def _tokens_for(
    vocabulary: Sequence[str] | Mapping[str, Sequence[str]], field_id: str
) -> Sequence[str]:
    if isinstance(vocabulary, Mapping):
        return vocabulary.get(field_id, ())
    return vocabulary


def _key(state: Mapping[str, Any], exclude: frozenset[str]) -> tuple[tuple[str, Any], ...]:
    return tuple(sorted((k, v) for k, v in state.items() if k not in exclude))


def _redundant_keys(env: MockComputer) -> set[str]:
    """State keys whose value is implied by another key already in the state.

    Selecting a radio moves three keys at once — the group's selection plus
    both members' booleans — but that is one fact, not three. Left alone it
    produces goals that dedupe as distinct and instructions that say the same
    thing three times ("choose EXPRESS, turn on EXPRESS, turn off STANDARD").

    The group key survives because it is the one that names the choice.
    """
    return {
        widget.id
        for widget in env._widgets
        if widget.kind == "radio" and widget.group
    }


def _project(env: MockComputer, state: Mapping[str, Any]) -> dict[str, Any]:
    """The part of a state that a task can meaningfully be *about*."""
    redundant = _redundant_keys(env)
    groups = {w.group for w in env._widgets if w.kind == "radio" and w.group}
    return {
        k: v
        for k, v in state.items()
        if k not in _NON_GOAL and not (k in redundant and groups & set(state))
    }


def explore(
    env: MockComputer, config: SynthesisConfig | None = None
) -> list[Discovery]:
    """Breadth-first search the reachable state space.

    BFS rather than DFS for the property that matters: the first path to a
    state is the shortest one, so every solution this returns is optimal by
    construction. That is the whole basis for `optimal_steps` being a fact
    rather than a guess.
    """
    cfg = config or SynthesisConfig()
    start = env.snapshot()
    initial_state = dict(env.state())
    initial_projection = _project(env, initial_state)
    initial_goal = _key(initial_projection, frozenset())

    # Search identity keeps focus/scroll/screen — they change what is possible.
    # Goal identity drops them — they are not what a task is about.
    seen_search = {_key(initial_state, _BOOKKEEPING)}
    best_goal: dict[tuple[tuple[str, Any], ...], Discovery] = {}

    queue: deque[tuple[dict[str, Any], tuple[AbstractAction, ...]]] = deque(
        [(start, ())]
    )

    while queue and len(seen_search) < cfg.max_states:
        snap, path = queue.popleft()
        if len(path) >= cfg.max_depth:
            continue

        env.restore(snap)
        for abstract in available_actions(env, cfg):
            env.restore(snap)
            _apply(env, abstract, cfg)

            state = dict(env.state())
            search_key = _key(state, _BOOKKEEPING)
            if search_key in seen_search:
                continue
            seen_search.add(search_key)

            next_path = (*path, abstract)
            projection = _project(env, state)
            goal_key = _key(projection, frozenset())
            if goal_key != initial_goal and goal_key not in best_goal:
                best_goal[goal_key] = Discovery(
                    goal=projection,
                    delta=_delta(initial_projection, projection),
                    path=next_path,
                    depth=len(next_path),
                )
            queue.append((env.snapshot(), next_path))

    env.restore(start)
    return sorted(best_goal.values(), key=lambda d: (d.depth, str(d.delta)))


def _delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in after.items() if before.get(k) != v}


# --------------------------------------------------------------------------- #
# Turning discoveries into tasks
# --------------------------------------------------------------------------- #


def _widgets_by_id(env: MockComputer) -> dict[str, Widget]:
    return {w.id: w for w in env._widgets}


def _label_for(env: MockComputer, key: str) -> str:
    """A human name for a state key, read off the UI the agent will see.

    Prefers the adjacent label widget a form field would have, because that is
    the text actually rendered next to it — the instruction should name what is
    on screen, not an internal identifier.
    """
    widgets = _widgets_by_id(env)
    sibling = widgets.get(f"{key}_label")
    if sibling is not None and sibling.label:
        return sibling.label
    widget = widgets.get(key)
    if widget is not None and widget.label:
        return widget.label
    setter = next((w for w in env._widgets if w.sets == key), None)
    if setter is not None and setter.label:
        return setter.label
    return key.replace("_", " ").upper()


def _phrase(env: MockComputer, key: str, value: Any) -> str:
    widgets = _widgets_by_id(env)
    widget = widgets.get(key)

    if isinstance(value, str) and value in widgets:  # a radio group's selection
        return f"choose {widgets[value].label or value}"
    if widget is not None and widget.kind == "field":
        return f'set {_label_for(env, key)} to "{value}"'
    if widget is not None and widget.kind in ("checkbox", "radio"):
        return f"turn {'on' if value else 'off'} {_label_for(env, key)}"
    if value is True:  # a flag set by a button — name the control, not the verb
        return f"press {_label_for(env, key)}"
    return f"set {_label_for(env, key)} to {value}"


def _join(parts: Sequence[str]) -> str:
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return ", ".join(parts[:-1]) + f", and {parts[-1]}"


def describe(env: MockComputer, delta: Mapping[str, Any]) -> str:
    """Render a goal delta as an instruction.

    Template-based on purpose. The instruction has to mean *exactly* the
    verifier, and a generated sentence that drifts from the goal state would
    reintroduce the gap this module exists to close.
    """
    ordered = sorted(delta.items(), key=lambda kv: (isinstance(kv[1], bool), kv[0]))
    sentence = _join([_phrase(env, k, v) for k, v in ordered])
    return sentence[0].upper() + sentence[1:] + "."


def _difficulty(discovery: Discovery, path_kinds: set[str]) -> str:
    if discovery.depth <= 2:
        return "easy"
    if discovery.depth <= 4 and "navigate" not in path_kinds:
        return "medium"
    return "hard"


def synthesize(
    env_factory: Callable[[], MockComputer],
    config: SynthesisConfig | None = None,
    *,
    name_prefix: str = "synth",
    limit: int | None = None,
    min_depth: int = 1,
    seed: int | None = None,
) -> list[GUITask]:
    """Generate benchmark tasks by searching an environment.

    Every returned task is solvable — its `gold` is the path the search took —
    and optimally so. Nothing here is asserted; it all falls out of the search.

    `limit` samples from the discovered set rather than truncating it, so a
    capped run still spans the difficulty range instead of returning only the
    shallowest tasks. `seed` makes that sampling reproducible.
    """
    cfg = config or SynthesisConfig()
    probe = env_factory()
    discoveries = [d for d in explore(probe, cfg) if d.depth >= min_depth]

    if limit is not None and len(discoveries) > limit:
        rng = random.Random(seed)
        by_depth: dict[int, list[Discovery]] = {}
        for discovery in discoveries:
            by_depth.setdefault(discovery.depth, []).append(discovery)
        # Round-robin across depths so the sample keeps the difficulty spread.
        picked: list[Discovery] = []
        pools = [rng.sample(v, len(v)) for _, v in sorted(by_depth.items())]
        while len(picked) < limit and any(pools):
            for pool in pools:
                if pool and len(picked) < limit:
                    picked.append(pool.pop())
        discoveries = sorted(picked, key=lambda d: (d.depth, str(d.delta)))

    tasks: list[GUITask] = []
    for index, discovery in enumerate(discoveries):
        env = env_factory()
        instruction = describe(env, discovery.delta)

        # Ground the path by replaying it: a click's pixel depends on the
        # scroll offset at the moment it happens, so coordinates can only be
        # resolved along the way, not up front.
        gold: list[Action] = []
        kinds: set[str] = set()
        for abstract in discovery.path:
            gold.append(ground(env, abstract, cfg))
            before = env.state()["screen"]
            _apply(env, abstract, cfg)
            if env.state()["screen"] != before:
                kinds.add("navigate")
            kinds.add({"click": "click", "type": "typing",
                       "scroll": "scroll", "key": "keyboard"}[abstract.kind])

        tasks.append(GUITask(
            name=f"{name_prefix}.{index:02d}",
            instruction=instruction,
            env_factory=env_factory,
            verifier=StateVerifier(dict(discovery.delta)),
            optimal_steps=discovery.depth,
            difficulty=_difficulty(discovery, kinds),
            gold=tuple(gold),
            tags=(*sorted(kinds), "synthetic"),
            metadata={
                "synthesized": True,
                "goal": dict(discovery.goal),
                "delta": dict(discovery.delta),
                "abstract_path": [a.describe() for a in discovery.path],
            },
        ))

    return tasks


# --------------------------------------------------------------------------- #
# Ready-made generators for the bundled environments
# --------------------------------------------------------------------------- #

#: Per-field vocabularies. Keyed by field id so an email address is only ever
#: offered to the email field — a flat list generates "set MAX RETRIES to
#: ada@example.com", which is reachable, verifiable, and nonsense.
VOCABULARIES: dict[str, dict[str, tuple[str, ...]]] = {
    "settings": {"email": ("ada@example.com",), "retries": ("5",)},
    "checkout": {"promo": ("SPRING20",)},
    "files": {"newname": ("q4-final.pdf",)},
}


def synthesize_suite(
    *,
    per_environment: int | None = 6,
    max_depth: int = 5,
    seed: int | None = 0,
) -> list[GUITask]:
    """Synthesize a benchmark across all three bundled environments.

    The point of spanning environments rather than one: a suite generated from
    a single app measures how well an agent knows *that* app. Held-out
    environments are what measure whether it can operate a GUI at all.
    """
    generators = [
        ("settings", MockComputer.settings_form, VOCABULARIES["settings"]),
        ("checkout", MockComputer.checkout_flow, VOCABULARIES["checkout"]),
        ("files", MockComputer.file_manager, VOCABULARIES["files"]),
    ]
    tasks: list[GUITask] = []
    for prefix, factory, vocabulary in generators:
        tasks.extend(synthesize(
            factory,
            SynthesisConfig(vocabulary=vocabulary, max_depth=max_depth),
            name_prefix=f"synth.{prefix}",
            limit=per_environment,
            min_depth=2,
            seed=seed,
        ))
    return tasks


__all__ = [
    "VOCABULARIES",
    "AbstractAction",
    "Discovery",
    "SynthesisConfig",
    "available_actions",
    "describe",
    "explore",
    "ground",
    "synthesize",
    "synthesize_suite",
]
