"""A policy that learns to operate a GUI from this pipeline's own output.

Everything upstream generates: applications, tasks, gold paths, rollouts,
preference pairs. Nothing consumed any of it. So the framework's central claim
— that verified GUI episodes are useful post-training data — was argued rather
than demonstrated, and a bug that quietly destroyed the data's value would have
shown up as nothing at all.

This module closes the loop offline. It trains a grounding model on
trajectories the pipeline produced, and the resulting policy plugs into
`run_episode` like any other, so it is scored by the same verifiers on the same
benchmark.

What is learned, and what is not — worth being exact, because a demo that
hardcodes the answer and calls it learning is worse than no demo:

  **Not learned.** Reading the instruction into an ordered list of goals
  ("set PHONE to 5550142", then "press APPLY"). That is language, the
  instructions are generated from templates, and pretending to learn it would
  prove nothing.

  **Learned.** Which thing on the screen a goal refers to, and where to click
  it. The model scores every element the parser found against the goal using
  generic features — token overlap, control kind, geometry, what has already
  been done — and picks the best. Nothing in the feature set says "click the
  element whose label matches"; if that is the rule, the weights have to find
  it in the data.

  **From pixels.** The policy sees `perception.parse_screen(frame)` and
  nothing else. It never touches `env.state()`, so its score means what a
  benchmark score is supposed to mean.

The learner is a softmax over candidates trained by plain SGD, in pure Python
with no numpy. That is not a limitation to apologize for: the point is to show
the *data* carries signal, and a model small enough to read end to end makes it
obvious the signal is not coming from the model.
"""

from __future__ import annotations

import math
import random
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from computer_use.perception import Element, Screen, parse_screen
from computer_use.policies import Decision
from computer_use.types import Action, ActionKind, Screenshot, Trajectory

# --------------------------------------------------------------------------- #
# Reading the instruction
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Goal:
    """One thing the instruction asks for."""

    kind: str  # "set" | "toggle" | "choose" | "press"
    phrase: str
    value: str = ""

    @property
    def tokens(self) -> tuple[str, ...]:
        return tuple(t for t in re.split(r"[^A-Z0-9@.\-]+", self.phrase.upper()) if t)


#: `\b` is load-bearing: without it "press" matches inside "EXPRESS - 1 DAY",
#: inventing a goal to press a button called "- 1 DAY".
_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("set", re.compile(r'\bset\s+(?P<phrase>.+?)\s+to\s+"(?P<value>[^"]*)"', re.I)),
    ("toggle", re.compile(r"\bturn\s+on\s+(?P<phrase>[^.,]+?)(?=\s+and\s|\.|,|$)", re.I)),
    ("choose", re.compile(r"\bchoose\s+(?P<phrase>[^.,]+?)(?=\s+and\s|\.|,|$)", re.I)),
    ("press", re.compile(r"\bpress\s+(?P<phrase>[^.,]+?)(?=\s+and\s|\.|,|$)", re.I)),
)


def read_instruction(text: str) -> tuple[Goal, ...]:
    """Split an instruction into ordered goals.

    Deliberately dumb pattern matching, and deliberately declared as given
    rather than learned. Order matters: the generated instructions describe
    steps in the order a person would do them, and the final press is last.
    """
    found: list[tuple[int, Goal]] = []
    for kind, pattern in _PATTERNS:
        for match in pattern.finditer(text):
            phrase = match.group("phrase").strip().strip(".")
            if not phrase:
                continue
            value = match.groupdict().get("value") or ""
            found.append((match.start(), Goal(kind=kind, phrase=phrase, value=value)))
    found.sort(key=lambda pair: pair[0])
    return tuple(goal for _, goal in found)


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #


def _overlap(goal_tokens: Sequence[str], element_tokens: Sequence[str]) -> float:
    if not goal_tokens or not element_tokens:
        return 0.0
    shared = len(set(goal_tokens) & set(element_tokens))
    return shared / len(set(goal_tokens) | set(element_tokens))


def features(goal: Goal, element: Element, screen: Screen) -> dict[str, float]:
    """Describe one (goal, element) pair.

    Generic on purpose. There is no feature that means "this is the right
    one" — the closest is token overlap, which is equally high for the caption
    *next to* the field and for the field itself, so the model still has to
    learn that a `set` goal wants the box and not the words.
    """
    goal_tokens = goal.tokens
    element_tokens = element.tokens
    overlap = _overlap(goal_tokens, element_tokens)
    exact = 1.0 if element.label.upper() == goal.phrase.upper() else 0.0
    contains = 1.0 if goal_tokens and set(goal_tokens) <= set(element_tokens) else 0.0

    out = {
        "bias": 1.0,
        "overlap": overlap,
        "exact": exact,
        "contains": contains,
        "no_overlap": 1.0 if overlap == 0.0 else 0.0,
        f"kind:{element.kind}": 1.0,
        f"goal:{goal.kind}": 1.0,
        f"goal:{goal.kind}|kind:{element.kind}": 1.0,
        f"goal:{goal.kind}|overlap": overlap,
        f"goal:{goal.kind}|exact": exact,
        f"goal:{goal.kind}|x": element.click[0] / max(screen.width, 1),
        "has_box": 1.0 if element.box is not None else 0.0,
        "filled": 1.0 if element.filled else 0.0,
        f"goal:{goal.kind}|filled": 1.0 if element.filled else 0.0,
        "y": element.click[1] / max(screen.height, 1),
        "x": element.click[0] / max(screen.width, 1),
    }
    if goal.kind == "navigate":
        # Vocabulary, but only for the one decision that needs it: which way
        # off this screen. "BACK" and "CONTINUE" are equally unrelated to what
        # the task asked for, so overlap cannot separate them and geometry only
        # weakly can — what separates them is what the words mean, which the
        # model can learn from training apps and carry to unseen ones. Confined
        # to navigation so grounding cannot quietly memorize labels instead.
        for token in element_tokens:
            out[f"word:{token}"] = 1.0
    if element.box is not None:
        out["width"] = element.box.width / max(screen.width, 1)
        out["height"] = element.box.height / max(screen.height, 1)
    # A caption sitting directly above a field is how a form names its input,
    # so whether the goal's words appear just above this element is a real
    # signal — and one the model has to weigh for itself.
    above = _label_above(element, screen)
    out["above_overlap"] = _overlap(goal_tokens, above)
    out[f"goal:{goal.kind}|above"] = out["above_overlap"]
    return out


def _label_above(element: Element, screen: Screen) -> tuple[str, ...]:
    if element.box is None:
        return ()
    best: Element | None = None
    for other in screen.elements:
        if other is element or other.kind != "text" or other.text_run is None:
            continue
        gap = element.box.y - (other.text_run.y + other.text_run.height)
        if 0 <= gap <= 24 and abs(other.text_run.x - element.box.x) <= 40:
            if best is None or other.text_run.y > (best.text_run.y if best.text_run else -1):
                best = other
    return best.tokens if best is not None else ()


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #


@dataclass
class Grounder:
    """A linear scorer over (goal, element) features, with a softmax head."""

    weights: dict[str, float] = field(default_factory=dict)

    def score(self, feats: dict[str, float]) -> float:
        return sum(self.weights.get(name, 0.0) * value for name, value in feats.items())

    def rank(self, goal: Goal, screen: Screen) -> list[tuple[float, Element]]:
        scored = [
            (self.score(features(goal, element, screen)), element)
            for element in screen.elements
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return scored

    def choose(self, goal: Goal, screen: Screen) -> Element | None:
        ranked = self.rank(goal, screen)
        return ranked[0][1] if ranked else None

    def to_dict(self) -> dict[str, float]:
        return dict(self.weights)

    @classmethod
    def from_dict(cls, weights: dict[str, float]) -> Grounder:
        return cls(weights=dict(weights))


@dataclass(frozen=True, slots=True)
class Example:
    """One decision: the candidates that were on screen, and the right one."""

    candidates: tuple[dict[str, float], ...]
    correct: int

    @property
    def usable(self) -> bool:
        return len(self.candidates) > 1 and 0 <= self.correct < len(self.candidates)


def train(
    examples: Sequence[Example],
    *,
    epochs: int = 30,
    learning_rate: float = 0.5,
    l2: float = 1e-4,
    seed: int = 0,
) -> Grounder:
    """Fit the scorer by maximizing the log-likelihood of the chosen element.

    Softmax over the candidates that were actually on the screen, which is the
    decision the policy faces — training against a fixed action vocabulary
    would let it succeed by memorizing an action rather than by finding the
    thing on screen.
    """
    model = Grounder()
    rng = random.Random(seed)
    order = [i for i, e in enumerate(examples) if e.usable]
    for _ in range(epochs):
        rng.shuffle(order)
        for index in order:
            example = examples[index]
            scores = [model.score(c) for c in example.candidates]
            highest = max(scores)
            weights = [math.exp(s - highest) for s in scores]
            total = sum(weights) or 1.0
            probabilities = [w / total for w in weights]
            for i, candidate in enumerate(example.candidates):
                error = (1.0 if i == example.correct else 0.0) - probabilities[i]
                if error == 0.0:
                    continue
                for name, value in candidate.items():
                    current = model.weights.get(name, 0.0)
                    model.weights[name] = current + learning_rate * (
                        error * value - l2 * current
                    )
    return model


def accuracy(model: Grounder, examples: Sequence[Example]) -> float:
    """Fraction of decisions where the model's top choice is the right one."""
    usable = [e for e in examples if e.usable]
    if not usable:
        return 0.0
    correct = 0
    for example in usable:
        scores = [model.score(c) for c in example.candidates]
        correct += scores.index(max(scores)) == example.correct
    return correct / len(usable)


# --------------------------------------------------------------------------- #
# Turning trajectories into examples
# --------------------------------------------------------------------------- #


def _target_index(screen: Screen, point: tuple[int, int]) -> int | None:
    """Which parsed element a click landed on."""
    best: int | None = None
    best_area = None
    for i, element in enumerate(screen.elements):
        if element.box is None or not element.box.contains(*point):
            continue
        area = element.box.width * element.box.height
        if best_area is None or area < best_area:
            best, best_area = i, area
    return best


def examples_from(trajectories: Iterable[Trajectory], *, successful_only: bool = True) -> list[Example]:
    """Build training decisions out of episodes the pipeline recorded.

    Only clicks, and only from runs the verifier passed: a step from a failed
    episode is not known to be a mistake, and treating it as one is exactly the
    mis-attribution that `dataset._blame_step` exists to avoid.
    """
    built: list[Example] = []
    for trajectory in trajectories:
        if successful_only and not trajectory.succeeded:
            continue
        goals = read_instruction(trajectory.task)
        if not goals:
            continue
        cursor = 0
        for step in trajectory.steps:
            if step.observation is None or step.action.kind is not ActionKind.LEFT_CLICK:
                continue
            if step.action.coordinate is None or cursor >= len(goals):
                continue
            screen = parse_screen(step.observation.data)
            target = _target_index(screen, step.action.coordinate)
            if target is None:
                continue
            goal = goals[cursor]
            chosen = screen.elements[target]
            evidence = features(goal, chosen, screen)
            satisfied = evidence["overlap"] > 0.4 or evidence["above_overlap"] > 0.4

            # A click on something the goal does not name is the agent going to
            # look for it. That is a different decision from grounding, and
            # training it as if it were grounding teaches the model that a goal
            # can be satisfied by a button with nothing to do with it.
            labelled = goal if satisfied else Goal("navigate", goal.phrase, goal.value)
            built.append(Example(
                candidates=tuple(features(labelled, e, screen) for e in screen.elements),
                correct=target,
            ))
            if satisfied:
                cursor += 1
    return built


# --------------------------------------------------------------------------- #
# The policy
# --------------------------------------------------------------------------- #


@dataclass
class LearnedPolicy:
    """Acts on a GUI using only the pixels and a learned grounding model.

    The loop is the obvious one: take the next unfinished goal, score what is
    on screen, click the best candidate, type when the goal carries a value,
    and scroll when the thing being looked for is not here. What makes the
    score meaningful is what is absent — no environment state, no widget ids,
    no coordinates from the task definition.
    """

    model: Grounder
    width: int = 1280
    height: int = 720
    max_scrolls: int = 2
    max_explore: int = 4
    confidence: float = float("-inf")

    _goals: tuple[Goal, ...] = field(default_factory=tuple, init=False)
    _index: int = field(default=0, init=False)
    _pending_value: str = field(default="", init=False)
    _scrolls: int = field(default=0, init=False)
    _explored: int = field(default=0, init=False)
    _visited: set[str] = field(default_factory=set, init=False)

    def reset(self) -> None:
        self._goals = ()
        self._index = 0
        self._pending_value = ""
        self._scrolls = 0
        self._explored = 0
        self._visited = set()

    async def begin(self, task: str, screenshot: Screenshot) -> Decision:
        self.reset()
        self._goals = read_instruction(task)
        return self._decide(screenshot)

    async def observe(self, screenshot: Screenshot, *, error: str | None = None) -> Decision:
        return self._decide(screenshot)

    # -- internals ---------------------------------------------------------- #

    def _decide(self, screenshot: Screenshot) -> Decision:
        if self._pending_value:
            value, self._pending_value = self._pending_value, ""
            return Decision(
                Action(ActionKind.TYPE, text=value),
                f"type {value!r}", "learned", "tool_use",
            )
        if self._index >= len(self._goals):
            return Decision(None, "all goals satisfied", None, "end_turn")

        goal = self._goals[self._index]
        screen = parse_screen(screenshot.data)
        ranked = self.model.rank(goal, screen)
        if not ranked:
            return Decision(None, "nothing on screen", None, "end_turn")

        score, element = ranked[0]
        # A goal names its target by what the screen says about it — which for
        # a button is the word on it, and for an input is the caption above it.
        # If neither shares anything with the goal, the target is not on this
        # screen, and acting on the best of a bad set is how an agent
        # confidently clicks the wrong control instead of going to look.
        evidence = features(goal, element, screen)
        visible = element.box is not None and (
            evidence["overlap"] > 0.0 or evidence["above_overlap"] > 0.0
        )
        if not visible or score < self.confidence:
            return self._look_further(screen, goal)

        self._index += 1
        self._scrolls = 0
        if goal.kind == "set":
            self._pending_value = goal.value
        return Decision(
            Action(ActionKind.LEFT_CLICK, coordinate=element.click),
            f"{goal.kind} {goal.phrase!r} -> {element.label!r} ({score:+.2f})",
            "learned", "tool_use",
        )

    def _look_further(self, screen: Screen, goal: Goal) -> Decision:
        """Go looking: scroll this screen, then try a way off it."""
        if (
            screen.scroll is not None
            and screen.scroll[0] < screen.scroll[1]
            and self._scrolls < self.max_scrolls
        ):
            self._scrolls += 1
            return Decision(
                Action(
                    ActionKind.SCROLL, coordinate=(self.width // 2, self.height // 2),
                    scroll_direction="down", scroll_amount=3,
                ),
                f"looking for {goal.phrase!r}", "learned", "tool_use",
            )

        door = self._exit(screen, goal)
        if door is not None and self._explored < self.max_explore:
            self._explored += 1
            self._scrolls = 0
            self._visited.add(_key(screen, door))
            return Decision(
                Action(ActionKind.LEFT_CLICK, coordinate=door.click),
                f"{goal.phrase!r} is not here; through {door.label!r}",
                "learned", "tool_use",
            )
        return Decision(None, f"cannot find {goal.phrase!r}", None, "end_turn")

    def _exit(self, screen: Screen, goal: Goal) -> Element | None:
        """The best way off this screen, scored by the same model.

        Two exclusions carry weight before scoring. A button named by a
        remaining goal is not a corridor, it is the destination — pressing it
        early submits a half-finished form. And a *filled* button is the
        screen's primary action, the commit rather than the way out.

        Which of the survivors to take is a judgement, not a rule: a wizard
        offers BACK and CONTINUE side by side and taking the first one in
        reading order walks backwards forever. So the model ranks them, having
        learned from gold paths which direction makes progress.
        """
        wanted = {token for g in self._goals[self._index:] for token in g.tokens}
        navigate = Goal("navigate", goal.phrase, goal.value)
        doors = [
            element for element in screen.elements
            if element.kind == "button" and element.label and not element.filled
            and not (set(element.tokens) & wanted)
            and _key(screen, element) not in self._visited
        ]
        if not doors:
            return None
        return max(doors, key=lambda e: self.model.score(features(navigate, e, screen)))


def _key(screen: Screen, element: Element) -> str:
    return f"{screen.title}|{element.label}"


@dataclass(frozen=True, slots=True)
class LoopResult:
    """What one closed-loop run measured."""

    train_tasks: int
    test_tasks: int
    decisions: int
    train_accuracy: float
    trained: int
    untrained: tuple[int, ...]
    model: Grounder

    @property
    def trained_rate(self) -> float:
        return self.trained / self.test_tasks if self.test_tasks else 0.0

    @property
    def untrained_rate(self) -> float:
        if not self.untrained or not self.test_tasks:
            return 0.0
        return sum(self.untrained) / len(self.untrained) / self.test_tasks

    def to_dict(self) -> dict[str, object]:
        return {
            "train_tasks": self.train_tasks,
            "test_tasks": self.test_tasks,
            "decisions": self.decisions,
            "train_accuracy": round(self.train_accuracy, 4),
            "held_out_success": round(self.trained_rate, 4),
            "untrained_success": round(self.untrained_rate, 4),
            "untrained_runs": list(self.untrained),
            "weights": {k: round(v, 4) for k, v in self.model.weights.items()},
        }


def closed_loop(
    *,
    train_worlds: Sequence[int],
    test_worlds: Sequence[int],
    per_world: int = 3,
    max_steps: int = 24,
    epochs: int = 30,
    seed: int = 0,
    controls: int = 3,
) -> LoopResult:
    """Generate, collect, train, and score on applications never seen.

    The whole argument in one function: the apps come from a generator, the
    tasks are searched out of them, the training data is rollouts of the
    searched solutions, and the score is on a disjoint set of generated apps.
    No step of it was written by hand, and the policy reads pixels.

    `controls` random-weight runs come back alongside, because "the policy
    solves held-out apps" means nothing without knowing what the feature set
    alone would have scored.
    """
    from computer_use.rollout import RolloutConfig
    from computer_use.worlds import curriculum

    overlap = set(train_worlds) & set(test_worlds)
    if overlap:
        raise ValueError(
            f"train and test worlds overlap on seeds {sorted(overlap)} — the "
            "held-out score would be measuring memorization"
        )

    config = RolloutConfig(store_frames=True, max_steps=max_steps)
    training = curriculum(train_worlds, per_world=per_world)
    held_out = curriculum(test_worlds, per_world=per_world)

    from computer_use.policies import ScriptedPolicy

    demonstrations = [
        _run(task, ScriptedPolicy(list(task.gold)), config) for task in training
    ]
    decisions = examples_from(demonstrations)
    model = train(decisions, epochs=epochs, seed=seed)

    scored = sum(
        _run(task, LearnedPolicy(model=model), config).succeeded for task in held_out
    )
    blind = tuple(
        sum(
            _run(task, LearnedPolicy(model=untrained(model, seed=s)), config).succeeded
            for task in held_out
        )
        for s in range(1, controls + 1)
    )
    return LoopResult(
        train_tasks=len(training), test_tasks=len(held_out), decisions=len(decisions),
        train_accuracy=accuracy(model, decisions), trained=scored, untrained=blind,
        model=model,
    )


def _run(task: object, policy: object, config: object) -> Trajectory:
    import asyncio

    from computer_use.rollout import run_episode

    return asyncio.run(run_episode(
        task.instruction,  # type: ignore[attr-defined]
        task.env_factory(),  # type: ignore[attr-defined]
        policy,  # type: ignore[arg-type]
        verifier=task.verifier,  # type: ignore[attr-defined]
        reward_config=task.reward_config(),  # type: ignore[attr-defined]
        config=config,  # type: ignore[arg-type]
    ))


def policy_factory(model: Grounder, **kwargs: object) -> object:
    """A `run_episode`-compatible factory that shares one model across episodes."""
    def build(env: object) -> LearnedPolicy:
        width = int(getattr(env, "width", 1280))
        height = int(getattr(env, "height", 720))
        return LearnedPolicy(model=model, width=width, height=height, **kwargs)  # type: ignore[arg-type]
    return build


def untrained(reference: Grounder, *, seed: int = 0, scale: float = 1.0) -> Grounder:
    """The same model with random weights — the control for any claim of learning.

    Without this, "the policy solves held-out apps" says nothing about the
    data: the feature set alone might solve it, in which case the training step
    is decoration.
    """
    rng = random.Random(seed)
    return Grounder(weights={k: rng.gauss(0.0, scale) for k in reference.weights})


def merge(*models: Grounder) -> Grounder:
    """Average several fits — used to check a result is not one lucky seed."""
    names: set[str] = set()
    for model in models:
        names |= set(model.weights)
    return Grounder(weights={
        name: sum(m.weights.get(name, 0.0) for m in models) / len(models)
        for name in names
    })


def top_weights(model: Grounder, count: int = 12) -> list[tuple[str, float]]:
    """The largest weights, for reading what the model decided mattered."""
    return sorted(model.weights.items(), key=lambda kv: abs(kv[1]), reverse=True)[:count]


__all__ = [
    "Example",
    "Goal",
    "Grounder",
    "LearnedPolicy",
    "accuracy",
    "examples_from",
    "features",
    "merge",
    "policy_factory",
    "read_instruction",
    "top_weights",
    "train",
    "untrained",
]

