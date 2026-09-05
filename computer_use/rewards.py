"""Verification and reward shaping for GUI trajectories.

The reward is the whole ballgame for post-training. A GUI agent that gets
credit for "looked busy" learns to look busy, so this module keeps two things
strictly apart:

  **Verification** — did the environment reach the goal state? Binary, checked
  against ground truth, no model in the loop. Everything downstream trusts it.

  **Shaping** — given that verdict, how good was the *path*? Efficiency,
  grounding accuracy, and redundancy are what separate two successful
  trajectories, and separating them is exactly what preference data needs.

Shaping never turns a failure into a win: every shaping term is scaled by the
verdict or bounded below it, so the ordering "any success > any failure" holds.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from itertools import pairwise
from typing import Any, Protocol, runtime_checkable

from computer_use.types import Step, Trajectory, TrajectoryStatus


@dataclass(frozen=True, slots=True)
class Verdict:
    """The outcome of checking a trajectory against its goal."""

    success: bool
    reason: str = ""
    #: Per-criterion results, so a partial failure says *which* part failed.
    details: Mapping[str, bool] = field(default_factory=dict)

    @property
    def partial_credit(self) -> float:
        """Fraction of criteria met. 1.0 on success, useful signal on failure."""
        if self.success:
            return 1.0
        if not self.details:
            return 0.0
        return sum(1 for ok in self.details.values() if ok) / len(self.details)


@runtime_checkable
class Verifier(Protocol):
    """Checks a finished episode against ground truth."""

    name: str

    def __call__(self, trajectory: Trajectory, state: Mapping[str, Any]) -> Verdict: ...


#: A criterion is either a literal to compare against, or a predicate.
Criterion = Any | Callable[[Any], bool]


@dataclass
class StateVerifier:
    """Checks final environment state against expected values.

    Values may be literals (compared with `==`) or callables (predicates):

        StateVerifier({
            "email": "ada@example.com",
            "notify": True,
            "retries": lambda v: v.isdigit() and int(v) <= 5,
            "saved": True,
        })

    Exact-match verification is the reason the mock environment exposes its
    state. It keeps the reward honest while the pipeline is under development;
    swap in a judge only where ground truth genuinely isn't available.
    """

    expected: Mapping[str, Criterion]
    name: str = "state"
    case_sensitive: bool = True

    def __call__(self, trajectory: Trajectory, state: Mapping[str, Any]) -> Verdict:
        details: dict[str, bool] = {}
        for key, criterion in self.expected.items():
            actual = state.get(key)
            if callable(criterion):
                try:
                    details[key] = bool(criterion(actual))
                except Exception:
                    details[key] = False
            elif isinstance(criterion, str) and isinstance(actual, str) and not self.case_sensitive:
                details[key] = criterion.casefold() == actual.casefold()
            else:
                details[key] = criterion == actual

        missed = [k for k, ok in details.items() if not ok]
        if missed:
            return Verdict(False, f"unmet: {', '.join(sorted(missed))}", details)
        return Verdict(True, "all criteria met", details)


@dataclass
class AllOf:
    """Conjunction of verifiers. Succeeds only if every child succeeds."""

    verifiers: Sequence[Verifier]
    name: str = "all_of"

    def __call__(self, trajectory: Trajectory, state: Mapping[str, Any]) -> Verdict:
        details: dict[str, bool] = {}
        reasons: list[str] = []
        for verifier in self.verifiers:
            verdict = verifier(trajectory, state)
            details[verifier.name] = verdict.success
            if not verdict.success:
                reasons.append(f"{verifier.name}: {verdict.reason}")
        success = all(details.values())
        return Verdict(success, "; ".join(reasons) or "all verifiers passed", details)


# --------------------------------------------------------------------------- #
# Shaping
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RewardConfig:
    """Weights for the shaped reward.

    Two properties the defaults are chosen to guarantee, both worth preserving
    if you retune them:

    **Successes never saturate.** The weights sum to exactly 1.0, which is also
    the clip ceiling, so a flawless run scores 1.0 and every other success
    lands strictly below it. Weights that overflow the clip would collapse all
    successes to the same number and quietly delete the shaping signal — which
    is the whole reason to shape.

    **Success dominates.** `success_weight` (0.55) exceeds everything a failing
    trajectory can collect (`partial_credit` + `grounding` = 0.25), so no
    amount of efficient, well-aimed flailing outranks a clumsy success.
    """

    success_weight: float = 0.55
    partial_credit_weight: float = 0.10
    efficiency_weight: float = 0.20
    grounding_weight: float = 0.15
    invalid_action_penalty: float = 0.10
    redundancy_penalty: float = 0.05
    #: Steps a competent agent needs. Efficiency is measured against this, not
    #: against "fewest possible", so an agent isn't punished for verifying.
    optimal_steps: int = 6
    #: Discount for spreading terminal reward back over the steps that earned
    #: it. 1.0 gives every step equal credit; lower favors later steps.
    discount: float = 0.95


def score_trajectory(
    trajectory: Trajectory,
    verdict: Verdict,
    config: RewardConfig | None = None,
) -> Trajectory:
    """Return a copy of `trajectory` with rewards and a scoring breakdown.

    The terminal reward lands on `Trajectory.reward`; per-step credit lands on
    each `Step.reward`, which is what a step-level RL method consumes. The
    breakdown goes into `metadata["reward_breakdown"]` so a regression in the
    aggregate number can be traced to the term that moved.
    """
    cfg = config or RewardConfig()
    steps = trajectory.steps

    invalid = sum(1 for s in steps if s.failed)
    effective = max(1, trajectory.num_effective_steps)
    total = max(1, len(steps))

    # Efficiency only pays out on success — "fast and wrong" is not a virtue.
    efficiency = min(1.0, cfg.optimal_steps / effective) if verdict.success else 0.0

    grounded = [s for s in steps if "hit" in s.metadata]
    grounding = (
        sum(1 for s in grounded if s.metadata["hit"]) / len(grounded) if grounded else 0.0
    )

    redundancy = _redundancy_rate(steps)

    components = {
        "success": cfg.success_weight * (1.0 if verdict.success else 0.0),
        "partial_credit": cfg.partial_credit_weight * verdict.partial_credit,
        "efficiency": cfg.efficiency_weight * efficiency,
        "grounding": cfg.grounding_weight * grounding,
        "invalid_actions": -cfg.invalid_action_penalty * (invalid / total),
        "redundancy": -cfg.redundancy_penalty * redundancy,
    }
    reward = max(-1.0, min(1.0, sum(components.values())))

    scored_steps = tuple(
        replace(step, reward=credit)
        for step, credit in zip(steps, assign_step_credit(steps, reward, cfg), strict=True)
    )

    status = trajectory.status
    if status is TrajectoryStatus.SUCCESS and not verdict.success:
        status = TrajectoryStatus.FAILURE
    elif verdict.success and status in (TrajectoryStatus.FAILURE, TrajectoryStatus.MAX_STEPS):
        status = TrajectoryStatus.SUCCESS

    return replace(
        trajectory,
        steps=scored_steps,
        status=status,
        reward=reward,
        metadata={
            **trajectory.metadata,
            "reward_breakdown": components,
            "verdict": {
                "success": verdict.success,
                "reason": verdict.reason,
                "details": dict(verdict.details),
            },
            "stats": {
                "steps": len(steps),
                "effective_steps": trajectory.num_effective_steps,
                "invalid_actions": invalid,
                "grounding_rate": grounding,
                "redundancy_rate": redundancy,
            },
        },
    )


def assign_step_credit(
    steps: Sequence[Step], terminal_reward: float, config: RewardConfig | None = None
) -> list[float]:
    """Spread the terminal reward back over the steps, discounted.

    Step-level RL needs a per-step signal, and a GUI episode gives you exactly
    one ground-truth number at the end. Discounting from the terminal step is
    the standard, honest approximation: later steps get more credit because
    they are closer to the outcome. Invalid actions carry their own local
    penalty on top, since those are attributable to the step that made them.
    """
    cfg = config or RewardConfig()
    n = len(steps)
    if n == 0:
        return []
    credits: list[float] = []
    for i, step in enumerate(steps):
        discounted = terminal_reward * (cfg.discount ** (n - 1 - i))
        if step.failed:
            discounted -= cfg.invalid_action_penalty
        credits.append(discounted)
    return credits


def _redundancy_rate(steps: Sequence[Step]) -> float:
    """Fraction of steps that repeat the immediately preceding action.

    Repeating an action verbatim is the signature failure of a GUI agent that
    isn't reading its screenshots — it clicks the same dead pixel five times.
    Taking a screenshot twice in a row counts too.

    Identical coordinates are not enough to call it, though. A wizard's
    CONTINUE and the SUBMIT on the screen after it routinely occupy the same
    rectangle, so pressing the same pixel twice can be the *shortest* path
    rather than a stuck agent. Where the environment can name the control that
    was hit, that name decides; where it cannot, the coordinates are all there
    is.
    """
    if len(steps) < 2:
        return 0.0
    repeats = 0
    for prev, cur in pairwise(steps):
        if prev.action.to_tool_input() != cur.action.to_tool_input():
            continue
        targets = [s.metadata.get("target") for s in (prev, cur)
                   if "target" in s.metadata]
        if len(targets) == 2 and targets[0] != targets[1]:
            continue
        repeats += 1
    return repeats / (len(steps) - 1)
