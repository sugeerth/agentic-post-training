"""Self-improvement: the agent's own attempts become its next training set.

`learn` trains on demonstrations. Demonstrations are the expensive part — in
this repo they come free from search, but in any real setting they are the
budget. The question this module answers is the one that actually matters for
post-training: **given very few demonstrations and a verifier, can an agent
improve by practising?**

The loop follows the shape that current GUI-agent work has converged on
(UI-Voyager, arXiv 2603.24533), in two parts:

  **Rejection fine-tuning.** Roll the current policy out several times per
  task, keep the attempts the verifier passes, and retrain on them. No labels,
  no human, no gold — just attempts that provably reached the goal state.

  **Fork-point supervision.** Rejection sampling throws away every failure,
  which is most of the data early on. When a task has both a success and a
  failure in the same group, the two runs agree up to some step and then
  diverge — and at that step the successful run is a *correction* for the
  failed one. The screen is the same in both (they agreed until then), so the
  failed run's frame paired with the successful run's choice is a labelled
  decision at exactly the point the failure was decided.

Two things keep this from being self-congratulation:

  **The verifier is ground truth, not the model.** Recent work finds that
  self-improving loops built on self-authored verification degrade quietly
  (arXiv 2607.24300) — the agent's opinion of its own work drifts up while its
  ability does not. Here the verifier is an exact state check the policy cannot
  see, so nothing it does can move the bar.

  **Both numbers are reported every round.** The documented failure mode of
  verifier-in-the-loop training is that the training-pool pass rate climbs
  while held-out accuracy stalls. `EvolveReport` prints them side by side, so
  that pattern is visible rather than inferred.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from computer_use.learn import (
    Example,
    Goal,
    Grounder,
    LearnedPolicy,
    _target_index,
    accuracy,
    examples_from,
    features,
    read_instruction,
    train,
)
from computer_use.perception import parse_screen
from computer_use.policies import ScriptedPolicy
from computer_use.rollout import RolloutConfig, run_episode
from computer_use.types import ActionKind, Trajectory


@dataclass(frozen=True, slots=True)
class Round:
    """What one round of practice produced."""

    index: int
    attempts: int
    solved: int
    from_success: int
    from_forks: int
    pool: int
    train_accuracy: float
    held_out: int
    held_out_total: int

    @property
    def practice_rate(self) -> float:
        """How often practice succeeded — what the loop can see."""
        return self.solved / self.attempts if self.attempts else 0.0

    @property
    def held_out_rate(self) -> float:
        """How often it works on applications never seen — the truth."""
        return self.held_out / self.held_out_total if self.held_out_total else 0.0

    def to_dict(self) -> dict[str, float | int]:
        return {
            "round": self.index,
            "attempts": self.attempts,
            "solved": self.solved,
            "practice_rate": round(self.practice_rate, 4),
            "from_success": self.from_success,
            "from_forks": self.from_forks,
            "pool": self.pool,
            "train_accuracy": round(self.train_accuracy, 4),
            "held_out": self.held_out,
            "held_out_rate": round(self.held_out_rate, 4),
        }


@dataclass
class EvolveReport:
    """The whole run: where it started, and what practice added."""

    rounds: list[Round] = field(default_factory=list)
    seed_tasks: int = 0
    practice_tasks: int = 0
    held_out_total: int = 0
    model: Grounder | None = None

    @property
    def start(self) -> float:
        return self.rounds[0].held_out_rate if self.rounds else 0.0

    @property
    def best(self) -> float:
        return max((r.held_out_rate for r in self.rounds), default=0.0)

    @property
    def stalled(self) -> bool:
        """Practice getting easier while held-out accuracy does not move.

        The signature failure of verifier-in-the-loop training: the pool fills
        with tasks the policy already solves, the visible number climbs, and
        nothing transfers.
        """
        if len(self.rounds) < 3:
            return False
        last = self.rounds[-1]
        first = self.rounds[0]
        return (
            last.practice_rate > first.practice_rate + 0.10
            and last.held_out_rate <= first.held_out_rate + 0.02
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "seed_tasks": self.seed_tasks,
            "practice_tasks": self.practice_tasks,
            "held_out_total": self.held_out_total,
            "start": round(self.start, 4),
            "best": round(self.best, 4),
            "stalled": self.stalled,
            "rounds": [r.to_dict() for r in self.rounds],
        }


# --------------------------------------------------------------------------- #
# Fork points
# --------------------------------------------------------------------------- #


def fork_supervision(good: Trajectory, bad: Trajectory) -> list[Example]:
    """Label the step where a failed run stopped agreeing with a successful one.

    Both runs start in the same place and take the same actions for a while.
    The first step whose *effect* differs — a different control, not merely
    different pixels — is where the failure was decided, and up to that point
    the two agents were looking at the same screen. So the failed run's frame
    at that step, labelled with what the successful run clicked, is a correct
    decision at precisely the moment the agent got it wrong.

    Comparing effects rather than coordinates matters. Two clicks twenty pixels
    apart on the same button are the same decision, and blaming a failure on
    the difference between them produces supervision whose "correction" is a
    control the agent already chose.
    """
    if not good.succeeded or bad.succeeded:
        return []
    goals = read_instruction(good.task)
    if not goals:
        return []

    cursor = 0
    for left, right in zip(good.steps, bad.steps, strict=False):
        matched_kind = left.action.kind is right.action.kind
        if matched_kind and left.action.kind is not ActionKind.LEFT_CLICK:
            continue  # both typed, both scrolled: no decision to compare
        if right.action.kind is not ActionKind.LEFT_CLICK:
            # The failed run did something a click cannot correct. Steps after
            # this no longer line up, and guessing across the gap is how a
            # correction gets attached to the wrong screen.
            break
        if (
            matched_kind
            and left.metadata.get("target") == right.metadata.get("target")
        ):
            # Same effect, whatever the coordinates were: still in agreement.
            cursor += _satisfied(left, goals, cursor)
            continue
        if right.observation is None or left.action.coordinate is None:
            break

        screen = parse_screen(right.observation.data)
        correct = _target_index(screen, left.action.coordinate)
        if correct is None:
            break
        goal = goals[min(cursor, len(goals) - 1)]
        evidence = features(goal, screen.elements[correct], screen)
        satisfied = evidence["overlap"] > 0.4 or evidence["above_overlap"] > 0.4
        labelled = goal if satisfied else Goal("navigate", goal.phrase, goal.value)
        return [Example(
            candidates=tuple(features(labelled, e, screen) for e in screen.elements),
            correct=correct,
        )]
    return []


def _satisfied(step: object, goals: Sequence[Goal], cursor: int) -> int:
    """Whether this step finished the goal it was working on.

    Judged the same way `learn.examples_from` judges it — by reading the screen
    and asking whether the control the click landed on is the one the goal
    names. Inferring it from the widget's internal id instead would be reading
    ground truth, and it would disagree with the labels every other decision in
    the pool was built with.
    """
    if cursor >= len(goals) - 1:
        return 0
    observation = getattr(step, "observation", None)
    coordinate = getattr(step, "action", None) and step.action.coordinate  # type: ignore[attr-defined]
    if observation is None or coordinate is None:
        return 0
    screen = parse_screen(observation.data)
    index = _target_index(screen, coordinate)
    if index is None:
        return 0
    evidence = features(goals[cursor], screen.elements[index], screen)
    return 1 if evidence["overlap"] > 0.4 or evidence["above_overlap"] > 0.4 else 0


def harvest(group: Sequence[Trajectory]) -> tuple[list[Example], int, int]:
    """Turn one task's attempts into training decisions.

    Successes teach directly. Failures teach only where a sibling succeeded,
    which is what makes them safe to use: a failed run on its own is not known
    to be wrong at any particular step, and guessing which step to blame is how
    a training set fills with confidently mislabelled corrections.
    """
    wins = [t for t in group if t.succeeded]
    losses = [t for t in group if not t.succeeded]
    decisions = examples_from(wins)
    from_success = len(decisions)

    forks: list[Example] = []
    if wins:
        best = min(wins, key=lambda t: len(t.steps))
        for lost in losses:
            forks.extend(fork_supervision(best, lost))
    return decisions + forks, from_success, len(forks)


# --------------------------------------------------------------------------- #
# The loop
# --------------------------------------------------------------------------- #


def evolve(
    *,
    seed_worlds: Sequence[int],
    practice_worlds: Sequence[int],
    test_worlds: Sequence[int],
    rounds: int = 4,
    group_size: int = 6,
    temperature: float = 0.8,
    per_world: int = 3,
    max_steps: int = 24,
    epochs: int = 30,
    use_forks: bool = True,
    seed: int = 0,
    hard: bool = False,
) -> EvolveReport:
    """Practise on unlabelled applications, keep what the verifier passes.

    `seed_worlds` supply the only demonstrations — deliberately few, because
    the point is what practice adds rather than what more gold would.
    `practice_worlds` are never demonstrated: the agent only ever sees its own
    attempts there, filtered by a verifier it cannot read. `test_worlds` are
    touched once per round, to score, and never trained on.
    """
    from computer_use.worlds import curriculum

    _no_overlap(seed_worlds, practice_worlds, "seed", "practice")
    _no_overlap(seed_worlds, test_worlds, "seed", "test")
    _no_overlap(practice_worlds, test_worlds, "practice", "test")

    config = RolloutConfig(store_frames=True, max_steps=max_steps)
    seeded = curriculum(seed_worlds, per_world=per_world, hard=hard)
    practice = curriculum(practice_worlds, per_world=per_world, hard=hard)
    held_out = curriculum(test_worlds, per_world=per_world, hard=hard)

    pool = examples_from([
        _episode(task, ScriptedPolicy(list(task.gold)), config) for task in seeded
    ])
    model = train(pool, epochs=epochs, seed=seed)

    report = EvolveReport(
        seed_tasks=len(seeded), practice_tasks=len(practice),
        held_out_total=len(held_out),
    )
    report.rounds.append(Round(
        index=0, attempts=0, solved=0, from_success=len(pool), from_forks=0,
        pool=len(pool), train_accuracy=accuracy(model, pool),
        held_out=_score(model, held_out, config), held_out_total=len(held_out),
    ))

    for index in range(1, rounds + 1):
        attempts = solved = gained = forked = 0
        fresh: list[Example] = []
        for offset, task in enumerate(practice):
            group = [
                _episode(
                    task,
                    LearnedPolicy(
                        model=model, temperature=temperature,
                        seed=seed + 1000 * index + 10 * offset + n,
                    ),
                    config,
                )
                for n in range(group_size)
            ]
            attempts += len(group)
            solved += sum(t.succeeded for t in group)
            decisions, from_success, from_forks = harvest(group)
            if not use_forks:
                decisions = decisions[:from_success]
                from_forks = 0
            fresh.extend(decisions)
            gained += from_success
            forked += from_forks

        pool = pool + fresh
        model = train(pool, epochs=epochs, seed=seed + index)
        report.rounds.append(Round(
            index=index, attempts=attempts, solved=solved, from_success=gained,
            from_forks=forked, pool=len(pool), train_accuracy=accuracy(model, pool),
            held_out=_score(model, held_out, config), held_out_total=len(held_out),
        ))

    report.model = model
    return report


def _score(model: Grounder, tasks: Iterable[object], config: RolloutConfig) -> int:
    return sum(
        _episode(task, LearnedPolicy(model=model), config).succeeded for task in tasks
    )


def _episode(task: object, policy: object, config: RolloutConfig) -> Trajectory:
    return asyncio.run(run_episode(
        task.instruction,  # type: ignore[attr-defined]
        task.env_factory(),  # type: ignore[attr-defined]
        policy,  # type: ignore[arg-type]
        verifier=task.verifier,  # type: ignore[attr-defined]
        reward_config=task.reward_config(),  # type: ignore[attr-defined]
        config=config,
    ))


def _no_overlap(a: Sequence[int], b: Sequence[int], left: str, right: str) -> None:
    shared = set(a) & set(b)
    if shared:
        raise ValueError(
            f"{left} and {right} worlds overlap on seeds {sorted(shared)} — the "
            "result would measure memorization rather than improvement"
        )


def format_report(report: EvolveReport) -> str:
    """The two numbers side by side, because only one of them is the truth."""
    lines = [
        "",
        f"  {report.seed_tasks} demonstrated tasks, "
        f"{report.practice_tasks} practised with no demonstrations, "
        f"{report.held_out_total} held out",
        "",
        f"  {'round':<7}{'practice':>10}{'kept':>8}{'forks':>7}"
        f"{'pool':>7}{'held out':>11}",
        "  " + "─" * 50,
    ]
    for entry in report.rounds:
        practice = f"{entry.practice_rate:.0%}" if entry.attempts else "—"
        lines.append(
            f"  {entry.index:<7}{practice:>10}{entry.from_success:>8}"
            f"{entry.from_forks:>7}{entry.pool:>7}"
            f"{entry.held_out:>6}/{entry.held_out_total:<4}"
            f"{entry.held_out_rate:>5.0%}"
        )
    lines.append("  " + "─" * 50)
    delta = report.best - report.start
    lines.append(
        f"  held out: {report.start:.0%} → {report.best:.0%} ({delta:+.0%}) "
        "with no new demonstrations"
    )
    if report.stalled:
        lines.append(
            "  ⚠ practice is getting easier while held-out accuracy is not "
            "moving — the loop is feeding on what it already solves"
        )
    return "\n".join(lines)


__all__ = [
    "EvolveReport",
    "Round",
    "evolve",
    "fork_supervision",
    "format_report",
    "harvest",
]
