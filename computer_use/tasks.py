"""A GUI benchmark suite: eight verified tasks across three applications.

Three properties make this a benchmark rather than a pile of demos:

  **Every task is exactly verified.** The check is equality against
  ground-truth environment state, not a judge's opinion, so a score is a fact
  about what happened rather than an estimate.

  **Every task ships a reference solution.** `GUITask.gold` is a working action
  sequence, which gives you a baseline to beat, reference trajectories to pair
  model attempts against — and a test asserting the suite is solvable at all. A
  benchmark nobody has solved is indistinguishable from a broken one.

  **Difficulty is a property of the work, not a label.** `easy` is one
  interaction on a visible widget; `medium` adds off-screen targets and typing;
  `hard` requires navigating between screens, where the agent must remember a
  goal across a change of context.

`optimal_steps` is what a competent agent needs — not the theoretical minimum.
Efficiency is scored against it, so an agent is not punished for taking a
screenshot to check its work.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from computer_use.environments import ComputerEnvironment, MockComputer
from computer_use.rewards import RewardConfig, StateVerifier, Verifier
from computer_use.types import Action, ActionKind

# --------------------------------------------------------------------------- #
# Screen geometry — the click targets in each scenario, named once.
# --------------------------------------------------------------------------- #

# MockComputer.settings_form(): SAVE sits below the fold at 1280x720.
S_EMAIL = (290, 218)
S_RETRIES = (160, 310)
S_NOTIFY = (76, 382)
S_SAVE_SCROLLED = (305, 624)

# MockComputer.checkout_flow(): cart → shipping → payment.
C_CHECKOUT = (160, 354)
C_EXPRESS = (76, 236)
C_CONTINUE = (330, 324)
C_PROMO = (210, 222)
C_PLACE_ORDER = (180, 304)

# MockComputer.file_manager(): list → rename / confirm-delete.
F_REPORT = (74, 174)
F_NOTES = (74, 219)
F_BUDGET = (74, 264)
F_RENAME = (135, 342)
F_DELETE = (305, 342)
F_NEWNAME = (260, 182)
F_RENAME_SAVE = (305, 262)
F_CONFIRM_DELETE = (305, 252)


def click(x: int, y: int) -> Action:
    return Action(ActionKind.LEFT_CLICK, coordinate=(x, y))


def type_text(text: str) -> Action:
    return Action(ActionKind.TYPE, text=text)


def scroll_down(amount: int = 3) -> Action:
    return Action(
        ActionKind.SCROLL, coordinate=(640, 400),
        scroll_direction="down", scroll_amount=amount,
    )


@dataclass(frozen=True)
class GUITask:
    """One benchmark task: an instruction, an environment, and a checkable goal."""

    name: str
    instruction: str
    env_factory: Callable[[], ComputerEnvironment]
    verifier: Verifier
    optimal_steps: int
    difficulty: str  # "easy" | "medium" | "hard"
    #: A known-good action sequence. Baseline, reference trajectory, and the
    #: thing that proves the task is solvable.
    gold: tuple[Action, ...] = ()
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def reward_config(self, **overrides: Any) -> RewardConfig:
        """Reward weights for this task, with its own step budget."""
        return RewardConfig(optimal_steps=self.optimal_steps, **overrides)

    @property
    def spec(self) -> tuple[str, Verifier]:
        """The `(instruction, verifier)` pair `run_suite` takes."""
        return (self.instruction, self.verifier)


SUITE: tuple[GUITask, ...] = (
    # ---- settings form: typing, toggling, and a target below the fold ------ #
    GUITask(
        name="settings.notify",
        instruction="Turn on failure notifications and save the settings.",
        env_factory=MockComputer.settings_form,
        verifier=StateVerifier({"notify": True, "saved": True}),
        optimal_steps=3,
        difficulty="easy",
        gold=(click(*S_NOTIFY), scroll_down(), click(*S_SAVE_SCROLLED)),
        tags=("toggle", "scroll"),
    ),
    GUITask(
        name="settings.email",
        instruction=(
            "Set the email to ada@example.com, turn on failure notifications, "
            "and save."
        ),
        env_factory=MockComputer.settings_form,
        verifier=StateVerifier({
            "email": "ada@example.com", "notify": True, "saved": True,
        }),
        optimal_steps=5,
        difficulty="medium",
        gold=(
            click(*S_EMAIL), type_text("ada@example.com"), click(*S_NOTIFY),
            scroll_down(), click(*S_SAVE_SCROLLED),
        ),
        tags=("typing", "toggle", "scroll"),
    ),
    GUITask(
        name="settings.retries",
        instruction="Set max retries to 5 and save. Leave everything else alone.",
        env_factory=MockComputer.settings_form,
        # The "leave everything else alone" half is verified too: an agent that
        # flips an unrelated toggle on its way through fails this task.
        verifier=StateVerifier({"retries": "5", "saved": True, "notify": False}),
        optimal_steps=4,
        difficulty="medium",
        gold=(
            click(*S_RETRIES), type_text("5"),
            scroll_down(), click(*S_SAVE_SCROLLED),
        ),
        tags=("typing", "scroll", "precision"),
    ),

    # ---- checkout: navigation across screens, single-choice selection ------ #
    GUITask(
        name="checkout.express",
        instruction="Choose express shipping and place the order.",
        env_factory=MockComputer.checkout_flow,
        verifier=StateVerifier({"speed": "express", "ordered": True}),
        optimal_steps=4,
        difficulty="hard",
        gold=(
            click(*C_CHECKOUT), click(*C_EXPRESS),
            click(*C_CONTINUE), click(*C_PLACE_ORDER),
        ),
        tags=("navigation", "radio"),
    ),
    GUITask(
        name="checkout.promo",
        instruction=(
            "Place the order with promo code SPRING20 applied. "
            "Keep the default shipping speed."
        ),
        env_factory=MockComputer.checkout_flow,
        verifier=StateVerifier({
            "promo": "SPRING20", "speed": "standard", "ordered": True,
        }),
        optimal_steps=5,
        difficulty="hard",
        gold=(
            click(*C_CHECKOUT), click(*C_CONTINUE),
            click(*C_PROMO), type_text("SPRING20"), click(*C_PLACE_ORDER),
        ),
        tags=("navigation", "typing", "precision"),
    ),

    # ---- file manager: selection, and a destructive action behind a gate --- #
    GUITask(
        name="files.select",
        instruction="Select budget.csv in the file list.",
        env_factory=MockComputer.file_manager,
        verifier=StateVerifier({"selected": "budget"}),
        optimal_steps=1,
        difficulty="easy",
        gold=(click(*F_BUDGET),),
        tags=("selection",),
    ),
    GUITask(
        name="files.delete",
        instruction="Delete notes.txt.",
        env_factory=MockComputer.file_manager,
        # `deleted` only flips on the confirmation screen, so an agent that
        # clicks DELETE and declares victory fails here.
        verifier=StateVerifier({"selected": "notes", "deleted": True}),
        optimal_steps=3,
        difficulty="medium",
        gold=(click(*F_NOTES), click(*F_DELETE), click(*F_CONFIRM_DELETE)),
        tags=("selection", "navigation", "confirmation"),
    ),
    GUITask(
        name="files.rename",
        instruction="Rename report-2024.pdf to q4-final.pdf.",
        env_factory=MockComputer.file_manager,
        verifier=StateVerifier({
            "selected": "report", "newname": "q4-final.pdf", "renamed": True,
        }),
        optimal_steps=5,
        difficulty="hard",
        gold=(
            click(*F_REPORT), click(*F_RENAME), click(*F_NEWNAME),
            type_text("q4-final.pdf"), click(*F_RENAME_SAVE),
        ),
        tags=("selection", "navigation", "typing"),
    ),
)

DIFFICULTIES = ("easy", "medium", "hard")


def suite(
    *,
    difficulty: str | Sequence[str] | None = None,
    tags: Sequence[str] | None = None,
    names: Sequence[str] | None = None,
) -> tuple[GUITask, ...]:
    """Filter the suite. Every filter is a conjunction; `None` means "any".

    `tags` matches a task carrying *any* of the given tags, which is the useful
    default — you ask for "the navigation tasks", not "tasks that are both
    navigation and typing".
    """
    selected = SUITE
    if difficulty is not None:
        wanted = {difficulty} if isinstance(difficulty, str) else set(difficulty)
        unknown = wanted - set(DIFFICULTIES)
        if unknown:
            raise ValueError(
                f"unknown difficulty {sorted(unknown)}; expected {list(DIFFICULTIES)}"
            )
        selected = tuple(t for t in selected if t.difficulty in wanted)
    if tags is not None:
        wanted_tags = set(tags)
        selected = tuple(t for t in selected if wanted_tags & set(t.tags))
    if names is not None:
        wanted_names = set(names)
        unknown = wanted_names - {t.name for t in SUITE}
        if unknown:
            raise ValueError(f"unknown task(s): {sorted(unknown)}")
        selected = tuple(t for t in selected if t.name in wanted_names)
    return selected


def get_task(name: str) -> GUITask:
    for task in SUITE:
        if task.name == name:
            return task
    known = ", ".join(t.name for t in SUITE)
    raise KeyError(f"unknown task {name!r}. Known tasks: {known}")
