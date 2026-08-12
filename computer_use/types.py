"""Typed data carriers for computer-use (GUI) rollouts.

Same convention as `core.types`: frozen dataclasses with `slots`, not Pydantic.
A rollout produces one `Step` per model turn and these objects are created in a
hot loop; validation belongs in `computer_use.actions`, not in the carrier.

The vocabulary here mirrors Anthropic's computer-use tool so a `Action` can be
serialized straight into a `tool_use.input` payload and back, with no adapter
layer in between:

    Action.from_tool_input({"action": "left_click", "coordinate": [640, 400]})
    action.to_tool_input()  ->  {"action": "left_click", "coordinate": [640, 400]}
"""

from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any


class ActionKind(str, Enum):
    """The computer-use action space.

    Values match the `action` field the model emits in `tool_use.input`, so the
    enum doubles as the wire format. Ordered roughly by how often a GUI agent
    reaches for them.
    """

    SCREENSHOT = "screenshot"
    LEFT_CLICK = "left_click"
    RIGHT_CLICK = "right_click"
    MIDDLE_CLICK = "middle_click"
    DOUBLE_CLICK = "double_click"
    TRIPLE_CLICK = "triple_click"
    MOUSE_MOVE = "mouse_move"
    LEFT_CLICK_DRAG = "left_click_drag"
    LEFT_MOUSE_DOWN = "left_mouse_down"
    LEFT_MOUSE_UP = "left_mouse_up"
    SCROLL = "scroll"
    TYPE = "type"
    KEY = "key"
    HOLD_KEY = "hold_key"
    CURSOR_POSITION = "cursor_position"
    WAIT = "wait"

    def __str__(self) -> str:  # so f-strings render "left_click", not "ActionKind.LEFT_CLICK"
        return self.value


#: Actions that move or press the pointer and therefore require `coordinate`.
POINTING_ACTIONS = frozenset({
    ActionKind.LEFT_CLICK,
    ActionKind.RIGHT_CLICK,
    ActionKind.MIDDLE_CLICK,
    ActionKind.DOUBLE_CLICK,
    ActionKind.TRIPLE_CLICK,
    ActionKind.MOUSE_MOVE,
    ActionKind.LEFT_CLICK_DRAG,
})

#: Actions that carry a `text` payload (keystrokes or literal text).
TEXT_ACTIONS = frozenset({ActionKind.TYPE, ActionKind.KEY, ActionKind.HOLD_KEY})

#: Actions that change nothing on screen. Used by the efficiency reward to tell
#: "the agent looked again" apart from "the agent did something".
READ_ONLY_ACTIONS = frozenset({
    ActionKind.SCREENSHOT,
    ActionKind.CURSOR_POSITION,
    ActionKind.WAIT,
    ActionKind.MOUSE_MOVE,
})


@dataclass(frozen=True, slots=True)
class Screenshot:
    """One frame of the environment, as the VLM sees it.

    `data` is the raw encoded image (PNG unless `media_type` says otherwise).
    We keep bytes rather than a PIL handle so a trajectory can be pickled,
    written to JSONL, and replayed on a machine without Pillow installed.
    """

    data: bytes
    width: int
    height: int
    media_type: str = "image/png"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_base64(self) -> str:
        return base64.standard_b64encode(self.data).decode("utf-8")

    def to_content_block(self) -> dict[str, Any]:
        """Render as an Anthropic `image` content block."""
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": self.media_type,
                "data": self.to_base64(),
            },
        }

    @classmethod
    def from_base64(
        cls, b64: str, width: int, height: int, media_type: str = "image/png"
    ) -> Screenshot:
        return cls(base64.standard_b64decode(b64), width, height, media_type)

    def __repr__(self) -> str:  # never dump megabytes of PNG into a traceback
        return f"<Screenshot {self.width}x{self.height} {len(self.data)}B>"


@dataclass(frozen=True, slots=True)
class Action:
    """A single computer-use action.

    Every field except `kind` is optional because the action space is a tagged
    union — `left_click` needs `coordinate`, `type` needs `text`, `wait` needs
    neither. `computer_use.actions.validate` enforces the per-kind contract;
    constructing an invalid `Action` is allowed so a malformed model output can
    be captured in the trajectory and scored as a failure rather than crashing
    the rollout.
    """

    kind: ActionKind
    coordinate: tuple[int, int] | None = None
    start_coordinate: tuple[int, int] | None = None
    text: str | None = None
    scroll_direction: str | None = None
    scroll_amount: int | None = None
    duration: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    # ---- wire format ------------------------------------------------------ #

    def to_tool_input(self) -> dict[str, Any]:
        """Serialize to the `tool_use.input` shape the API uses."""
        payload: dict[str, Any] = {"action": self.kind.value}
        if self.coordinate is not None:
            payload["coordinate"] = list(self.coordinate)
        if self.start_coordinate is not None:
            payload["start_coordinate"] = list(self.start_coordinate)
        if self.text is not None:
            payload["text"] = self.text
        if self.scroll_direction is not None:
            payload["scroll_direction"] = self.scroll_direction
        if self.scroll_amount is not None:
            payload["scroll_amount"] = self.scroll_amount
        if self.duration is not None:
            payload["duration"] = self.duration
        return payload

    @classmethod
    def from_tool_input(cls, payload: Mapping[str, Any]) -> Action:
        """Parse a `tool_use.input` payload.

        Raises `ValueError` on an unknown action name — that is a genuine
        protocol violation, distinct from a well-named action with bad
        arguments (which `actions.validate` reports).
        """
        raw = payload.get("action")
        try:
            kind = ActionKind(raw)
        except ValueError as exc:
            known = ", ".join(sorted(k.value for k in ActionKind))
            raise ValueError(f"unknown action {raw!r}. Known actions: {known}") from exc

        return cls(
            kind=kind,
            coordinate=_as_point(payload.get("coordinate")),
            start_coordinate=_as_point(payload.get("start_coordinate")),
            text=payload.get("text"),
            scroll_direction=payload.get("scroll_direction"),
            scroll_amount=payload.get("scroll_amount"),
            duration=payload.get("duration"),
        )

    # ---- convenience ------------------------------------------------------ #

    def scaled(self, factor_x: float, factor_y: float) -> Action:
        """Rescale coordinates — for when the model saw a resized screenshot.

        Not needed on models with high-resolution vision, where the coordinates
        the model returns already map 1:1 to screenshot pixels.
        """
        def _scale(pt: tuple[int, int] | None) -> tuple[int, int] | None:
            if pt is None:
                return None
            return (round(pt[0] * factor_x), round(pt[1] * factor_y))

        return replace(
            self,
            coordinate=_scale(self.coordinate),
            start_coordinate=_scale(self.start_coordinate),
        )

    @property
    def is_read_only(self) -> bool:
        return self.kind in READ_ONLY_ACTIONS

    def describe(self) -> str:
        """One-line human summary, for logs and the message bus."""
        if self.kind in POINTING_ACTIONS and self.coordinate:
            target = f"{self.coordinate[0]},{self.coordinate[1]}"
            if self.kind is ActionKind.LEFT_CLICK_DRAG and self.start_coordinate:
                start = f"{self.start_coordinate[0]},{self.start_coordinate[1]}"
                return f"{self.kind} {start} → {target}"
            return f"{self.kind} @ {target}"
        if self.kind in TEXT_ACTIONS and self.text is not None:
            preview = self.text if len(self.text) <= 40 else self.text[:37] + "..."
            return f"{self.kind} {preview!r}"
        if self.kind is ActionKind.SCROLL:
            return f"scroll {self.scroll_direction} x{self.scroll_amount}"
        return str(self.kind)


def _as_point(value: Any) -> tuple[int, int] | None:
    """Coerce `[x, y]` (or a 2-tuple) to a tuple of ints; `None` passes through."""
    if value is None:
        return None
    if isinstance(value, Sequence) and not isinstance(value, str | bytes) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    raise ValueError(f"coordinate must be a 2-element sequence, got {value!r}")


@dataclass(frozen=True, slots=True)
class Step:
    """One turn of the loop: what the agent saw, thought, did, and got back.

    `observation` is the frame the model conditioned on *before* acting, which
    is what a training example needs. `result` is the frame after the action —
    it becomes the next step's observation, so it is stored only on the last
    step of a trajectory unless `keep_frames` is on.
    """

    index: int
    action: Action
    observation: Screenshot | None = None
    result: Screenshot | None = None
    rationale: str = ""
    error: str | None = None
    reward: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def failed(self) -> bool:
        return self.error is not None


class TrajectoryStatus(str, Enum):
    SUCCESS = "success"          # the verifier confirmed the goal state
    FAILURE = "failure"          # the agent stopped but the goal was not reached
    MAX_STEPS = "max_steps"      # the step budget ran out
    ERROR = "error"              # the environment or policy raised
    REFUSED = "refused"          # the model declined the task

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class Trajectory:
    """A complete episode: one task, one attempt.

    This is the unit that flows into post-training. `computer_use.dataset`
    turns groups of these into `PreferencePair`s (for DPO/ORPO/SimPO) or a
    `RolloutBatch` (for GRPO/PPO), so the GUI agent's own experience becomes
    the training signal.
    """

    task: str
    steps: tuple[Step, ...]
    status: TrajectoryStatus
    reward: float = 0.0
    final_response: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.status is TrajectoryStatus.SUCCESS

    @property
    def num_steps(self) -> int:
        return len(self.steps)

    @property
    def num_effective_steps(self) -> int:
        """Steps that actually changed the environment."""
        return sum(1 for s in self.steps if not s.action.is_read_only)

    def action_sequence(self) -> list[str]:
        return [s.action.describe() for s in self.steps]

    def summary(self) -> str:
        return (
            f"[{self.status}] {self.task!r} — {self.num_steps} steps "
            f"({self.num_effective_steps} effective), reward {self.reward:.3f}"
        )
