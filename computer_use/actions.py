"""The computer-use action-space contract: tool schema in, validation out.

Two jobs:

  1. Build the tool definition sent to the API. The computer-use tool is
     *Anthropic-defined and schema-less* — you declare it by `type` and `name`
     plus the display geometry, and never supply an `input_schema`. The tool
     version string is a parameter rather than a constant because the right
     one depends on the model you point the policy at.

  2. Validate an `Action` before handing it to an environment. A VLM will
     occasionally emit `left_click` with no coordinate or `type` with no text;
     the loop needs to turn that into a scored failure and a corrective
     `tool_result`, not an unhandled `KeyError` three frames deeper.
"""

from __future__ import annotations

from typing import Any

from computer_use.types import (
    POINTING_ACTIONS,
    TEXT_ACTIONS,
    Action,
    ActionKind,
)

#: Default tool version + matching beta flag. Both are constructor arguments on
#: `ClaudeComputerUsePolicy`, so pointing the policy at an older model is a
#: config change, not a code change.
DEFAULT_TOOL_VERSION = "computer_20251124"
DEFAULT_TOOL_BETA = "computer-use-2025-11-24"

#: Screenshot geometry. Sending frames at 1080p balances grounding accuracy
#: against image-token cost; 1366x768 and 720p are cheaper options that still
#: perform well. Larger is not automatically better — image tokens scale with
#: area, and past the model's resolution ceiling the frame is downscaled anyway.
RESOLUTIONS: dict[str, tuple[int, int]] = {
    "1080p": (1920, 1080),
    "wxga": (1366, 768),
    "720p": (1280, 720),
    "xga": (1024, 768),
}

VALID_SCROLL_DIRECTIONS = frozenset({"up", "down", "left", "right"})


class ActionError(ValueError):
    """An action is well-named but its arguments don't satisfy its contract."""


def computer_tool(
    width: int,
    height: int,
    *,
    version: str = DEFAULT_TOOL_VERSION,
    display_number: int | None = None,
) -> dict[str, Any]:
    """Build the computer-use tool definition.

    Note the absence of `input_schema`: this is an Anthropic-defined tool whose
    schema is built into the model. Passing your own schema — or naming a
    custom tool `computer` — creates a *different*, ordinary tool that does not
    carry the trained behavior.

    `display_number` is the X11 display for a self-hosted Linux VM; leave it
    unset for environments that don't have one (a browser, a macOS host).
    """
    tool: dict[str, Any] = {
        "type": version,
        "name": "computer",
        "display_width_px": int(width),
        "display_height_px": int(height),
    }
    if display_number is not None:
        tool["display_number"] = int(display_number)
    return tool


def validate(action: Action, *, width: int | None = None, height: int | None = None) -> None:
    """Raise `ActionError` if `action` can't be executed as written.

    When `width`/`height` are given, coordinates are bounds-checked too — an
    off-screen click is a grounding failure worth surfacing to the model, and
    silently clamping it would hide the error from both the agent and the
    reward model.
    """
    kind = action.kind

    if kind in POINTING_ACTIONS and action.coordinate is None:
        raise ActionError(f"{kind} requires a coordinate")

    if kind is ActionKind.LEFT_CLICK_DRAG and action.start_coordinate is None:
        raise ActionError("left_click_drag requires start_coordinate and coordinate")

    if kind in TEXT_ACTIONS and not action.text:
        raise ActionError(f"{kind} requires non-empty text")

    if kind is ActionKind.SCROLL:
        if action.coordinate is None:
            raise ActionError("scroll requires a coordinate (the point to scroll over)")
        if action.scroll_direction not in VALID_SCROLL_DIRECTIONS:
            raise ActionError(
                f"scroll_direction must be one of {sorted(VALID_SCROLL_DIRECTIONS)}, "
                f"got {action.scroll_direction!r}"
            )
        if action.scroll_amount is None or action.scroll_amount <= 0:
            raise ActionError("scroll requires a positive scroll_amount")

    if kind is ActionKind.WAIT and (action.duration is None or action.duration <= 0):
        raise ActionError("wait requires a positive duration in seconds")

    if width is not None and height is not None:
        for label, point in (("coordinate", action.coordinate),
                             ("start_coordinate", action.start_coordinate)):
            if point is None:
                continue
            x, y = point
            if not (0 <= x < width and 0 <= y < height):
                raise ActionError(
                    f"{label} ({x}, {y}) is outside the {width}x{height} display"
                )


def parse_tool_use(block: Any) -> Action:
    """Turn an API `tool_use` content block into an `Action`.

    Accepts either an SDK block object (attribute access) or the plain dict a
    replayed transcript gives you.
    """
    payload = block.get("input") if isinstance(block, dict) else getattr(block, "input", None)
    if payload is None:
        raise ValueError("tool_use block has no input payload")
    return Action.from_tool_input(payload)


def system_prompt(task_context: str = "") -> str:
    """A GUI-agent system prompt, deliberately short.

    The instructions here are the ones the model cannot infer: that a
    screenshot follows every action, that coordinates are read off the image it
    was just shown, and that finishing means saying so in text rather than
    calling another tool. Everything else — how to plan, when to scroll, how to
    recover from a misclick — the model already does well, and scripting it
    tends to make GUI agents worse rather than better.
    """
    base = (
        "You are operating a computer through screenshots.\n\n"
        "Each action you take returns a fresh screenshot. Read coordinates "
        "directly off the most recent screenshot — they are in its pixel space.\n\n"
        "Take a screenshot first if you are unsure of the current state. "
        "Verify that an action had the effect you expected before building on it; "
        "if a click missed, look again rather than repeating it blindly.\n\n"
        "When the task is complete, stop calling the computer tool and reply in "
        "text with what you did and what the final state is. If the task cannot "
        "be completed, say so plainly and explain what blocked you."
    )
    return f"{base}\n\n{task_context}".strip() if task_context else base
