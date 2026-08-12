"""Environments a computer-use policy can drive.

`ComputerEnvironment` is the seam. Everything above it — the rollout loop, the
reward model, the dataset builders — is written against the protocol, so a
trajectory collected in the mock has exactly the same shape as one collected in
a browser or a VM. That is what makes the mock useful: you develop the reward
and the data pipeline against a deterministic environment, then swap in the
real one without touching either.

Two implementations ship here:

  `MockComputer`      — a deterministic widget GUI with no dependencies. Runs
                        in CI, renders real (legible) PNG frames, and exposes
                        ground-truth state so verifiers can be exact.
  `PlaywrightComputer` — a real browser. Imported lazily; Playwright is an
                        optional dependency.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from computer_use._render import Canvas, text_width
from computer_use.actions import ActionError, validate
from computer_use.types import Action, ActionKind, Screenshot


@runtime_checkable
class ComputerEnvironment(Protocol):
    """Anything a GUI agent can act on.

    Async because environments are I/O-bound (a browser round-trip, a VNC
    frame) and the rollout runner fans episodes out concurrently.

    `execute` returns the frame *after* the action. It should raise
    `ActionError` for an action the environment cannot perform — the rollout
    turns that into a corrective tool_result rather than aborting the episode.
    """

    width: int
    height: int

    async def reset(self) -> Screenshot: ...

    async def screenshot(self) -> Screenshot: ...

    async def execute(self, action: Action) -> Screenshot: ...

    def state(self) -> Mapping[str, Any]:
        """Ground-truth state, for verifiers. May be empty for opaque envs."""
        ...

    async def close(self) -> None: ...


# --------------------------------------------------------------------------- #
# Mock environment
# --------------------------------------------------------------------------- #

Color = tuple[int, int, int]

PALETTE: dict[str, Color] = {
    "bg": (246, 247, 249),
    "chrome": (32, 38, 48),
    "chrome_text": (236, 240, 245),
    "panel": (255, 255, 255),
    "border": (188, 195, 206),
    "border_focus": (52, 120, 246),
    "text": (28, 33, 42),
    "muted": (122, 132, 148),
    "accent": (52, 120, 246),
    "accent_text": (255, 255, 255),
    "ok": (34, 152, 92),
    "cursor": (222, 62, 62),
}


@dataclass
class Widget:
    """One interactive element with a bounding box in page coordinates."""

    id: str
    kind: str  # "button" | "field" | "checkbox" | "label"
    x: int
    y: int
    width: int
    height: int
    label: str = ""
    value: str = ""
    checked: bool = False
    placeholder: str = ""
    #: For buttons: the state key set to True when clicked.
    sets: str | None = None

    def contains(self, px: int, py: int) -> bool:
        return self.x <= px < self.x + self.width and self.y <= py < self.y + self.height


class MockComputer:
    """A deterministic, dependency-free GUI.

    Not a simulation of any real application — a small widget world with
    exactly the affordances a GUI agent has to handle: clicking, focusing,
    typing, toggling, scrolling, and a submit that commits state. Because
    `state()` is ground truth, a verifier can be a plain equality check instead
    of an LLM judge, which keeps the reward signal honest while you build the
    rest of the pipeline.
    """

    def __init__(
        self,
        widgets: Sequence[Widget],
        *,
        width: int = 1280,
        height: int = 720,
        title: str = "SETTINGS",
        content_height: int | None = None,
    ) -> None:
        self.width = width
        self.height = height
        self.title = title
        self._initial = [Widget(**vars(w)) for w in widgets]
        self._widgets: list[Widget] = []
        self._flags: dict[str, bool] = {}
        self._focus: str | None = None
        self._cursor: tuple[int, int] = (width // 2, height // 2)
        self._scroll_y = 0
        self._content_height = content_height or height
        self.action_log: list[str] = []
        self._apply_initial()

    # ---- scenarios -------------------------------------------------------- #

    @classmethod
    def settings_form(cls, **kwargs: Any) -> MockComputer:
        """The default scenario: a settings dialog behind one scroll.

        The SAVE button sits below the fold, so a competent agent has to
        scroll to reach it — which makes "did the agent verify before acting"
        an observable behavior rather than a guess.
        """
        widgets = [
            Widget("title", "label", 60, 110, 400, 30, label="ACCOUNT SETTINGS"),
            Widget("email_label", "label", 60, 170, 200, 22, label="EMAIL"),
            Widget("email", "field", 60, 196, 460, 44, placeholder="NAME@EXAMPLE.COM"),
            Widget("retries_label", "label", 60, 262, 200, 22, label="MAX RETRIES"),
            Widget("retries", "field", 60, 288, 200, 44, placeholder="3"),
            Widget("notify", "checkbox", 60, 366, 32, 32, label="EMAIL ME ON FAILURE"),
            Widget("theme_label", "label", 60, 440, 300, 22, label="THEME: SYSTEM DEFAULT"),
            Widget("cancel", "button", 60, 780, 150, 48, label="CANCEL"),
            Widget("save", "button", 230, 780, 150, 48, label="SAVE", sets="saved"),
        ]
        kwargs.setdefault("content_height", 900)
        return cls(widgets, **kwargs)

    def _apply_initial(self) -> None:
        self._widgets = [Widget(**vars(w)) for w in self._initial]
        self._flags = {w.sets: False for w in self._widgets if w.sets}
        self._focus = None
        self._scroll_y = 0
        self._cursor = (self.width // 2, self.height // 2)
        self.action_log = []

    # ---- ComputerEnvironment --------------------------------------------- #

    async def reset(self) -> Screenshot:
        self._apply_initial()
        return await self.screenshot()

    async def screenshot(self) -> Screenshot:
        return self._render()

    async def execute(self, action: Action) -> Screenshot:
        validate(action, width=self.width, height=self.height)
        self.action_log.append(action.describe())
        handler = getattr(self, f"_do_{action.kind.value}", None)
        if handler is None:
            raise ActionError(f"{action.kind} is not supported by MockComputer")
        handler(action)
        return self._render()

    def state(self) -> Mapping[str, Any]:
        """Ground truth: every field value, every toggle, every flag."""
        snapshot: dict[str, Any] = {
            "focus": self._focus,
            "scroll_y": self._scroll_y,
            "steps": len(self.action_log),
        }
        for widget in self._widgets:
            if widget.kind == "field":
                snapshot[widget.id] = widget.value
            elif widget.kind == "checkbox":
                snapshot[widget.id] = widget.checked
        snapshot.update(self._flags)
        return snapshot

    async def close(self) -> None:
        return None

    # ---- action handlers -------------------------------------------------- #
    # One method per ActionKind value; `execute` dispatches by name.

    def _do_screenshot(self, action: Action) -> None:
        return None

    def _do_cursor_position(self, action: Action) -> None:
        return None

    def _do_wait(self, action: Action) -> None:
        return None

    def _do_mouse_move(self, action: Action) -> None:
        self._cursor = action.coordinate  # type: ignore[assignment]

    def _do_left_click(self, action: Action) -> None:
        self._cursor = action.coordinate  # type: ignore[assignment]
        widget = self._hit_test(*self._cursor)
        if widget is None:
            self._focus = None
            return
        if widget.kind == "field":
            self._focus = widget.id
        elif widget.kind == "checkbox":
            widget.checked = not widget.checked
            self._focus = widget.id
        elif widget.kind == "button":
            self._focus = widget.id
            if widget.sets:
                self._flags[widget.sets] = True

    _do_double_click = _do_left_click
    _do_triple_click = _do_left_click
    _do_right_click = _do_left_click
    _do_middle_click = _do_left_click
    _do_left_mouse_down = _do_mouse_move
    _do_left_mouse_up = _do_left_click

    def _do_left_click_drag(self, action: Action) -> None:
        self._cursor = action.coordinate  # type: ignore[assignment]

    def _do_scroll(self, action: Action) -> None:
        step = 60 * int(action.scroll_amount or 1)
        delta = step if action.scroll_direction == "down" else -step
        if action.scroll_direction in ("left", "right"):
            delta = 0
        limit = max(0, self._content_height - self.height)
        self._scroll_y = max(0, min(limit, self._scroll_y + delta))

    def _do_type(self, action: Action) -> None:
        target = self._focused_field()
        if target is not None:
            target.value += action.text or ""

    def _do_key(self, action: Action) -> None:
        key = (action.text or "").lower()
        if key in ("return", "enter", "kp_enter"):
            submit = next((w for w in self._widgets if w.sets == "saved"), None)
            if submit is not None:
                self._flags["saved"] = True
        elif key == "tab":
            self._focus_next()
        elif key in ("backspace", "ctrl+h"):
            target = self._focused_field()
            if target is not None and target.value:
                target.value = target.value[:-1]
        elif key in ("ctrl+a", "cmd+a"):
            target = self._focused_field()
            if target is not None:
                target.value = ""

    def _do_hold_key(self, action: Action) -> None:
        self._do_key(action)

    # ---- internals -------------------------------------------------------- #

    def _hit_test(self, x: int, y: int) -> Widget | None:
        page_y = y + self._scroll_y
        for widget in self._widgets:
            if widget.kind == "label":
                continue
            if widget.contains(x, page_y):
                return widget
        return None

    def _focused_field(self) -> Widget | None:
        if self._focus is None:
            return None
        widget = next((w for w in self._widgets if w.id == self._focus), None)
        return widget if widget is not None and widget.kind == "field" else None

    def _focus_next(self) -> None:
        focusable = [w for w in self._widgets if w.kind in ("field", "checkbox", "button")]
        if not focusable:
            return
        ids = [w.id for w in focusable]
        idx = ids.index(self._focus) + 1 if self._focus in ids else 0
        self._focus = ids[idx % len(ids)]

    def _render(self) -> Screenshot:
        canvas = Canvas(self.width, self.height, PALETTE["bg"])

        # Window chrome.
        canvas.fill_rect(0, 0, self.width, 56, PALETTE["chrome"])
        canvas.text(24, 20, self.title, PALETTE["chrome_text"], scale=3)
        if max(0, self._content_height - self.height) > 0:
            hint = f"SCROLL {self._scroll_y}/{self._content_height - self.height}"
            canvas.text(self.width - text_width(hint, 2) - 24, 24, hint, PALETTE["muted"], 2)

        # Content panel.
        canvas.fill_rect(24, 80, self.width - 48, self.height - 104, PALETTE["panel"])
        canvas.stroke_rect(24, 80, self.width - 48, self.height - 104, PALETTE["border"])

        for widget in self._widgets:
            self._draw_widget(canvas, widget)

        # Pointer, so a frame carries where the last action landed.
        cx, cy = self._cursor
        canvas.fill_rect(cx - 6, cy - 1, 13, 3, PALETTE["cursor"])
        canvas.fill_rect(cx - 1, cy - 6, 3, 13, PALETTE["cursor"])

        return Screenshot(
            data=canvas.to_png(),
            width=self.width,
            height=self.height,
            metadata={"scroll_y": self._scroll_y, "focus": self._focus},
        )

    def _draw_widget(self, canvas: Canvas, widget: Widget) -> None:
        y = widget.y - self._scroll_y
        if y + widget.height < 70 or y > self.height:
            return  # scrolled out of view

        focused = widget.id == self._focus
        border = PALETTE["border_focus"] if focused else PALETTE["border"]

        if widget.kind == "label":
            canvas.text(widget.x, y, widget.label, PALETTE["text"], scale=3)

        elif widget.kind == "field":
            canvas.fill_rect(widget.x, y, widget.width, widget.height, PALETTE["panel"])
            canvas.stroke_rect(widget.x, y, widget.width, widget.height, border, 2 if focused else 1)
            shown = widget.value or widget.placeholder
            color = PALETTE["text"] if widget.value else PALETTE["muted"]
            canvas.text(widget.x + 12, y + 14, shown[:34], color, scale=3)

        elif widget.kind == "checkbox":
            canvas.stroke_rect(widget.x, y, widget.width, widget.height, border, 2)
            if widget.checked:
                canvas.fill_rect(widget.x + 7, y + 7, widget.width - 14, widget.height - 14, PALETTE["ok"])
            canvas.text(widget.x + widget.width + 16, y + 8, widget.label, PALETTE["text"], scale=3)

        elif widget.kind == "button":
            primary = widget.sets is not None
            fill = PALETTE["accent"] if primary else PALETTE["panel"]
            canvas.fill_rect(widget.x, y, widget.width, widget.height, fill)
            canvas.stroke_rect(widget.x, y, widget.width, widget.height, border, 2 if focused else 1)
            label_color = PALETTE["accent_text"] if primary else PALETTE["text"]
            offset = max(8, (widget.width - text_width(widget.label, 3)) // 2)
            canvas.text(widget.x + offset, y + 16, widget.label, label_color, scale=3)


# --------------------------------------------------------------------------- #
# Browser environment
# --------------------------------------------------------------------------- #


@dataclass
class PlaywrightComputer:
    """A real browser, driven through Playwright.

    Playwright is an optional dependency (`pip install "agentic-post-training[computer-use]"`
    then `playwright install chromium`). Import and launch are lazy so the rest
    of the package stays importable without it.

    The action mapping is intentionally thin — Playwright's mouse and keyboard
    APIs already take viewport pixel coordinates, which is the same space the
    model reads off the screenshot.
    """

    start_url: str = "about:blank"
    width: int = 1280
    height: int = 720
    headless: bool = True
    _playwright: Any = field(default=None, init=False, repr=False)
    _browser: Any = field(default=None, init=False, repr=False)
    _page: Any = field(default=None, init=False, repr=False)

    async def _ensure_page(self) -> Any:
        if self._page is not None:
            return self._page
        try:
            from playwright.async_api import async_playwright
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "PlaywrightComputer needs Playwright. Install with "
                '`pip install "agentic-post-training[computer-use]"` then '
                "`playwright install chromium`."
            ) from exc

        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=self.headless)
        self._page = await self._browser.new_page(
            viewport={"width": self.width, "height": self.height}
        )
        await self._page.goto(self.start_url)
        return self._page

    async def reset(self) -> Screenshot:
        page = await self._ensure_page()
        await page.goto(self.start_url)
        return await self.screenshot()

    async def screenshot(self) -> Screenshot:
        page = await self._ensure_page()
        data = await page.screenshot(type="png")
        return Screenshot(data=data, width=self.width, height=self.height)

    async def execute(self, action: Action) -> Screenshot:
        validate(action, width=self.width, height=self.height)
        page = await self._ensure_page()
        kind = action.kind

        if kind is ActionKind.SCREENSHOT:
            pass
        elif kind is ActionKind.MOUSE_MOVE:
            await page.mouse.move(*action.coordinate)  # type: ignore[misc]
        elif kind in (ActionKind.LEFT_CLICK, ActionKind.RIGHT_CLICK, ActionKind.MIDDLE_CLICK):
            button = {"left_click": "left", "right_click": "right", "middle_click": "middle"}[kind.value]
            await page.mouse.click(*action.coordinate, button=button)  # type: ignore[misc]
        elif kind is ActionKind.DOUBLE_CLICK:
            await page.mouse.dblclick(*action.coordinate)  # type: ignore[misc]
        elif kind is ActionKind.TRIPLE_CLICK:
            await page.mouse.click(*action.coordinate, click_count=3)  # type: ignore[misc]
        elif kind is ActionKind.LEFT_MOUSE_DOWN:
            await page.mouse.move(*action.coordinate)  # type: ignore[misc]
            await page.mouse.down()
        elif kind is ActionKind.LEFT_MOUSE_UP:
            await page.mouse.move(*action.coordinate)  # type: ignore[misc]
            await page.mouse.up()
        elif kind is ActionKind.LEFT_CLICK_DRAG:
            await page.mouse.move(*action.start_coordinate)  # type: ignore[misc]
            await page.mouse.down()
            await page.mouse.move(*action.coordinate)  # type: ignore[misc]
            await page.mouse.up()
        elif kind is ActionKind.SCROLL:
            await page.mouse.move(*action.coordinate)  # type: ignore[misc]
            step = 100 * int(action.scroll_amount or 1)
            deltas = {
                "down": (0, step), "up": (0, -step),
                "right": (step, 0), "left": (-step, 0),
            }
            await page.mouse.wheel(*deltas[action.scroll_direction])  # type: ignore[index]
        elif kind is ActionKind.TYPE:
            await page.keyboard.type(action.text or "")
        elif kind in (ActionKind.KEY, ActionKind.HOLD_KEY):
            await page.keyboard.press(_to_playwright_key(action.text or ""))
        elif kind is ActionKind.WAIT:
            await asyncio.sleep(min(float(action.duration or 0), 10.0))
        elif kind is ActionKind.CURSOR_POSITION:
            pass
        else:  # pragma: no cover - ActionKind is exhaustive above
            raise ActionError(f"{kind} is not supported by PlaywrightComputer")

        return await self.screenshot()

    def state(self) -> Mapping[str, Any]:
        if self._page is None:
            return {}
        return {"url": self._page.url, "title": ""}

    async def close(self) -> None:
        if self._browser is not None:
            await self._browser.close()
            self._browser = None
            self._page = None
        if self._playwright is not None:
            await self._playwright.stop()
            self._playwright = None


#: The model emits X11-style key names; Playwright wants its own spelling.
_KEY_ALIASES = {
    "return": "Enter",
    "kp_enter": "Enter",
    "escape": "Escape",
    "backspace": "Backspace",
    "tab": "Tab",
    "space": " ",
    "page_down": "PageDown",
    "page_up": "PageUp",
}


def _to_playwright_key(key: str) -> str:
    parts = key.replace("cmd", "Meta").split("+")
    mapped = [_KEY_ALIASES.get(p.lower(), p.capitalize() if len(p) > 1 else p) for p in parts]
    return "+".join(mapped)
