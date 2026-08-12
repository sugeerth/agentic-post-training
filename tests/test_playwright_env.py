"""Tests for the browser environment.

Split in two deliberately:

  **Key translation** runs everywhere, with no browser. It is pure string
  mapping, and it is where the expensive bug lives — a modifier spelled wrong
  makes `ctrl+a` / `ctrl+c` fail on a real page while every mock test stays
  green, because the mock never sees Playwright's spelling.

  **The browser integration** runs only where Playwright and a Chromium build
  are actually available, and skips cleanly otherwise. CI stays
  dependency-light; anyone with a browser gets the real coverage.

Point `PLAYWRIGHT_CHROMIUM_EXECUTABLE` at an existing binary to run these
against an image that ships its own Chromium instead of downloading one.
"""

import asyncio
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from computer_use import (
    Action,
    ActionKind,
    PlaywrightComputer,
    ScriptedPolicy,
    StateVerifier,
    run_episode,
)
from computer_use.environments import _to_playwright_key

PAGE = """<!doctype html><meta charset="utf-8"><title>Settings</title>
<style>body{font:16px system-ui;margin:40px;height:1400px}
input[type=text]{width:400px;padding:12px;font-size:16px}
button{padding:12px 28px;font-size:16px}
#save{position:absolute;top:1000px}</style>
<h1>Account Settings</h1>
<label>Email</label><br><input type="text" id="email" placeholder="name@example.com"><br><br>
<label><input type="checkbox" id="notify"> Email me on failure</label>
<button id="save">Save</button>
<script>document.getElementById('save').onclick=()=>{window.__saved=true}</script>
"""

STATE_SCRIPT = """() => ({
  email: document.getElementById('email').value,
  notify: document.getElementById('notify').checked,
  saved: window.__saved === true,
})"""


def _browser_available() -> bool:
    try:
        import playwright  # noqa: F401
    except ImportError:
        return False
    executable = os.environ.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE")
    return executable is None or Path(executable).exists()


requires_browser = unittest.skipUnless(
    _browser_available(), "Playwright and a Chromium build are required"
)


class TestKeyTranslation(unittest.TestCase):
    """The computer-use tool speaks X11; Playwright does not."""

    def test_modifiers_use_playwright_names(self):
        # Regression test: title-casing produced "Ctrl", which Playwright
        # rejects outright — select-all, copy, and paste all failed on a real
        # page while every mock test passed.
        self.assertEqual(_to_playwright_key("ctrl+a"), "Control+a")
        self.assertEqual(_to_playwright_key("cmd+c"), "Meta+c")
        self.assertEqual(_to_playwright_key("super+l"), "Meta+l")
        self.assertEqual(_to_playwright_key("shift+Tab"), "Shift+Tab")
        self.assertEqual(_to_playwright_key("alt+F4"), "Alt+F4")

    def test_x11_names_are_translated(self):
        self.assertEqual(_to_playwright_key("Return"), "Enter")
        self.assertEqual(_to_playwright_key("BackSpace"), "Backspace")
        self.assertEqual(_to_playwright_key("Page_Down"), "PageDown")
        self.assertEqual(_to_playwright_key("Down"), "ArrowDown")
        self.assertEqual(_to_playwright_key("Escape"), "Escape")

    def test_single_characters_keep_their_case(self):
        # Playwright wants a bare lowercase letter, not "A".
        self.assertEqual(_to_playwright_key("a"), "a")
        self.assertEqual(_to_playwright_key("ctrl+shift+p"), "Control+Shift+p")

    def test_unknown_multichar_keys_are_title_cased(self):
        self.assertEqual(_to_playwright_key("f5"), "F5")


@requires_browser
class TestBrowserEnvironment(unittest.TestCase):
    """The real thing: a real Chromium, a real page, real pixels."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        page = Path(cls._tmp.name) / "page.html"
        page.write_text(PAGE, encoding="utf-8")
        cls.url = f"file://{page}"

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _env(self):
        return PlaywrightComputer(start_url=self.url, state_script=STATE_SCRIPT)

    def test_screenshot_is_a_real_png_at_viewport_size(self):
        async def run():
            env = self._env()
            try:
                frame = await env.reset()
                return frame
            finally:
                await env.close()

        frame = asyncio.run(run())
        self.assertTrue(frame.data.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertEqual((frame.width, frame.height), (1280, 720))

    def test_state_script_exposes_ground_truth(self):
        async def run():
            env = self._env()
            try:
                await env.reset()
                before = dict(env.state())
                box = await env._page.evaluate(
                    "() => {const r=document.getElementById('email')"
                    ".getBoundingClientRect(); "
                    "return [Math.round(r.x+r.width/2), Math.round(r.y+r.height/2)]}"
                )
                await env.execute(Action(ActionKind.LEFT_CLICK, coordinate=tuple(box)))
                await env.execute(Action(ActionKind.TYPE, text="ada@example.com"))
                return before, dict(env.state())
            finally:
                await env.close()

        before, after = asyncio.run(run())
        self.assertEqual(before["email"], "")
        self.assertEqual(after["email"], "ada@example.com")
        self.assertIn("url", after)

    def test_ctrl_a_backspace_clears_a_field(self):
        # The chord that the old mapping broke.
        async def run():
            env = self._env()
            try:
                await env.reset()
                box = await env._page.evaluate(
                    "() => {const r=document.getElementById('email')"
                    ".getBoundingClientRect(); "
                    "return [Math.round(r.x+r.width/2), Math.round(r.y+r.height/2)]}"
                )
                await env.execute(Action(ActionKind.LEFT_CLICK, coordinate=tuple(box)))
                await env.execute(Action(ActionKind.TYPE, text="wrong@example.com"))
                await env.execute(Action(ActionKind.KEY, text="ctrl+a"))
                await env.execute(Action(ActionKind.KEY, text="BackSpace"))
                return dict(env.state())["email"]
            finally:
                await env.close()

        self.assertEqual(asyncio.run(run()), "")

    def test_every_action_kind_executes(self):
        """No action in the space may blow up against a real browser."""
        async def run():
            env = self._env()
            try:
                await env.reset()
                point = (400, 300)
                actions = [
                    Action(ActionKind.SCREENSHOT),
                    Action(ActionKind.CURSOR_POSITION),
                    Action(ActionKind.MOUSE_MOVE, coordinate=point),
                    Action(ActionKind.LEFT_CLICK, coordinate=point),
                    Action(ActionKind.RIGHT_CLICK, coordinate=point),
                    Action(ActionKind.MIDDLE_CLICK, coordinate=point),
                    Action(ActionKind.DOUBLE_CLICK, coordinate=point),
                    Action(ActionKind.TRIPLE_CLICK, coordinate=point),
                    Action(ActionKind.LEFT_MOUSE_DOWN, coordinate=point),
                    Action(ActionKind.LEFT_MOUSE_UP, coordinate=(410, 310)),
                    Action(ActionKind.LEFT_CLICK_DRAG,
                           start_coordinate=(100, 100), coordinate=(300, 300)),
                    Action(ActionKind.SCROLL, coordinate=point,
                           scroll_direction="down", scroll_amount=2),
                    Action(ActionKind.SCROLL, coordinate=point,
                           scroll_direction="up", scroll_amount=1),
                    Action(ActionKind.TYPE, text="hello"),
                    Action(ActionKind.KEY, text="Page_Down"),
                    Action(ActionKind.HOLD_KEY, text="Escape"),
                    Action(ActionKind.WAIT, duration=0.05),
                ]
                for action in actions:
                    await env.execute(action)
                return len(actions)
            finally:
                await env.close()

        self.assertEqual(asyncio.run(run()), 17)

    def test_scroll_moves_the_viewport(self):
        async def run():
            env = self._env()
            try:
                await env.reset()
                await env.execute(Action(
                    ActionKind.SCROLL, coordinate=(640, 400),
                    scroll_direction="down", scroll_amount=6,
                ))
                return await env._page.evaluate("() => window.scrollY")
            finally:
                await env.close()

        self.assertGreater(asyncio.run(run()), 0)

    def test_full_episode_against_a_real_browser(self):
        """The whole pipeline — rollout, verifier, reward — on real Chromium.

        This is the test that says the framework works outside the mock: the
        same `run_episode` and `StateVerifier` used everywhere else, driving a
        real page and scoring against real DOM state.
        """
        async def run():
            env = PlaywrightComputer(start_url=self.url, state_script=STATE_SCRIPT)
            try:
                await env.reset()
                coords = await env._page.evaluate(
                    "() => Object.fromEntries(['email','notify'].map(id => {"
                    "const r = document.getElementById(id).getBoundingClientRect();"
                    "return [id, [Math.round(r.x+r.width/2), Math.round(r.y+r.height/2)]]}))"
                )
                scroll = Action(ActionKind.SCROLL, coordinate=(640, 400),
                                scroll_direction="down", scroll_amount=6)
                # SAVE starts below the fold, so its coordinate has to be read
                # *after* scrolling — exactly as an agent would, from the
                # screenshot that follows the scroll. `run_episode` re-navigates
                # on reset, so the page returns to the top and the episode
                # repeats this scroll before clicking.
                await env.execute(scroll)
                save = await _center(env, "save")

                trajectory = await run_episode(
                    "Set the email, enable notifications, and save.",
                    env,
                    ScriptedPolicy([
                        Action(ActionKind.LEFT_CLICK, coordinate=tuple(coords["email"])),
                        Action(ActionKind.TYPE, text="ada@example.com"),
                        Action(ActionKind.LEFT_CLICK, coordinate=tuple(coords["notify"])),
                        scroll,
                        Action(ActionKind.LEFT_CLICK, coordinate=save),
                    ]),
                    verifier=StateVerifier({
                        "email": "ada@example.com", "notify": True, "saved": True,
                    }),
                )
                return trajectory
            finally:
                await env.close()

        trajectory = asyncio.run(run())
        self.assertTrue(
            trajectory.succeeded,
            f"browser episode failed: {trajectory.metadata['verdict']['reason']}",
        )
        self.assertEqual(trajectory.num_steps, 5)
        self.assertGreater(trajectory.reward, 0.5)


async def _center(env, element_id):
    """Viewport-space center of an element, as the agent would read it."""
    point = await env._page.evaluate(
        f"() => {{const r=document.getElementById('{element_id}')"
        ".getBoundingClientRect(); "
        "return [Math.round(r.x+r.width/2), Math.round(r.y+r.height/2)]}"
    )
    return tuple(point)


if __name__ == "__main__":
    unittest.main()
