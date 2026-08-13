"""Tests for reading a GUI back out of its pixels.

These are legibility tests as much as parser tests. If a label cannot be
recovered from the frame then the frame does not carry it, and the claim that
these screenshots are readable by a vision model is false — which would make
every benchmark number a measurement of the renderer rather than the agent.

The strong assertion is `test_every_control_is_located_typed_and_labelled`: for
every control the environment says is on screen, the parser must find a box
whose click point hit-tests back to that exact widget, with the right kind and
the right words. It runs across generated worlds at two scroll positions, so it
covers layouts nobody chose.
"""

import asyncio
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from computer_use import Action, ActionKind, MockComputer
from computer_use.perception import (
    DecodeError,
    boxes,
    decode_png,
    legibility,
    parse_screen,
    text_runs,
)
from computer_use.worlds import generate

#: Environment widget kind -> what the parser should call it from pixels alone.
KINDS = {"field": "field", "checkbox": "toggle", "radio": "toggle", "button": "button"}

SEEDS = range(10)


def frame(env):
    return asyncio.run(env.screenshot())


def on_screen(env):
    """The widgets actually drawn: `_visible` includes ones below the fold."""
    out = []
    for widget in env._visible:
        y = widget.y - env._scroll_y
        if widget.kind in KINDS and not (y + widget.height < 70 or y > env.height):
            out.append(widget)
    return out


def scrolled(env, amount=3):
    asyncio.run(env.execute(Action(
        ActionKind.SCROLL, coordinate=(640, 400),
        scroll_direction="down", scroll_amount=amount,
    )))
    return env


class TestDecoder(unittest.TestCase):
    def test_round_trips_the_renderers_own_output(self):
        env = MockComputer.settings_form()
        shot = asyncio.run(env.reset())
        raster = decode_png(shot.data)
        self.assertEqual((raster.width, raster.height), (shot.width, shot.height))
        self.assertEqual(len(raster.pixels), shot.width * shot.height * 3)

    def test_pixels_match_what_was_drawn(self):
        env = MockComputer.settings_form()
        raster = decode_png(asyncio.run(env.reset()).data)
        # The window chrome is a solid band across the top; the panel is not.
        self.assertEqual(raster.at(5, 5), raster.at(1200, 40))
        self.assertNotEqual(raster.at(5, 5), raster.at(640, 400))

    def test_rejects_things_that_are_not_readable_pngs(self):
        with self.assertRaises(DecodeError):
            decode_png(b"not a png at all")
        with self.assertRaises(DecodeError):
            decode_png(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)

    def test_positions_finds_only_whole_pixels(self):
        env = MockComputer.settings_form()
        raster = decode_png(asyncio.run(env.reset()).data)
        chrome = raster.at(5, 5)
        for x, y in raster.positions(chrome):
            self.assertEqual(raster.at(x, y), chrome)
            break
        else:
            self.fail("no pixels of a color read straight out of the image")


class TestTextRecovery(unittest.TestCase):
    def test_reads_the_labels_off_a_form(self):
        env = MockComputer.settings_form()
        recovered = {run.text for run in text_runs(decode_png(asyncio.run(env.reset()).data))}
        for expected in ("ACCOUNT SETTINGS", "EMAIL", "MAX RETRIES", "EMAIL ME ON FAILURE"):
            self.assertIn(expected, recovered)

    def test_every_generated_label_is_legible(self):
        """A label that cannot be read back is noise on the screen."""
        for seed in SEEDS:
            world = generate(seed)
            env = world.build()
            asyncio.run(env.reset())
            expected = [
                w.label for w in on_screen(env) + [x for x in env._visible if x.kind == "label"]
                if w.label and w.y - env._scroll_y + w.height <= env.height
            ]
            report = legibility(frame(env).data, expected)
            self.assertEqual(report.missing, (), f"{world.name} lost {report.missing}")

    def test_a_value_typed_into_a_field_can_be_read_back(self):
        env = MockComputer.settings_form()
        asyncio.run(env.reset())
        asyncio.run(env.execute(Action(ActionKind.LEFT_CLICK, coordinate=(290, 218))))
        asyncio.run(env.execute(Action(ActionKind.TYPE, text="ada@example.com")))
        recovered = {run.text for run in text_runs(decode_png(frame(env).data))}
        self.assertIn("ADA@EXAMPLE.COM", recovered)


class TestControls(unittest.TestCase):
    def test_every_control_is_located_typed_and_labelled(self):
        environments = [
            ("settings", MockComputer.settings_form()),
            ("checkout", MockComputer.checkout_flow()),
            ("files", MockComputer.file_manager()),
        ]
        environments += [(f"world{s:03d}", generate(s).build()) for s in SEEDS]

        checked = 0
        for name, env in environments:
            asyncio.run(env.reset())
            for view in (env, scrolled(env)):
                screen = parse_screen(frame(view).data)
                for widget in on_screen(view):
                    match = next(
                        (
                            element for element in screen.elements
                            if element.box is not None
                            and (hit := view._hit_test(*element.click)) is not None
                            and hit.id == widget.id
                        ),
                        None,
                    )
                    self.assertIsNotNone(
                        match, f"{name}: nothing clickable found for {widget.id}",
                    )
                    self.assertEqual(
                        match.kind, KINDS[widget.kind],
                        f"{name}: {widget.id} read as {match.kind!r}",
                    )
                    if widget.label:
                        self.assertIn(
                            widget.label.upper(), match.label.upper(),
                            f"{name}: {widget.id} read as {match.label!r}",
                        )
                    checked += 1
        self.assertGreater(checked, 50, "the sweep stopped covering anything")

    def test_no_phantom_controls(self):
        """A box that hit-tests to nothing is a control the agent cannot use.

        Stacked radio buttons produce one: the bottom edge of one and the top
        edge of the next have the same width, so they pair into a rectangle
        spanning the gap — sitting exactly over the label of a real control.
        """
        for seed in SEEDS:
            env = generate(seed).build()
            asyncio.run(env.reset())
            screen = parse_screen(frame(env).data)
            for element in screen.elements:
                if element.box is None or element.box.width > 600:
                    continue  # the content panel itself is not a control
                self.assertIsNotNone(
                    env._hit_test(*element.click),
                    f"world{seed:03d}: phantom {element.kind} {element.label!r}",
                )

    def test_a_wide_button_is_not_read_as_a_field(self):
        # A 200px CONTINUE is exactly as wide as a short input. Width cannot
        # separate them; where the caption sits can, and getting it wrong
        # leaves the agent with no way off the screen.
        multi = next(generate(s) for s in range(40) if generate(s).spec.screens > 1)
        env = multi.build()
        asyncio.run(env.reset())
        screen = parse_screen(frame(env).data)
        nav = [e for e in screen.elements if e.label in ("CONTINUE", "NEXT")]
        self.assertTrue(nav, f"{multi.name}: no navigation control found")
        self.assertTrue(all(e.kind == "button" for e in nav))

    def test_the_primary_action_looks_different(self):
        # How a design says "this is the commit". The agent uses it to avoid
        # pressing submit while looking for a way to the next screen, so it has
        # to survive the trip through the pixels.
        single = next(
            generate(s) for s in range(40)
            # One screen and no scrolling, so the action is on the first frame.
            if generate(s).spec.screens == 1 and not generate(s).spec.scroll
        )
        env = single.build()
        asyncio.run(env.reset())
        screen = parse_screen(frame(env).data)
        buttons = [e for e in screen.elements if e.kind == "button"]
        self.assertTrue(buttons, f"{single.name}: no buttons parsed")
        self.assertTrue(
            any(e.filled for e in buttons),
            f"{single.name}: no primary action stands out",
        )

    def test_scroll_position_is_read_from_the_frame(self):
        env = MockComputer.settings_form()
        asyncio.run(env.reset())
        before = parse_screen(frame(env).data).scroll
        after = parse_screen(frame(scrolled(env)).data).scroll
        self.assertIsNotNone(before)
        self.assertIsNotNone(after)
        self.assertLess(before[0], after[0])
        self.assertEqual(before[1], after[1])

    def test_boxes_do_not_double_count_a_thick_border(self):
        env = MockComputer.settings_form()
        raster = decode_png(asyncio.run(env.reset()).data)
        found = boxes(raster)
        for i, a in enumerate(found):
            for b in found[i + 1:]:
                same = (
                    abs(a.x - b.x) <= 4 and abs(a.y - b.y) <= 4
                    and abs(a.width - b.width) <= 8 and abs(a.height - b.height) <= 8
                )
                self.assertFalse(same, f"{a} and {b} are the same control")


if __name__ == "__main__":
    unittest.main()
