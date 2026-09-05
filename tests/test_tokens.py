"""Tests for encoding interactions as tokens.

A lossy encoding of actions is only safe if the actions still *work* after the
round trip, so the load-bearing test is not that coordinates come back close —
it is `test_a_tokenized_episode_still_solves_its_task`, which decodes an
episode from its ids, replays it, and checks the verifier still passes.

That test is what caught the bug this module shipped with. Text was folded to
uppercase on the theory that the renderer's font defines the alphabet; it does
define what the environment can *draw*, but a field stores what it was *typed*,
and `ada@example.com` came back as `ADA@EXAMPLE.COM`. Every coordinate in those
episodes was still correct, so nothing else in the suite would have noticed.
"""

import asyncio
import os
import string
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from computer_use import (
    Action,
    ActionKind,
    MockComputer,
    RolloutConfig,
    ScriptedPolicy,
    run_episode,
)
from computer_use.tasks import SUITE
from computer_use.tokens import (
    MAX_MARKS,
    PAD,
    VOCAB,
    X_BINS,
    Y_BINS,
    Vocabulary,
    cost,
    decode_action,
    decode_actions,
    dequantize,
    encode_action,
    encode_screen,
    encode_trajectory,
    quantize,
    to_token_batch,
)
from computer_use.worlds import curriculum, generate

CONFIG = RolloutConfig(store_frames=True, max_steps=24)


def episode(task, policy):
    return asyncio.run(run_episode(
        task.instruction, task.env_factory(), policy,
        verifier=task.verifier, reward_config=task.reward_config(), config=CONFIG,
    ))


class TestVocabulary(unittest.TestCase):
    def test_ids_are_stable_and_unique(self):
        self.assertEqual(VOCAB.size, len(set(VOCAB.tokens)))
        self.assertEqual(Vocabulary().tokens, VOCAB.tokens)

    def test_padding_is_zero(self):
        # A zero-filled matrix should read as a batch of empty rows, which is
        # only true if PAD is id 0.
        self.assertEqual(VOCAB.id(PAD), 0)

    def test_it_stays_small(self):
        # The whole point is a compact alphabet. If this grows unnoticed the
        # encoding has stopped being cheaper than spelling JSON.
        self.assertLess(VOCAB.size, 400)

    def test_lowercase_is_representable(self):
        # Not decorative: field values are compared verbatim by verifiers.
        for char in string.ascii_lowercase:
            self.assertNotEqual(VOCAB.id(f"<c:{char}>"), VOCAB.id("<c:?unk>"))


class TestCoordinates(unittest.TestCase):
    def test_a_cell_maps_back_inside_itself(self):
        for x in range(0, 1280, 37):
            for y in range(0, 720, 29):
                bx, by = quantize(x, y)
                rx, ry = dequantize(bx, by)
                self.assertEqual((bx, by), quantize(rx, ry))

    def test_out_of_frame_coordinates_stay_in_vocabulary(self):
        for point in ((-50, -50), (5000, 5000)):
            bx, by = quantize(*point)
            self.assertTrue(0 <= bx < X_BINS)
            self.assertTrue(0 <= by < Y_BINS)

    def test_every_control_survives_the_round_trip(self):
        """The grid is chosen by this measurement, not by taste.

        A coarser one is a smaller vocabulary and a corrupted dataset: at
        32x18, eight percent of controls decode to a click that lands on a
        different control, and nothing downstream would report it.
        """
        environments = [generate(s, hard=h).build() for s in range(12) for h in (False, True)]
        environments += [MockComputer.settings_form(), MockComputer.checkout_flow()]
        checked = 0
        for env in environments:
            asyncio.run(env.reset())
            for widget in env._visible:
                if widget.kind not in ("field", "checkbox", "radio", "button"):
                    continue
                px = widget.x + widget.width // 2
                py = widget.y - env._scroll_y + widget.height // 2
                if not 0 <= py < 720:
                    continue
                intended = env._hit_test(px, py)
                if intended is None:
                    continue
                landed = env._hit_test(*dequantize(*quantize(px, py)))
                self.assertIsNotNone(landed, f"{widget.id} decoded onto nothing")
                self.assertEqual(
                    landed.id, intended.id,
                    f"{widget.id} decoded onto {landed.id}",
                )
                checked += 1
        self.assertGreater(checked, 50)


class TestActions(unittest.TestCase):
    def test_every_action_kind_round_trips(self):
        samples = [
            Action(ActionKind.SCREENSHOT),
            Action(ActionKind.LEFT_CLICK, coordinate=(290, 218)),
            Action(ActionKind.DOUBLE_CLICK, coordinate=(10, 10)),
            Action(ActionKind.TYPE, text="ada@example.com"),
            Action(ActionKind.KEY, text="ctrl+a"),
            Action(ActionKind.SCROLL, coordinate=(640, 400),
                   scroll_direction="down", scroll_amount=3),
            Action(ActionKind.LEFT_CLICK_DRAG, coordinate=(400, 300),
                   start_coordinate=(100, 100)),
        ]
        for action in samples:
            restored = decode_action(encode_action(action))
            self.assertIsNotNone(restored)
            self.assertIs(restored.kind, action.kind)
            self.assertEqual(restored.text, action.text)
            self.assertEqual(restored.scroll_direction, action.scroll_direction)
            self.assertEqual(restored.scroll_amount, action.scroll_amount)

    def test_typed_text_is_not_case_folded(self):
        # The bug this module shipped with. Uppercasing is invisible to a
        # coordinate check and fatal to a value check.
        action = Action(ActionKind.TYPE, text="ada.alt@example.com")
        self.assertEqual(decode_action(encode_action(action)).text, action.text)

    def test_a_drag_keeps_both_ends(self):
        action = Action(ActionKind.LEFT_CLICK_DRAG,
                        coordinate=(400, 300), start_coordinate=(100, 100))
        restored = decode_action(encode_action(action))
        self.assertIsNotNone(restored.start_coordinate)
        self.assertNotEqual(restored.start_coordinate, restored.coordinate)

    def test_tokens_that_do_not_start_with_a_kind_decode_to_nothing(self):
        self.assertIsNone(decode_action(["<x:3>", "<y:4>"]))
        self.assertIsNone(decode_action([]))

    def test_a_click_costs_three_tokens(self):
        self.assertEqual(
            len(encode_action(Action(ActionKind.LEFT_CLICK, coordinate=(1, 1)))), 3,
        )


class TestVocabularyStability(unittest.TestCase):
    """New symbols must not renumber the old ones.

    A checkpoint addresses its embedding rows by token id. Inserting a block
    of symbols into the middle of the vocabulary shifts every id after it and
    silently repoints a trained model's rows at symbols it never saw — nothing
    raises, the model just becomes a different model. This happened once, when
    the mark tokens were first added between the scroll amounts and the
    characters, and it invalidated two trained checkpoints without a single
    test failing.
    """

    def test_marks_are_appended_not_inserted(self):
        tokens = list(VOCAB.tokens)
        marks = [t for t in tokens if t.startswith("<m:")]

        self.assertEqual(tokens[-len(marks):], marks)

    def test_every_non_mark_token_precedes_every_mark(self):
        """The property that keeps pre-mark ids valid, stated directly."""
        tokens = list(VOCAB.tokens)
        first_mark = min(i for i, t in enumerate(tokens) if t.startswith("<m:"))

        self.assertTrue(
            all(not t.startswith("<m:") for t in tokens[:first_mark])
        )
        self.assertEqual(first_mark, len(tokens) - MAX_MARKS)

    def test_the_character_block_sits_where_it_always_did(self):
        """A canary on the largest block, which is what shifted last time."""
        self.assertLess(VOCAB.id("<c:A>"), VOCAB.id("<m:0>"))
        self.assertLess(VOCAB.id("<c:?unk>"), VOCAB.id("<m:0>"))


class TestScreenFieldOrder(unittest.TestCase):
    """The two element encodings, which differ in exactly one thing.

    `label_first` exists to test a claim about attention — that copying an
    answer that *follows* its cue is easier than copying one that precedes it.
    The claim is only testable if the two encodings are otherwise identical, so
    that is what these check: same controls, same tokens, different order.
    """

    def _screen(self):
        from computer_use.perception import Box, Element

        class Screen:
            width, height = 1280, 720
            elements = (
                Element(
                    label="SAVE", kind="button",
                    box=Box(x=100, y=200, width=80, height=40, color=(20, 20, 20)),
                    text_run=None, click=(140, 220),
                ),
            )

        return Screen()

    def test_the_default_puts_the_coordinate_before_the_label(self):
        tokens = encode_screen(self._screen())

        self.assertLess(tokens.index("<x:7>"), tokens.index("<c:S>"))

    def test_label_first_puts_the_label_before_the_coordinate(self):
        tokens = encode_screen(self._screen(), label_first=True)

        self.assertLess(tokens.index("<c:S>"), tokens.index("<x:7>"))

    def test_the_two_orders_carry_exactly_the_same_tokens(self):
        """Any difference beyond order would confound the comparison."""
        default = encode_screen(self._screen())
        flipped = encode_screen(self._screen(), label_first=True)

        self.assertEqual(sorted(default), sorted(flipped))
        self.assertNotEqual(default, flipped)

    def test_state_stays_beside_the_coordinate_in_both_orders(self):
        """A toggle's state qualifies the control, not its name."""
        from computer_use.perception import Box, Element

        class Screen:
            width, height = 1280, 720
            elements = (
                Element(
                    label="ON", kind="toggle",
                    box=Box(x=100, y=200, width=80, height=40, color=(20, 20, 20)),
                    text_run=None, click=(140, 220), checked=True,
                ),
            )

        flipped = encode_screen(Screen(), label_first=True)

        self.assertEqual(flipped.index("<s:on>"), flipped.index("<y:11>") + 1)


class TestTrajectories(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tasks = list(SUITE)[:4] + curriculum(range(6), per_world=2, hard=True)
        cls.gold = [episode(t, ScriptedPolicy(list(t.gold))) for t in cls.tasks]

    def test_a_tokenized_episode_still_solves_its_task(self):
        """Encode, throw the trajectory away, replay from the ids alone.

        This is the only check that means anything for a lossy encoding: not
        that the numbers come back similar, but that the actions still do what
        they did.
        """
        for task, trajectory in zip(self.tasks, self.gold, strict=True):
            self.assertTrue(trajectory.succeeded, f"{task.name} gold failed")
            actions = decode_actions(encode_trajectory(trajectory))
            replay = episode(task, ScriptedPolicy(actions))
            self.assertTrue(
                replay.succeeded,
                f"{task.name} broke when replayed from tokens: "
                f"{replay.metadata['verdict']['reason']}",
            )

    def test_it_costs_less_than_spelling_the_json(self):
        totals = [cost(t) for t in self.gold]
        interaction = sum(t["interaction_tokens"] for t in totals)
        estimate = sum(t["json_tokens_estimate"] for t in totals)
        self.assertLess(interaction, estimate)

    def test_screens_can_be_encoded_alongside_the_actions(self):
        plain = encode_trajectory(self.gold[0])
        with_screens = encode_trajectory(self.gold[0], include_screens=True)
        self.assertGreater(len(with_screens), len(plain))
        self.assertEqual(decode_actions(with_screens), decode_actions(plain))


class TestBatching(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tasks = curriculum(range(6), per_world=2)
        cls.gold = [episode(t, ScriptedPolicy(list(t.gold))) for t in tasks]

    def test_rows_are_padded_to_one_width(self):
        batch = to_token_batch(self.gold)
        widths = {len(row) for row in batch.input_ids}
        self.assertEqual(len(widths), 1)
        self.assertEqual(batch.shape, (len(self.gold), widths.pop()))

    def test_the_mask_marks_exactly_the_real_tokens(self):
        batch = to_token_batch(self.gold)
        pad = VOCAB.id(PAD)
        for ids, mask in zip(batch.input_ids, batch.attention_mask, strict=True):
            real = sum(mask)
            self.assertTrue(all(m == 1 for m in mask[:real]))
            self.assertTrue(all(i == pad for i in ids[real:]))

    def test_truncation_keeps_the_instruction(self):
        # A row that lost its task is not a shorter example, it is a
        # mislabelled one, so truncation has to take from the tail.
        batch = to_token_batch(self.gold, max_length=12)
        for ids in batch.input_ids:
            self.assertEqual(ids[0], VOCAB.id("<bos>"))

    def test_rewards_ride_along(self):
        batch = to_token_batch(self.gold)
        self.assertEqual(len(batch.rewards), len(self.gold))

    def test_ids_are_inside_the_vocabulary(self):
        batch = to_token_batch(self.gold, include_screens=True)
        for ids in batch.input_ids:
            for token_id in ids:
                self.assertTrue(0 <= token_id < batch.vocab_size)


class TestTorchAdapter(unittest.TestCase):
    def test_tensors_when_torch_is_installed_and_a_clear_error_when_not(self):
        tasks = curriculum(range(2), per_world=1)
        batch = to_token_batch([episode(t, ScriptedPolicy(list(t.gold))) for t in tasks])
        try:
            import torch  # noqa: F401
        except ImportError:
            with self.assertRaises(RuntimeError) as caught:
                batch.to_torch()
            self.assertIn("PyTorch", str(caught.exception))
            return
        tensors = batch.to_torch()
        self.assertEqual(tuple(tensors["input_ids"].shape), batch.shape)
        self.assertEqual(tuple(tensors["attention_mask"].shape), batch.shape)


if __name__ == "__main__":
    unittest.main()
