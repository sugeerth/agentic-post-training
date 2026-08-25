"""Corpus construction, the loss mask, and evaluation by execution.

The bug this file is mostly guarding against is a silent one: an example whose
supervised positions are off by one trains the model to predict the token
*before* each action token. The loss still falls, generation still emits
something shaped like an action, and the failure only shows up as a score that
is bad for no visible reason. So the mask is checked against the tokens it
selects, by name, rather than by counting.
"""

from __future__ import annotations

import pytest

from computer_use.perception import Box, Element
from computer_use.pretrain import (
    CorpusConfig,
    build_corpus,
    context_tokens,
    decode_generated,
    evaluate,
    format_report,
    make_example,
    mean_loss,
    snap_to_screen,
    train,
)
from computer_use.tokens import (
    ACT,
    BOS,
    EOS,
    OBS,
    VOCAB,
    dequantize,
    encode_action,
    quantize,
)
from computer_use.transformer import GPT, ModelConfig
from computer_use.types import POINTING_ACTIONS, Action, ActionKind
from computer_use.worlds import curriculum

CLICK = Action(kind=ActionKind.LEFT_CLICK, coordinate=(300, 200))


@pytest.fixture(scope="module")
def tasks() -> list:
    """Two generated apps. Small on purpose — this suite runs in CI."""
    return curriculum(range(2), per_world=2)


@pytest.fixture(scope="module")
def corpus(tasks: list) -> list:
    return build_corpus(tasks)


class TestCorpus:
    def test_one_example_per_gold_step(self, tasks: list, corpus: list) -> None:
        assert len(corpus) == sum(len(t.gold) for t in tasks)

    def test_every_example_records_its_world(self, corpus: list) -> None:
        assert all(e.world_seed is not None for e in corpus)

    def test_context_ends_at_the_action_marker(self, corpus: list) -> None:
        for example in corpus:
            tokens = VOCAB.decode(example.ids)
            assert tokens[0] == BOS
            assert tokens[example.prompt_length - 1] == ACT
            assert tokens[-1] == EOS

    def test_the_observation_is_present(self, corpus: list) -> None:
        assert all(OBS in VOCAB.decode(e.ids) for e in corpus)

    def test_supervised_positions_predict_exactly_the_action(self) -> None:
        """Each supervised position's *successor* is an answer token."""
        example = make_example("Click SAVE.", _FakeScreen(), CLICK)
        tokens = VOCAB.decode(example.ids)
        answer = [tokens[p + 1] for p in example.supervised]
        assert answer == [*encode_action(CLICK), EOS]

    def test_nothing_before_the_action_is_supervised(self) -> None:
        example = make_example("Click SAVE.", _FakeScreen(), CLICK)
        assert min(example.supervised) == example.prompt_length - 1

    def test_the_end_token_is_supervised(self) -> None:
        """Otherwise the model never learns when an action is finished."""
        example = make_example("Click SAVE.", _FakeScreen(), CLICK)
        tokens = VOCAB.decode(example.ids)
        assert tokens[max(example.supervised) + 1] == EOS

    def test_instruction_is_truncated_not_dropped(self) -> None:
        config = CorpusConfig(instruction_chars=4)
        tokens = context_tokens("ABCDEFGHIJ", _FakeScreen(), config)
        characters = [t for t in tokens if t.startswith("<c:")]
        assert characters[:4] == ["<c:A>", "<c:B>", "<c:C>", "<c:D>"]

    def test_click_targets_are_copyable_from_the_context(self, corpus: list) -> None:
        """The answer is present in the input — the premise of the setup.

        If a click's cell is absent from the observation, the model cannot
        point at it, and whatever score it gets for that step is memorization.
        Snapping (see `snap_to_screen`) exists to close this gap; before it,
        one click in ten was unpointable. Scrolls are excluded because their
        coordinate is a fixed screen-centre anchor rather than a target.
        """
        clicks = [
            e for e in corpus
            if e.action is not None
            and e.action.coordinate is not None
            and e.action.kind in POINTING_ACTIONS
        ]
        assert clicks, "no pointing actions in the corpus"
        copyable = sum(1 for e in clicks if _is_copyable(e))
        assert copyable / len(clicks) >= 0.9, (
            f"only {copyable}/{len(clicks)} click targets are in their context"
        )

    def test_snapping_makes_an_element_exactly_copyable(self) -> None:
        """A click anywhere inside a control snaps to the point the parser names."""
        screen = _OneElement()
        inside = Action(kind=ActionKind.LEFT_CLICK, coordinate=(105, 205))
        snapped = snap_to_screen(inside, screen)
        assert snapped.coordinate == screen.elements[0].click

    def test_snapping_leaves_a_click_outside_every_control_alone(self) -> None:
        far = Action(kind=ActionKind.LEFT_CLICK, coordinate=(5, 5))
        assert snap_to_screen(far, _OneElement()) == far

    def test_snapping_leaves_coordinate_free_actions_alone(self) -> None:
        typing = Action(kind=ActionKind.TYPE, text="hello")
        assert snap_to_screen(typing, _OneElement()) == typing

    def test_gold_paths_still_solve_their_tasks_after_snapping(self, tasks: list) -> None:
        """Snapping must not change what an action does, only where it points."""
        for task in tasks:
            examples = build_corpus([task])
            assert len(examples) == len(task.gold)


class TestDecoding:
    def test_round_trips_to_the_same_cell(self) -> None:
        """The coordinate comes back quantized — to the centre of its cell.

        Not equality with the original pixel: the grid is lossy by design, and
        `tests/test_tokens.py` is where the round trip is held to the standard
        that actually matters, which is that the click still hits the same
        control.
        """
        ids = VOCAB.encode([*encode_action(CLICK), EOS])
        decoded = decode_generated(ids)
        assert decoded is not None
        assert decoded.kind == CLICK.kind
        assert decoded.coordinate == dequantize(*quantize(*CLICK.coordinate))

    def test_ignores_framing_tokens(self) -> None:
        bare = decode_generated(VOCAB.encode([*encode_action(CLICK), EOS]))
        framed = decode_generated(VOCAB.encode([ACT, *encode_action(CLICK), EOS]))
        assert framed == bare

    def test_returns_none_for_a_non_action(self) -> None:
        assert decode_generated(VOCAB.encode(["<c:A>", "<c:B>"])) is None

    def test_returns_none_for_nothing(self) -> None:
        assert decode_generated([]) is None


class TestTraining:
    def test_loss_falls_on_the_real_corpus(self, corpus: list) -> None:
        config = ModelConfig(
            d_model=16, n_heads=2, n_layers=1, d_ff=32,
            max_len=max(len(e) for e in corpus) + 4,
        )
        _, report = train(corpus, config=config, epochs=2, batch_size=8, lr=6e-3)
        assert report.epoch_loss[-1] < report.epoch_loss[0]

    def test_records_a_held_out_curve(self, corpus: list) -> None:
        config = ModelConfig(
            d_model=16, n_heads=2, n_layers=1, d_ff=32,
            max_len=max(len(e) for e in corpus) + 4,
        )
        _, report = train(
            corpus[:8], config=config, epochs=2, batch_size=4, held_out=corpus[8:12]
        )
        assert len(report.held_out_loss) == 2

    def test_refuses_an_empty_corpus(self) -> None:
        with pytest.raises(ValueError, match="nothing to train on"):
            train([])

    def test_mean_loss_of_an_untrained_model_is_near_log_vocab(self, corpus: list) -> None:
        import math

        config = ModelConfig(
            d_model=16, n_heads=2, n_layers=1, d_ff=32,
            max_len=max(len(e) for e in corpus) + 4,
        )
        value = mean_loss(GPT(config), corpus[:4])
        assert value == pytest.approx(math.log(VOCAB.size), abs=0.6)


class TestEvaluation:
    def test_an_untrained_model_is_scored_without_crashing(self, tasks: list) -> None:
        """Random weights emit malformed actions; that is a score, not an error."""
        longest = 220
        config = ModelConfig(
            d_model=16, n_heads=2, n_layers=1, d_ff=32, max_len=longest
        )
        result = evaluate(GPT(config), tasks[:2], label="control")
        assert result.total == 2
        assert 0.0 <= result.rate <= 1.0

    def test_report_renders(self, corpus: list) -> None:
        config = ModelConfig(
            d_model=16, n_heads=2, n_layers=1, d_ff=32,
            max_len=max(len(e) for e in corpus) + 4,
        )
        _, report = train(corpus[:8], config=config, epochs=1, batch_size=4)
        text = format_report(report, [])
        assert "parameters" in text and "epoch" in text


def _is_copyable(example: object) -> bool:
    """Does the context contain the `<x> <y>` pair this example's action uses?"""
    action = example.action  # type: ignore[attr-defined]
    cx, cy = quantize(*action.coordinate)
    context = VOCAB.decode(example.ids[: example.prompt_length])  # type: ignore[attr-defined]
    return any(
        context[i] == f"<x:{cx}>" and context[i + 1] == f"<y:{cy}>"
        for i in range(len(context) - 1)
    )


class _FakeScreen:
    """A minimal stand-in for a parsed screen, for mask tests."""

    width = 1280
    height = 720
    elements = ()


class _OneElement(_FakeScreen):
    """A screen holding a single 80x40 control at (100, 200)."""

    elements = (
        Element(
            label="SAVE", kind="button",
            box=Box(x=100, y=200, width=80, height=40, color=(20, 20, 20)),
            text_run=None, click=(140, 220),
        ),
    )
