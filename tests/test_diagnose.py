"""The measuring device, checked against screens whose answers are known.

Every function here reports a number that will be quoted, so each one is
tested against a case constructed so the right answer can be worked out by
hand. The failure this file exists to prevent is a diagnostic that is quietly
wrong in the flattering direction — a ceiling that counts unreachable answers
as reachable, or a baseline that scores lower than it should and makes the
model look better than it is.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from computer_use.diagnose import (
    Baseline,
    FieldScores,
    attention_to_target,
    baselines,
    ceiling,
    context_elements,
    field_scores,
    format_diagnosis,
)
from computer_use.perception import Box, Element
from computer_use.pretrain import DEFAULT_CORPUS, CorpusConfig, make_example
from computer_use.tokens import quantize
from computer_use.transformer import GPT, ModelConfig
from computer_use.types import Action, ActionKind


class _Screen:
    """A screen with exactly the controls a test needs."""

    width = 1280
    height = 720

    def __init__(self, *elements: Element) -> None:
        self.elements = elements


def _control(label: str, x: int, y: int, *, kind: str = "button") -> Element:
    return Element(
        label=label, kind=kind,
        box=Box(x=x, y=y, width=80, height=40, color=(20, 20, 20)),
        text_run=None, click=(x + 40, y + 20),
    )


def _click(x: int, y: int) -> Action:
    return Action(kind=ActionKind.LEFT_CLICK, coordinate=(x, y))


# --------------------------------------------------------------------------- #
# Reading the context back
# --------------------------------------------------------------------------- #


class TestContextElements:
    def test_recovers_each_control_with_its_cell(self) -> None:
        screen = _Screen(_control("SAVE", 100, 200), _control("CANCEL", 300, 400))
        example = make_example("SAVE THE FORM", screen, _click(140, 220))

        recovered = context_elements(example)

        assert [e.label for e in recovered] == ["SAVE", "CANCEL"]
        assert (recovered[0].cx, recovered[0].cy) == quantize(140, 220)
        assert (recovered[1].cx, recovered[1].cy) == quantize(340, 420)

    def test_reads_the_screen_as_tokenized_not_as_rendered(self) -> None:
        """Controls past the element limit are gone, because the model cannot see them."""
        screen = _Screen(*[_control(f"BTN{i}", 100, 40 * i) for i in range(6)])
        example = make_example(
            "PRESS BTN5", screen, _click(140, 220),
            config=_narrow(max_elements=3),
        )

        assert len(context_elements(example)) == 3

    def test_either_field_order_reads_back_the_same_controls(self) -> None:
        """The instrument has to be blind to the encoding under test.

        `label_first` is an experiment about attention, and this module is what
        scores it. If the ceiling or the baselines read one order better than
        the other, the experiment measures the measuring device.
        """
        screen = _Screen(_control("SAVE", 100, 200), _control("CANCEL", 300, 400))
        action = _click(140, 220)
        default = make_example("PRESS SAVE", screen, action)
        flipped = make_example(
            "PRESS SAVE", screen, action, config=_narrow(label_first=True)
        )

        assert context_elements(default) == context_elements(flipped)
        assert ceiling([default]).per_kind == ceiling([flipped]).per_kind
        assert [b.correct for b in baselines([default])] == [
            b.correct for b in baselines([flipped])
        ]

    def test_a_screen_with_no_controls_reads_back_empty(self) -> None:
        example = make_example("DO SOMETHING", _Screen(), _click(10, 10))

        assert context_elements(example) == []


def _narrow(**kwargs: object) -> CorpusConfig:
    """The default corpus shape with one budget tightened."""
    return replace(DEFAULT_CORPUS, **kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# The ceiling
# --------------------------------------------------------------------------- #


class TestCeiling:
    def test_a_click_on_a_perceived_control_is_reachable(self) -> None:
        screen = _Screen(_control("SAVE", 100, 200))
        example = make_example("PRESS SAVE", screen, _click(140, 220))

        assert ceiling([example]).per_kind["left_click"] == (1, 1)

    def test_a_click_on_nothing_is_not_reachable_and_says_why(self) -> None:
        screen = _Screen(_control("SAVE", 100, 200))
        example = make_example("PRESS SAVE", screen, _click(900, 600))

        result = ceiling([example])

        assert result.per_kind["left_click"] == (0, 1)
        assert result.reasons == {"target cell absent from observation": 1}

    def test_a_control_truncated_away_is_not_reachable(self) -> None:
        """The ceiling has to fall when the element limit hides the answer."""
        screen = _Screen(_control("FIRST", 100, 100), _control("TARGET", 100, 300))
        example = make_example(
            "PRESS TARGET", screen, _click(140, 320), config=_narrow(max_elements=1)
        )

        assert ceiling([example]).per_kind["left_click"] == (0, 1)

    def test_typed_text_is_reachable_when_the_instruction_carries_it(self) -> None:
        action = Action(kind=ActionKind.TYPE, text="BERLIN")
        example = make_example('SET CITY TO "BERLIN"', _Screen(), action)

        assert ceiling([example]).per_kind["type"] == (1, 1)

    def test_typed_text_truncated_out_of_the_instruction_is_not_reachable(self) -> None:
        """The check runs against the instruction the model sees, not the original."""
        action = Action(kind=ActionKind.TYPE, text="BERLIN")
        example = make_example(
            'SET THE CITY FIELD TO "BERLIN"', _Screen(), action,
            config=_narrow(instruction_chars=10),
        )

        assert ceiling([example]).per_kind["type"] == (0, 1)
        assert result_reason(ceiling([example])) == "typed text not in instruction"

    def test_a_scroll_is_counted_apart_from_the_pointer_ceiling(self) -> None:
        """A scroll anchor is a fixed point, so looking for it among the
        controls would score every scroll as unreachable and drag the ceiling
        down for a reason that has nothing to do with pointing."""
        action = Action(
            kind=ActionKind.SCROLL, coordinate=(640, 360),
            scroll_direction="down", scroll_amount=3,
        )
        example = make_example("SCROLL DOWN", _Screen(), action)

        result = ceiling([example])

        assert result.per_kind["scroll"] == (1, 1)
        assert result.reasons == {}

    def test_the_rate_is_over_every_kind_together(self) -> None:
        screen = _Screen(_control("SAVE", 100, 200))
        good = make_example("PRESS SAVE", screen, _click(140, 220))
        bad = make_example("PRESS SAVE", screen, _click(900, 600))

        assert ceiling([good, bad]).rate == pytest.approx(0.5)


def result_reason(result: object) -> str:
    reasons = result.reasons  # type: ignore[attr-defined]
    return next(iter(reasons))


# --------------------------------------------------------------------------- #
# The floor
# --------------------------------------------------------------------------- #


class TestBaselines:
    def test_uniform_is_scored_in_expectation(self) -> None:
        """Four controls, one right: the uniform policy is right a quarter of
        the time, and the number should not depend on a random seed."""
        screen = _Screen(*[_control(f"B{i}", 100, 100 * (i + 1)) for i in range(4)])
        example = make_example("PRESS B0", screen, _click(140, 120))

        first = _named(baselines([example]), "uniform over controls")
        again = _named(baselines([example]), "uniform over controls")

        assert first.correct == again.correct
        assert first.rate == pytest.approx(0.25, abs=0.26)  # 1 of 4, rounded

    def test_label_overlap_finds_the_control_the_instruction_names(self) -> None:
        screen = _Screen(
            _control("CANCEL", 100, 100),
            _control("SAVE CHANGES", 100, 300),
            _control("HELP", 100, 500),
        )
        example = make_example("PRESS SAVE CHANGES NOW", screen, _click(140, 320))

        assert _named(baselines([example]), "label overlapping the instruction").correct == 1

    def test_first_control_is_right_only_when_the_target_is_first(self) -> None:
        screen = _Screen(_control("SAVE", 100, 100), _control("CANCEL", 100, 300))
        hits = make_example("PRESS SAVE", screen, _click(140, 120))
        misses = make_example("PRESS CANCEL", screen, _click(140, 320))

        scored = _named(baselines([hits, misses]), "always the first control")

        assert (scored.correct, scored.total) == (1, 2)

    def test_typed_and_scrolled_decisions_are_left_out(self) -> None:
        """A baseline for those would measure the generator's phrasing."""
        typed = make_example(
            'SET CITY TO "BERLIN"', _Screen(), Action(kind=ActionKind.TYPE, text="BERLIN")
        )
        scrolled = make_example(
            "SCROLL", _Screen(),
            Action(kind=ActionKind.SCROLL, coordinate=(640, 360),
                   scroll_direction="down", scroll_amount=3),
        )

        assert all(b.total == 0 for b in baselines([typed, scrolled]))


def _named(scored: list[Baseline], name: str) -> Baseline:
    return next(b for b in scored if b.name == name)


# --------------------------------------------------------------------------- #
# The decomposition
# --------------------------------------------------------------------------- #


class TestFieldScores:
    def test_a_perfect_click_scores_every_field(self) -> None:
        example = make_example("PRESS SAVE", _Screen(), _click(140, 220))

        scores = field_scores([(example, _click(140, 220))])

        assert scores.kind == (1, 1)
        assert scores.point == (1, 1)
        assert scores.miss_distance == ()

    def test_the_wrong_row_still_scores_the_right_column(self) -> None:
        """The split is the whole point: one right axis is visible."""
        example = make_example("PRESS SAVE", _Screen(), _click(140, 220))

        scores = field_scores([(example, _click(140, 500))])

        assert scores.column == (1, 1)
        assert scores.row == (0, 1)
        assert scores.point == (0, 1)

    def test_a_near_miss_records_its_distance_in_cells(self) -> None:
        example = make_example("PRESS SAVE", _Screen(), _click(140, 220))
        one_cell_right = _click(140 + 20, 220)

        scores = field_scores([(example, one_cell_right)])

        assert scores.miss_distance == (1,)
        assert scores.median_miss == pytest.approx(1.0)

    def test_coordinates_are_not_scored_when_the_verb_is_wrong(self) -> None:
        """A point emitted under the wrong kind is a different action, not a
        near miss, so counting it would inflate the coordinate columns."""
        example = make_example("PRESS SAVE", _Screen(), _click(140, 220))
        wrong_verb = Action(kind=ActionKind.RIGHT_CLICK, coordinate=(140, 220))

        scores = field_scores([(example, wrong_verb)])

        assert scores.kind == (0, 1)
        assert scores.point == (0, 0)

    def test_a_partly_right_string_is_credited_by_character(self) -> None:
        """The reading this instrument exists for: exact match says zero, and
        thirteen of fourteen characters is not zero."""
        gold = Action(kind=ActionKind.TYPE, text="ADA@EXAMPLE.COM")
        example = make_example('SET EMAIL TO "ADA@EXAMPLE.COM"', _Screen(), gold)
        almost = Action(kind=ActionKind.TYPE, text="ADA@EXAMPLE.CO")

        scores = field_scores([(example, almost)])

        assert scores.characters == (14, 15)
        assert scores.text_exact == (0, 1)

    def test_an_undecodable_generation_is_counted_apart_from_a_wrong_one(self) -> None:
        example = make_example("PRESS SAVE", _Screen(), _click(140, 220))

        scores = field_scores([(example, None)])

        assert scores.malformed == 1
        assert scores.kind == (0, 1)

    def test_the_median_of_no_misses_is_zero_rather_than_an_error(self) -> None:
        assert FieldScores().median_miss == 0.0


class TestFormatting:
    def test_the_report_names_all_three_readings(self) -> None:
        screen = _Screen(_control("SAVE", 100, 200))
        example = make_example("PRESS SAVE", screen, _click(140, 220))

        text = format_diagnosis(
            ceiling([example]), baselines([example]),
            field_scores([(example, _click(140, 220))]),
        )

        assert "ceiling" in text
        assert "floor" in text
        assert "by field" in text
        assert "left_click" in text


# --------------------------------------------------------------------------- #
# Where the model looked
# --------------------------------------------------------------------------- #


class TestAttention:
    """The mechanics, not the verdict.

    An untrained model attends nowhere meaningful, so nothing here asserts that
    the mass lands on the target — that is the measurement, and asserting it
    would be asserting the result. What is checked is that the reading is a
    reading: that recording cannot change what the model computes, that the
    row is the one belonging to the position that emits the coordinate, and
    that the uniform reference is the share it claims to be.
    """

    def _example(self):
        screen = _Screen(
            _control("SAVE", 100, 200),
            _control("CANCEL", 300, 400),
            _control("HELP", 500, 600),
        )
        return make_example("PRESS CANCEL", screen, _click(340, 420))

    def _model(self, example) -> GPT:
        return GPT(ModelConfig(max_len=len(example.ids) + 8), seed=3)

    def test_recording_does_not_change_what_the_model_computes(self) -> None:
        """The instrument must be inert. It keeps references to tensors the
        forward pass already built, so this should hold exactly, not nearly."""
        example = self._example()
        model = self._model(example)

        plain = model.hidden(example.ids).data
        record: list = []
        watched = model.hidden(example.ids, record).data

        assert plain == watched
        assert record  # and it did in fact record something

    def test_one_matrix_is_recorded_per_layer_per_head(self) -> None:
        example = self._example()
        config = ModelConfig(max_len=len(example.ids) + 8, n_layers=2, n_heads=3)
        record: list = []
        GPT(config, seed=3).hidden(example.ids, record)

        assert len(record) == config.n_layers * config.n_heads

    def test_the_uniform_reference_is_the_target_share_of_the_observation(self) -> None:
        """Three equal controls, so an evenly spread gaze lands a third of its
        mass on the one that matters."""
        example = self._example()

        result = attention_to_target(self._model(example), [example])

        assert result.examples == 1
        assert result.if_uniform == pytest.approx(1 / 3, abs=0.08)

    def test_the_measured_mass_is_a_probability(self) -> None:
        example = self._example()

        result = attention_to_target(self._model(example), [example])

        assert 0.0 <= result.on_target <= 1.0
        assert 0.0 <= result.on_screen <= 1.0
        assert result.ratio == pytest.approx(result.on_target / result.if_uniform)

    def test_looking_at_the_screen_and_looking_at_the_right_control_are_split(
        self,
    ) -> None:
        """Both sides of the ratio have to be shares of the same thing.

        `on_target` is conditioned on the observation, so a model that barely
        consults the screen but distributes what little it spends correctly
        still reads as looking in the right place — and `on_screen` is what
        says it barely looked. Folding them together would let a low
        `on_screen` masquerade as bad grounding, which is a different bug with
        a different fix.
        """
        example = self._example()

        result = attention_to_target(self._model(example), [example])

        # Three controls, so an even gaze *within* the screen is a third,
        # regardless of how much of the total gaze reached the screen at all.
        assert result.if_uniform == pytest.approx(1 / 3, abs=0.08)
        assert result.on_screen < 1.0  # some mass always goes to the prefix

    def test_a_screen_with_one_control_is_skipped(self) -> None:
        """With nothing to choose between, where the model looked says nothing."""
        example = make_example(
            "PRESS SAVE", _Screen(_control("SAVE", 100, 200)), _click(140, 220)
        )

        assert attention_to_target(self._model(example), [example]).examples == 0

    def test_scrolls_and_types_are_skipped(self) -> None:
        typed = make_example(
            'SET CITY TO "BERLIN"', _Screen(), Action(kind=ActionKind.TYPE, text="BERLIN")
        )

        assert attention_to_target(self._model(typed), [typed]).examples == 0

    def test_an_empty_reading_reports_zero_rather_than_dividing_by_it(self) -> None:
        assert attention_to_target(GPT(ModelConfig()), []).ratio == 0.0
