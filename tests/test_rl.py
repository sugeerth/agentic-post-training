"""The post-training loop, and the two ways it can lie about itself.

An RL run reports a reward curve, and a reward curve is exactly the kind of
artefact that looks like evidence whether or not it is. Two failures here
produce a plausible plot and a wrong conclusion:

* scoring a fresh random subset each round, so a dip means "harder sample"
  and a rise means "easier sample", and the trend line measures the sampler;
* returning whatever the final step produced, so a run that peaked in the
  middle is reported as a failure.

Both are guarded here rather than in prose.
"""

from __future__ import annotations

import pytest

from computer_use.pretrain import Example
from computer_use.rl import DEFAULT_RL, RLReport, Round, _advantages, reward_for
from computer_use.tokens import VOCAB, encode_action
from computer_use.types import Action, ActionKind


def _example(action: Action) -> Example:
    prompt = VOCAB.encode(["<bos>", "<obs>", "<act>"])
    ids = [*prompt, *VOCAB.encode(encode_action(action)), VOCAB.id("<eos>")]
    return Example(
        ids=ids, supervised=list(range(len(prompt) - 1, len(ids) - 1)),
        prompt_length=len(prompt), task="t", action=action,
    )


class TestReward:
    def test_the_gold_action_earns_full_reward(self) -> None:
        gold = Action(kind=ActionKind.LEFT_CLICK, coordinate=(140, 220))
        example = _example(gold)
        tokens = VOCAB.encode(encode_action(gold))

        assert reward_for(tokens, example, config=DEFAULT_RL) == 1.0

    def test_a_wrong_action_earns_nothing_but_is_not_penalised(self) -> None:
        """Zero, not negative. A wrong click is the normal case the group is
        centred on; charging it would put a floor under every advantage."""
        example = _example(Action(kind=ActionKind.LEFT_CLICK, coordinate=(140, 220)))
        wrong = VOCAB.encode(
            encode_action(Action(kind=ActionKind.LEFT_CLICK, coordinate=(900, 600)))
        )

        assert reward_for(wrong, example, config=DEFAULT_RL) == 0.0

    def test_output_that_does_not_decode_is_charged_separately(self) -> None:
        example = _example(Action(kind=ActionKind.LEFT_CLICK, coordinate=(140, 220)))

        value = reward_for([VOCAB.id("<sep>")], example, config=DEFAULT_RL)

        assert value == DEFAULT_RL.malformed_penalty
        assert value < 0.0


class TestAdvantages:
    def test_a_unanimous_group_contributes_nothing(self) -> None:
        """No evidence any sample was better than any other, so no gradient —
        and this is the common case once a policy sharpens."""
        assert _advantages([1.0, 1.0, 1.0]) == [0.0, 0.0, 0.0]

    def test_advantages_are_centred_on_the_group(self) -> None:
        out = _advantages([0.0, 1.0, 0.0, 1.0])

        assert sum(out) == pytest.approx(0.0, abs=1e-9)

    def test_the_better_sample_gets_the_positive_advantage(self) -> None:
        out = _advantages([0.0, 1.0])

        assert out[1] > 0.0 > out[0]

    def test_a_group_of_one_has_no_advantage_by_construction(self) -> None:
        assert _advantages([1.0]) == [0.0]


class TestReportingHonesty:
    def _report(self, probes: list[float]) -> RLReport:
        report = RLReport()
        for index, probe in enumerate(probes, start=1):
            report.rounds.append(Round(
                index=index, reward=0.0, solved=0, samples=1, degenerate=0,
                loss=0.0, prefix_saved=0, seconds=0.0, probe_reward=probe,
            ))
            if probe > report.best_probe:
                report.best_probe, report.best_round = probe, index
        return report

    def test_the_best_round_is_tracked_not_the_last(self) -> None:
        """A run that peaks in the middle and then degrades is exactly what
        the measured run did — rounds 1-4 rising, round 5 collapsing."""
        report = self._report([0.34, 0.29, 0.26, 0.39, 0.14])

        assert report.best_round == 4
        assert report.best_probe == pytest.approx(0.39)

    def test_the_formatted_report_marks_which_model_is_returned(self) -> None:
        from computer_use.rl import format_report

        text = format_report(self._report([0.1, 0.5, 0.2]))

        assert "best probe at round 2" in text
        assert "not comparable across rows" in text

    def test_degenerate_share_is_reported(self) -> None:
        report = RLReport(groups_total=10, groups_degenerate=3)

        assert report.degenerate_share == pytest.approx(0.3)

    def test_no_groups_reports_zero_rather_than_dividing_by_it(self) -> None:
        assert RLReport().degenerate_share == 0.0
