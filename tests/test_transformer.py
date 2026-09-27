"""The transformer: shape, causality, generation, persistence, and learning.

The load-bearing test here is `test_overfits_a_single_sequence`. Everything
else checks that the pieces are wired together; that one checks that the wiring
produces a model capable of learning anything at all. A transformer with a
subtly broken residual or a mis-scaled attention will still run, still emit
tokens, and still pass every structural test in this file.
"""

from __future__ import annotations

import random

import pytest

from computer_use.nn import Adam
from computer_use.tokens import VOCAB
from computer_use.transformer import GPT, ModelConfig, generate, load_model, save_model


def tiny(**overrides: int) -> ModelConfig:
    """Small enough that a test is a second, big enough to be a transformer."""
    base = {
        "vocab_size": 24, "d_model": 16, "n_heads": 2,
        "n_layers": 2, "d_ff": 32, "max_len": 24,
    }
    base.update(overrides)
    return ModelConfig(**base)  # type: ignore[arg-type]


class TestShape:
    def test_head_dimension_must_divide(self) -> None:
        with pytest.raises(ValueError, match="does not divide"):
            ModelConfig(d_model=48, n_heads=5)

    def test_logits_cover_the_vocabulary(self) -> None:
        model = GPT(tiny())
        out = model.logits([1, 2, 3])
        assert out.shape == (3, 24)

    def test_selected_positions_only(self) -> None:
        model = GPT(tiny())
        assert model.logits([1, 2, 3, 4], [1, 3]).shape == (2, 24)

    def test_selected_rows_match_the_full_projection(self) -> None:
        """Restricting the projection is an optimization, not an approximation."""
        model = GPT(tiny())
        ids = [1, 2, 3, 4, 5]
        full = model.logits(ids)
        picked = model.logits(ids, [1, 4])
        assert picked.row(0) == pytest.approx(full.row(1))
        assert picked.row(1) == pytest.approx(full.row(4))

    def test_rejects_a_sequence_longer_than_max_len(self) -> None:
        model = GPT(tiny(max_len=8))
        with pytest.raises(ValueError, match="max_len"):
            model.logits(list(range(9)))

    def test_output_is_tied_to_the_embedding(self) -> None:
        """One matrix, so training the output trains the input."""
        model = GPT(tiny())
        assert sum(p is model.tok for p in model.params()) == 1
        before = list(model.tok.data)
        model.tok.data[0] += 1.0
        assert model.tok.data != before

    def test_parameter_count_is_what_the_shape_implies(self) -> None:
        config = tiny()
        model = GPT(config)
        d, ff, layers = config.d_model, config.d_ff, config.n_layers
        expected = (
            config.vocab_size * d          # token embedding (tied to output)
            + config.max_len * d           # positions
            + 2 * d                        # final layer norm
            + layers * (4 * d * d + 2 * d * ff + ff + d + 4 * d)
        )
        assert model.n_params == expected


class TestCausality:
    def test_a_later_token_cannot_change_an_earlier_prediction(self) -> None:
        """The property the whole architecture depends on, checked directly."""
        model = GPT(tiny())
        a = model.logits([1, 2, 3, 4])
        b = model.logits([1, 2, 3, 9])
        assert a.row(0) == pytest.approx(b.row(0))
        assert a.row(2) == pytest.approx(b.row(2))
        assert a.row(3) != pytest.approx(b.row(3))

    def test_prefix_states_are_stable_as_a_sequence_grows(self) -> None:
        """Appending a token must not disturb the states already computed."""
        model = GPT(tiny())
        short = model.hidden([3, 1, 4])
        longer = model.hidden([3, 1, 4, 1, 5])
        assert short.row(2) == pytest.approx(longer.row(2))


class TestGeneration:
    def test_greedy_is_deterministic(self) -> None:
        model = GPT(tiny())
        assert generate(model, [1, 2], max_new=5) == generate(model, [1, 2], max_new=5)

    def test_stops_at_a_stop_token(self) -> None:
        model = GPT(tiny())
        produced = generate(model, [1], max_new=10, stop=range(24))
        assert len(produced) == 1

    def test_respects_max_new(self) -> None:
        model = GPT(tiny())
        assert len(generate(model, [1, 2], max_new=4)) == 4

    def test_stops_at_max_len_rather_than_overrunning(self) -> None:
        model = GPT(tiny(max_len=8))
        assert len(generate(model, [1] * 6, max_new=20)) == 2

    def test_temperature_can_diverge_from_greedy(self) -> None:
        model = GPT(tiny(), seed=3)
        hot = generate(model, [1, 2], max_new=8, temperature=2.0, rng=random.Random(7))
        assert len(hot) <= 8


class TestPersistence:
    def test_round_trips_through_json(self, tmp_path) -> None:
        model = GPT(tiny(), seed=5)
        path = tmp_path / "model.json"
        save_model(model, path)
        restored = load_model(path)
        assert restored.config == model.config
        assert generate(restored, [1, 2], max_new=4) == generate(model, [1, 2], max_new=4)

    def test_rejects_a_mismatched_checkpoint(self, tmp_path) -> None:
        path = tmp_path / "model.json"
        save_model(GPT(tiny()), path)
        text = path.read_text()
        path.write_text(text.replace('"d_model": 16', '"d_model": 32'))
        with pytest.raises(ValueError, match="does not match"):
            load_model(path)


class TestLearning:
    def test_overfits_a_single_sequence(self) -> None:
        """The end-to-end check that the architecture can learn at all.

        One sequence, memorized. If the residual stream, the attention scaling
        or any backward is wrong, this loss plateaus well above zero while
        every structural test above still passes.
        """
        model = GPT(tiny(), seed=1)
        optimizer = Adam(model.params(), lr=0.02)
        ids = [1, 5, 9, 3, 7, 2, 8]
        supervised = list(range(len(ids) - 1))
        first = model.loss(ids, supervised).data[0]
        for _ in range(120):
            optimizer.zero_grad()
            loss = model.loss(ids, supervised)
            loss.backward()
            optimizer.step()
        assert loss.data[0] < 0.05, f"{first:.3f} -> {loss.data[0]:.3f}"

    def test_learns_to_copy_from_context(self) -> None:
        """The mechanism the GUI result depends on, in miniature.

        Sequences look like `k a k ?`, where the answer is whatever followed
        the earlier occurrence of `k`. Nothing about the answer is fixed — it
        differs per example — so a model can only score well by attending back
        to the earlier match and copying its successor. That is the same
        operation as reading a coordinate out of an observation, and it is what
        two layers buys over one.
        """
        rng = random.Random(0)
        model = GPT(tiny(vocab_size=12, max_len=12), seed=2)
        optimizer = Adam(model.params(), lr=0.02)

        def sample() -> tuple[list[int], list[int]]:
            key, value = rng.randrange(2, 12), rng.randrange(2, 12)
            noise = rng.randrange(2, 12)
            ids = [0, key, value, noise, key, value]
            return ids, [4]  # predict `value` after the second `key`

        for _ in range(400):
            optimizer.zero_grad()
            for _ in range(4):
                ids, supervised = sample()
                model.loss(ids, supervised).backward()
            for p in model.params():
                if p.grad is not None:
                    p.grad = [g * 0.25 for g in p.grad]
            optimizer.step()

        correct = 0
        for _ in range(30):
            ids, _ = sample()
            prediction = generate(model, ids[:5], max_new=1)[0]
            correct += int(prediction == ids[5])
        assert correct >= 24, f"copied {correct}/30 — induction did not form"


class TestDefaults:
    def test_default_config_matches_the_interaction_vocabulary(self) -> None:
        assert ModelConfig().vocab_size == VOCAB.size
