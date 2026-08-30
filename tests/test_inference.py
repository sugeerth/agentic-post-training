"""The sampler, checked against the trainer it has to agree with.

The failure this file exists to prevent is the quiet one. A sampler that is
merely *close* to its trainer produces rollouts for a policy that does not
quite exist: the loss still falls, the reward still rises, and the run
optimizes something slightly other than the objective with no test failing.
So the headline assertion here is exact equality, not approximate.

The rest guards the accounting. A cache that reports a hit it did not have,
or a staleness check that passes a batch it should refuse, is worse than
having neither — both make a broken run look healthy.
"""

from __future__ import annotations

import random

import pytest

from computer_use.transformer import GPT, ModelConfig, generate
from inference.cache import RadixCache
from inference.engine import LocalEngine, Request, RequestRejected
from inference.rollout import GroupSpec, sample_group, to_rollout_batch
from inference.runtime import KVCache, PolicyWeights, logits_at, prefill, step
from inference.version import StalenessError, check_staleness


def _model(max_len: int = 64, seed: int = 7) -> GPT:
    return GPT(ModelConfig(max_len=max_len), seed=seed)


def _ids(n: int, vocab: int, seed: int = 0) -> list[int]:
    rng = random.Random(seed)
    return [rng.randrange(vocab) for _ in range(n)]


# --------------------------------------------------------------------------- #
# The claim everything rests on
# --------------------------------------------------------------------------- #


class TestAgreesWithTheTrainer:
    def test_logits_are_bit_identical_not_merely_close(self) -> None:
        """`==`, not `approx`. A sampler that rounds differently from its
        trainer sasmples from a policy the trainer never had."""
        model = _model()
        weights = PolicyWeights.snapshot(model)
        ids = _ids(40, model.config.vocab_size)

        reference = model.logits(ids, [len(ids) - 1]).data
        cache = KVCache(model.config.n_layers, model.config.d_model)
        produced = logits_at(weights, prefill(weights, cache, ids))

        assert produced == reference

    def test_it_holds_at_every_position_not_just_the_last(self) -> None:
        model = _model()
        weights = PolicyWeights.snapshot(model)
        ids = _ids(12, model.config.vocab_size, seed=3)

        positions = list(range(len(ids)))
        reference = model.logits(ids, positions).data
        vocab = model.config.vocab_size

        cache = KVCache(model.config.n_layers, model.config.d_model)
        for index, token in enumerate(ids):
            state = step(weights, cache, token)
            row = logits_at(weights, state)
            assert row == reference[index * vocab : (index + 1) * vocab]

    def test_greedy_decoding_matches_the_reference_generator(self) -> None:
        """Same weights, same prompt, same tokens — through a different path."""
        model = _model()
        weights = PolicyWeights.snapshot(model)
        ids = _ids(20, model.config.vocab_size, seed=5)

        expected = generate(model, ids, max_new=8)
        engine = LocalEngine(weights)
        produced = engine.generate([Request(prompt=ids, max_new=8)])[0]

        assert produced.tokens == expected

    def test_a_snapshot_does_not_move_when_the_model_does(self) -> None:
        """Weights are copied, not referenced: a trainer step mid-rollout
        cannot retroactively change what a sample was drawn from."""
        model = _model()
        weights = PolicyWeights.snapshot(model)
        before = list(weights.tok)

        model.tok.data[0] += 1.0

        assert weights.tok == before


# --------------------------------------------------------------------------- #
# The cache
# --------------------------------------------------------------------------- #


class TestPrefixCache:
    def test_a_repeated_prompt_is_computed_once(self) -> None:
        model = _model()
        weights = PolicyWeights.snapshot(model)
        cache = RadixCache(weights)
        ids = _ids(24, model.config.vocab_size, seed=11)

        cache.acquire(list(ids))
        _, hit = cache.acquire(list(ids))

        assert hit == len(ids)
        assert cache.stats.tokens_computed == len(ids)
        assert cache.stats.tokens_saved == len(ids)

    def test_a_divergent_branch_still_pays_for_what_it_shares(self) -> None:
        """The case that matters for multi-turn episodes, and the one a
        terminal-only cache silently reports as a total miss."""
        model = _model()
        weights = PolicyWeights.snapshot(model)
        cache = RadixCache(weights)
        shared = _ids(20, model.config.vocab_size, seed=13)

        cache.acquire([*shared, 1, 2, 3])
        _, hit = cache.acquire([*shared, 4, 5, 6])

        # Reuse is block-aligned, so the hit is the deepest block boundary
        # inside the shared span — not zero, and not more than was shared.
        assert hit == RadixCache.BLOCK
        assert 0 < hit <= len(shared)

    def test_an_identical_prompt_reuses_all_of_it_not_just_a_block(self) -> None:
        """A GRPO group is k samples of one prompt, so this is the hot path."""
        model = _model()
        weights = PolicyWeights.snapshot(model)
        cache = RadixCache(weights)
        ids = _ids(21, model.config.vocab_size, seed=13)

        cache.acquire(list(ids))
        _, hit = cache.acquire(list(ids))

        assert hit == len(ids)

    def test_a_reused_cache_produces_the_same_logits_as_a_cold_one(self) -> None:
        """The whole point: reuse must be invisible in the output."""
        model = _model()
        weights = PolicyWeights.snapshot(model)
        shared = _ids(16, model.config.vocab_size, seed=17)
        full = [*shared, 5, 6, 7]

        cold = KVCache(model.config.n_layers, model.config.d_model)
        expected = logits_at(weights, prefill(weights, cold, full))

        cache = RadixCache(weights)
        cache.acquire(list(shared))
        warm, hit = cache.acquire(list(full))
        warm.truncate(len(full) - 1)
        produced = logits_at(weights, step(weights, warm, full[-1]))

        assert hit == len(shared)
        assert produced == expected

    def test_the_returned_cache_is_private_to_its_caller(self) -> None:
        """A continuation appended to the shared node would corrupt the prefix
        for every request that arrives afterwards."""
        model = _model()
        weights = PolicyWeights.snapshot(model)
        cache = RadixCache(weights)
        ids = _ids(10, model.config.vocab_size, seed=19)

        first, _ = cache.acquire(list(ids))
        step(weights, first, 3)
        second, hit = cache.acquire(list(ids))

        assert hit == len(ids)
        assert second.length == len(ids)

    def test_new_weights_drop_everything_cached_under_the_old_ones(self) -> None:
        model = _model()
        cache = RadixCache(PolicyWeights.snapshot(model, version=1))
        ids = _ids(12, model.config.vocab_size, seed=23)
        cache.acquire(list(ids))

        cache.rebind(PolicyWeights.snapshot(_model(seed=8), version=2))
        _, hit = cache.acquire(list(ids))

        assert hit == 0

    def test_capacity_is_enforced_by_evicting(self) -> None:
        model = _model()
        weights = PolicyWeights.snapshot(model)
        cache = RadixCache(weights, capacity=2)

        for seed in range(6):
            cache.acquire(_ids(8, model.config.vocab_size, seed=seed))

        assert cache.stats.evictions > 0


# --------------------------------------------------------------------------- #
# Versioning
# --------------------------------------------------------------------------- #


class TestStaleness:
    def test_an_on_policy_batch_passes_and_says_so(self) -> None:
        report = check_staleness([4, 4, 4], trainer_version=4)

        assert report.on_policy
        assert report.max_lag == 0

    def test_a_lagging_batch_is_refused_by_default(self) -> None:
        with pytest.raises(StalenessError, match="no longer exists"):
            check_staleness([3, 3, 4], trainer_version=4)

    def test_lag_is_allowed_when_the_caller_says_it_is_intended(self) -> None:
        report = check_staleness([2, 3, 4], trainer_version=4, max_staleness=2)

        assert report.max_lag == 2
        assert not report.on_policy

    def test_a_sample_ahead_of_the_trainer_is_a_different_error(self) -> None:
        """Two clocks, not one — a distinct bug from ordinary lag."""
        with pytest.raises(StalenessError, match="ahead of"):
            check_staleness([5], trainer_version=4)

    def test_an_empty_batch_is_refused_rather_than_passing_vacuously(self) -> None:
        with pytest.raises(ValueError, match="no samples"):
            check_staleness([], trainer_version=1)


# --------------------------------------------------------------------------- #
# Admission
# --------------------------------------------------------------------------- #


class TestAdmission:
    def test_a_prompt_with_no_room_to_generate_is_refused_up_front(self) -> None:
        model = _model(max_len=32)
        engine = LocalEngine(PolicyWeights.snapshot(model))

        with pytest.raises(RequestRejected, match="over the 32-token context"):
            engine.generate([Request(prompt=_ids(30, 100), max_new=8)])

    def test_an_empty_prompt_is_refused_at_construction(self) -> None:
        with pytest.raises(RequestRejected, match="empty prompt"):
            Request(prompt=[])

    def test_rejection_happens_before_any_work_is_spent(self) -> None:
        """One bad request in a batch costs nothing, not a partial run."""
        model = _model(max_len=32)
        engine = LocalEngine(PolicyWeights.snapshot(model))
        good = Request(prompt=_ids(4, 100), max_new=2)
        bad = Request(prompt=_ids(30, 100), max_new=8)

        with pytest.raises(RequestRejected):
            engine.generate([good, bad])

        assert engine.stats.requests == 0


# --------------------------------------------------------------------------- #
# Groups and batches
# --------------------------------------------------------------------------- #


class TestGroups:
    def _engine(self) -> LocalEngine:
        return LocalEngine(PolicyWeights.snapshot(_model(), version=2), seed=1)

    def test_a_group_of_one_is_refused_because_it_cannot_teach(self) -> None:
        with pytest.raises(ValueError, match="no learning signal"):
            GroupSpec(prompt=[1, 2, 3], size=1)

    def test_greedy_sampling_of_a_group_is_refused(self) -> None:
        with pytest.raises(ValueError, match="every advantage is"):
            GroupSpec(prompt=[1, 2, 3], size=4, temperature=0.0)

    def test_the_group_shares_one_policy_version(self) -> None:
        engine = self._engine()
        group = sample_group(
            engine, GroupSpec(prompt=_ids(12, 100), size=4, max_new=4), lambda t, g: 1.0
        )

        assert group.version == 2
        assert {c.version for c in group.completions} == {2}

    def test_the_shared_prompt_is_computed_once_for_the_whole_group(self) -> None:
        engine = self._engine()
        prompt = _ids(16, 100, seed=29)
        spec = GroupSpec(prompt=prompt, size=6, max_new=3)

        sample_group(engine, spec, lambda t, g: 1.0)

        # One member pays for the prompt; the other five hit it.
        assert engine.cache.stats.tokens_computed <= len(prompt) + 6 * 3
        assert group_hits(engine) == 5 * len(prompt)

    def test_advantages_centre_on_the_group(self) -> None:
        engine = self._engine()
        rewards = iter([0.0, 1.0, 0.0, 1.0])
        group = sample_group(
            engine, GroupSpec(prompt=_ids(10, 100), size=4, max_new=2),
            lambda t, g: next(rewards),
        )

        assert sum(group.advantages) == pytest.approx(0.0, abs=1e-9)

    def test_a_group_where_everything_scored_alike_is_marked_degenerate(self) -> None:
        engine = self._engine()
        group = sample_group(
            engine, GroupSpec(prompt=_ids(10, 100), size=4, max_new=2), lambda t, g: 0.5
        )

        assert group.degenerate
        assert group.advantages == (0.0, 0.0, 0.0, 0.0)

    def test_the_batch_carries_log_probs_so_grpo_takes_its_real_path(self) -> None:
        """With log_probs absent the technique falls back to a decay curve,
        which is a demo rather than training."""
        engine = self._engine()
        rewards = iter([0.0, 1.0] * 4)
        groups = [
            sample_group(
                engine, GroupSpec(prompt=_ids(10, 100, seed=s), size=2, max_new=3),
                lambda t, g: next(rewards),
            )
            for s in (31, 37)
        ]

        batch, report = to_rollout_batch(groups)

        assert batch.log_probs is not None
        assert len(batch.log_probs) == 2
        assert len(batch.log_probs[0]) == 2
        assert report.groups == 2
        assert report.samples == 4

    def test_degenerate_groups_are_kept_so_the_batch_size_stays_fixed(self) -> None:
        engine = self._engine()
        groups = [
            sample_group(
                engine, GroupSpec(prompt=_ids(10, 100, seed=s), size=2, max_new=2),
                lambda t, g: 1.0,
            )
            for s in (41, 43)
        ]

        batch, report = to_rollout_batch(groups)

        assert report.degenerate_groups == 2
        assert report.usable_groups == 0
        assert len(batch.prompts) == 2

    def test_ragged_samples_are_padded_into_a_rectangular_tensor(self) -> None:
        """Samples stop at different lengths, and a ragged nested list is not
        a `(B, G, T)` tensor. Unpadded this raises inside the trainer, on a
        GPU machine, after the sampling has already been paid for."""
        engine = self._engine()
        lengths = iter([3, 7, 5, 7])
        group = _group_with_lengths(engine, lengths)

        batch, _ = to_rollout_batch([group])
        widths = {len(sample) for sample in batch.log_probs[0]}

        assert len(widths) == 1

    def test_the_mask_records_which_positions_were_real(self) -> None:
        """Padding is 0.0, which is a log-probability of 1.0 — summing without
        the mask hands free probability mass to whichever samples stopped
        early. So the mask has to travel with the batch."""
        engine = self._engine()
        group = _group_with_lengths(engine, iter([3, 7, 5, 7]))

        batch, _ = to_rollout_batch([group])
        mask = batch.metadata["response_mask"][0]
        real = batch.metadata["response_lengths"][0]

        assert [int(sum(m)) for m in mask] == list(real)
        for row, length in zip(mask, real, strict=True):
            assert all(v == 1.0 for v in row[:length])
            assert all(v == 0.0 for v in row[length:])

    def test_padding_never_shortens_a_sample(self) -> None:
        engine = self._engine()
        group = _group_with_lengths(engine, iter([2, 9, 4, 9]))

        batch, _ = to_rollout_batch([group])
        longest = max(len(c.logprobs) for c in group.completions)

        assert all(len(s) == longest for s in batch.log_probs[0])

    def test_an_empty_batch_is_refused(self) -> None:
        with pytest.raises(ValueError, match="nothing to learn from"):
            to_rollout_batch([])


def group_hits(engine: LocalEngine) -> int:
    return engine.cache.stats.tokens_saved


def _group_with_lengths(engine: LocalEngine, lengths) -> object:
    """A group whose samples deliberately stop at different lengths."""
    from inference.engine import Completion
    from inference.rollout import Group

    spec = GroupSpec(prompt=_ids(8, 100), size=4, max_new=12, temperature=0.9)
    completions = tuple(
        Completion(
            tokens=[1] * n, logprobs=[-0.5] * n, version=engine.version,
            prefix_hit=0, stopped=True, tag=f"s{i}",
        )
        for i, n in enumerate(lengths)
    )
    return Group(
        spec=spec, completions=completions,
        rewards=tuple(float(len(c)) for c in completions), version=engine.version,
    )
