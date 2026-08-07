"""Phase 1 contract tests: core/ and the GRPO migration exemplar.

These tests pin the *new* shape of the framework. They do NOT depend on
torch — the simulation path is exercised on purpose so CI stays cheap.
"""
from __future__ import annotations

import sys
import unittest
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import itertools

from core import (
    PreferencePair,
    RolloutBatch,
    StepMetrics,
    Technique,
    TrainingExample,
    get_technique,
    list_techniques,
    register_technique,
)
from techniques._base.advantages import compute_group_advantages
from techniques._base.ratio_loss import RatioLossType
from techniques._base.simulator import simulate_decay
from techniques.dpo import DPO, DPOConfigV2, DPOTechnique
from techniques.grpo import GRPO, GRPOConfigV2, GRPOTechnique
from techniques.orpo import ORPO, ORPOConfigV2, ORPOTechnique
from techniques.ppo import PPO, PPOConfigV2, PPOTechnique


class TestCoreTypes(unittest.TestCase):
    def test_training_example_is_frozen(self):
        from dataclasses import FrozenInstanceError

        ex = TrainingExample(prompt="hi", response="hello")
        with self.assertRaises(FrozenInstanceError):
            ex.prompt = "bye"  # type: ignore[misc]

    def test_preference_pair_metadata_default_empty(self):
        p = PreferencePair(prompt="p", chosen="c", rejected="r")
        self.assertEqual(dict(p.metadata), {})

    def test_step_metrics_minimal(self):
        m = StepMetrics(loss=1.23, step=5)
        self.assertEqual(m.step, 5)
        self.assertAlmostEqual(m.loss, 1.23)
        self.assertEqual(dict(m.extras), {})


class TestRegistry(unittest.TestCase):
    def test_grpo_self_registers(self):
        names = list_techniques()
        self.assertIn("grpo", names)
        self.assertIs(get_technique("grpo"), GRPOTechnique)

    def test_duplicate_registration_raises(self):
        @register_technique("__unit_test_dup_1")
        class _A:
            name = "_A"

        with self.assertRaises(ValueError):
            register_technique("__unit_test_dup_1")(type("_B", (), {"name": "_B"}))

    def test_replace_flag_overrides(self):
        @register_technique("__unit_test_replace", replace=True)
        class _A:
            name = "_A"

        @register_technique("__unit_test_replace", replace=True)
        class _B:
            name = "_B"

        self.assertIs(get_technique("__unit_test_replace"), _B)

    def test_unknown_lookup_message_lists_alternatives(self):
        with self.assertRaises(KeyError) as ctx:
            get_technique("doesnotexist")
        self.assertIn("grpo", str(ctx.exception))


class TestConfigValidation(unittest.TestCase):
    def test_grpo_config_defaults(self):
        cfg = GRPOConfigV2()
        self.assertEqual(cfg.group_size, 8)
        self.assertEqual(cfg.reward_baseline, "group_mean")

    def test_grpo_config_rejects_extras(self):
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            GRPOConfigV2(typo_field=42)  # type: ignore[call-arg]

    def test_grpo_config_rejects_bad_baseline(self):
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            GRPOConfigV2(reward_baseline="totally_made_up")

    def test_grpo_config_yaml_roundtrip(self):
        cfg = GRPOConfigV2(group_size=4, kl_coef=0.25)
        y = cfg.to_yaml()
        self.assertIn("group_size", y)
        cfg2 = GRPOConfigV2.from_dict({"group_size": 4, "kl_coef": 0.25})
        self.assertEqual(cfg, cfg2)

    def test_grpo_config_is_frozen(self):
        from pydantic import ValidationError
        cfg = GRPOConfigV2()
        with self.assertRaises(ValidationError):
            cfg.group_size = 999  # type: ignore[misc]


class TestGRPOConformsToProtocol(unittest.TestCase):
    def test_isinstance_protocol(self):
        # runtime_checkable Protocol — checks method names only.
        self.assertIsInstance(GRPOTechnique(), Technique)

    def test_step_returns_step_metrics(self):
        impl = GRPOTechnique()
        impl.prepare(model=object(), tokenizer=object(), cfg=None)
        # No torch tensors → simulation path. Should still return StepMetrics.
        batch = RolloutBatch(prompts=[], responses=[], rewards=[])
        m = impl.step(batch)
        self.assertIsInstance(m, StepMetrics)
        self.assertEqual(m.step, 1)
        self.assertGreater(m.loss, 0)
        self.assertIn("reward", m.extras)

    def test_step_increments(self):
        impl = GRPOTechnique()
        batch = RolloutBatch(prompts=[], responses=[], rewards=[])
        impl.step(batch)
        impl.step(batch)
        m3 = impl.step(batch)
        self.assertEqual(m3.step, 3)

    def test_save_before_prepare_raises(self):
        with self.assertRaises(RuntimeError):
            GRPOTechnique().save("/tmp/nope")


class TestSharedHelpers(unittest.TestCase):
    def test_compute_group_advantages_python(self):
        adv = compute_group_advantages([1.0, 2.0, 3.0])
        # Centered, so the mean should be ~0
        self.assertAlmostEqual(sum(adv) / len(adv), 0.0, places=5)

    def test_simulate_decay_monotone(self):
        values = [simulate_decay(i) for i in range(10)]
        for a, b in itertools.pairwise(values):
            self.assertGreaterEqual(a, b)

    def test_ratio_loss_type_enum(self):
        self.assertEqual(RatioLossType.SIGMOID.value, "sigmoid")
        self.assertEqual(len(RatioLossType), 4)


class TestLegacyShimsStillWork(unittest.TestCase):
    """The old `GRPO`/`GRPOConfig` API must keep working for one minor version."""

    def test_grpo_legacy_instantiate(self):
        g = GRPO()
        self.assertEqual(g.name, "grpo")
        self.assertEqual(g.priority, 1)

    def test_grpo_legacy_compute_loss_simulation(self):
        g = GRPO()
        loss = g.compute_loss(epoch=0)
        self.assertIsInstance(loss, float)
        self.assertGreater(loss, 0)

    def test_grpo_legacy_train_step_returns_metrics(self):
        g = GRPO()
        m = g.train_step()
        self.assertIn("loss", m)
        self.assertEqual(m["step"], 1)


class TestDPOMigration(unittest.TestCase):
    """Phase 2: DPO must follow the same contract as GRPO."""

    def test_dpo_self_registers(self):
        self.assertIs(get_technique("dpo"), DPOTechnique)

    def test_dpo_config_rejects_extras(self):
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            DPOConfigV2(typo_field=42)  # type: ignore[call-arg]

    def test_dpo_config_rejects_bad_loss_type(self):
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            DPOConfigV2(loss_type="totally_made_up")

    def test_dpo_isinstance_protocol(self):
        self.assertIsInstance(DPOTechnique(), Technique)

    def test_dpo_step_returns_step_metrics_and_increments(self):
        impl = DPOTechnique()
        impl.prepare(model=object(), tokenizer=object(), cfg=None)
        pair = PreferencePair(prompt="p", chosen="c", rejected="r")
        m1 = impl.step(pair)
        self.assertIsInstance(m1, StepMetrics)
        self.assertEqual(m1.step, 1)
        self.assertGreater(m1.loss, 0)
        self.assertIn("chosen_reward", m1.extras)
        m2 = impl.step(pair)
        self.assertEqual(m2.step, 2)

    def test_dpo_legacy_compute_loss_matches_formula(self):
        # Legacy formula at epoch=0: 1.5 * exp(0) + 0.25 = 1.75
        d = DPO()
        loss = d.compute_loss(epoch=0)
        self.assertIsInstance(loss, float)
        self.assertAlmostEqual(loss, 1.75, places=6)


class TestORPOMigration(unittest.TestCase):
    """Phase 2: ORPO conforms to the same contract."""

    def test_orpo_self_registers(self):
        self.assertIs(get_technique("orpo"), ORPOTechnique)

    def test_orpo_config_rejects_extras(self):
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            ORPOConfigV2(typo_field=42)  # type: ignore[call-arg]

    def test_orpo_config_rejects_negative_lambda_or(self):
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            ORPOConfigV2(lambda_or=-0.5)

    def test_orpo_isinstance_protocol(self):
        self.assertIsInstance(ORPOTechnique(), Technique)

    def test_orpo_step_returns_step_metrics_and_increments(self):
        impl = ORPOTechnique()
        impl.prepare(model=object(), tokenizer=object(), cfg=None)
        pair = PreferencePair(prompt="p", chosen="c", rejected="r")
        m1 = impl.step(pair)
        self.assertIsInstance(m1, StepMetrics)
        self.assertEqual(m1.step, 1)
        self.assertGreater(m1.loss, 0)
        m2 = impl.step(pair)
        self.assertEqual(m2.step, 2)

    def test_orpo_legacy_shim_returns_float(self):
        loss = ORPO().compute_loss(epoch=0)
        self.assertIsInstance(loss, float)
        self.assertGreater(loss, 0)


class TestPPOMigration(unittest.TestCase):
    """Phase 2: PPO conforms to the same contract (RolloutBatch input)."""

    def test_ppo_self_registers(self):
        self.assertIs(get_technique("ppo"), PPOTechnique)

    def test_ppo_config_rejects_extras(self):
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            PPOConfigV2(typo_field=42)  # type: ignore[call-arg]

    def test_ppo_config_rejects_bad_clip_ratio(self):
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            PPOConfigV2(clip_ratio=2.0)  # must be in (0, 1)

    def test_ppo_isinstance_protocol(self):
        self.assertIsInstance(PPOTechnique(), Technique)

    def test_ppo_step_returns_step_metrics_and_increments(self):
        impl = PPOTechnique()
        impl.prepare(model=object(), tokenizer=object(), cfg=None)
        batch = RolloutBatch(prompts=[], responses=[], rewards=[])
        m1 = impl.step(batch)
        self.assertIsInstance(m1, StepMetrics)
        self.assertEqual(m1.step, 1)
        self.assertGreater(m1.loss, 0)

    def test_ppo_legacy_shim_returns_float(self):
        loss = PPO().compute_loss(epoch=0)
        self.assertIsInstance(loss, float)
        self.assertGreater(loss, 0)


class TestNoStubsRemain(unittest.TestCase):
    """Every technique now ships a real loss — nothing is experimental.

    SimPO/IPO graduated by wiring `pairwise_ratio_loss`; SPIN graduated with
    the same loss plus `build_spin_pairs` and iteration bookkeeping; RLAIF
    graduated with the `Judge` protocol and `label_pairs`.
    """

    def test_no_technique_is_experimental(self):
        from techniques import TECHNIQUE_REGISTRY
        for name, cls in TECHNIQUE_REGISTRY.items():
            self.assertFalse(cls.is_experimental, f"{name} should NOT be experimental")

    def test_no_technique_warns_on_instantiation(self):
        from techniques import TECHNIQUE_REGISTRY
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            for cls in TECHNIQUE_REGISTRY.values():
                cls()
            futures = [x for x in w if issubclass(x.category, FutureWarning)]
            self.assertEqual(futures, [])

    def test_graduated_techniques_registered_in_core_registry(self):
        from core.registry import get_technique
        from techniques.ipo import IPOTechnique
        from techniques.rlaif import RLAIFTechnique
        from techniques.simpo import SimPOTechnique
        from techniques.spin import SPINTechnique
        self.assertIs(get_technique("simpo"), SimPOTechnique)
        self.assertIs(get_technique("ipo"), IPOTechnique)
        self.assertIs(get_technique("spin"), SPINTechnique)
        self.assertIs(get_technique("rlaif"), RLAIFTechnique)

    def test_graduated_protocol_step_simulation(self):
        from core.types import PreferencePair
        from techniques.ipo import IPOTechnique
        from techniques.rlaif import RLAIFTechnique
        from techniques.simpo import SimPOTechnique
        from techniques.spin import SPINTechnique
        for cls in (SimPOTechnique, IPOTechnique, SPINTechnique, RLAIFTechnique):
            impl = cls()
            pair = PreferencePair(prompt="p", chosen="a", rejected="b")
            m = impl.step(pair)
            self.assertEqual(m.step, 1)
            self.assertGreater(m.loss, 0)


class TestSPINPairConstruction(unittest.TestCase):
    def test_build_spin_pairs(self):
        from techniques.spin import build_spin_pairs
        pairs = build_spin_pairs(
            prompts=["q1", "q2"],
            human_responses=["h1", "h2"],
            generated_responses=["g1", "g2"],
            iteration=1,
        )
        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs[0].chosen, "h1")
        self.assertEqual(pairs[0].rejected, "g1")
        self.assertEqual(pairs[0].metadata["spin_iteration"], 1)

    def test_build_spin_pairs_length_mismatch(self):
        from techniques.spin import build_spin_pairs
        with self.assertRaises(ValueError):
            build_spin_pairs(["q1"], ["h1", "h2"], ["g1"])

    def test_iteration_bookkeeping(self):
        from techniques.spin import SPINConfigV2, SPINTechnique
        impl = SPINTechnique(SPINConfigV2(num_iterations=2))
        self.assertEqual(impl.iteration, 0)
        self.assertEqual(impl.advance_iteration(), 1)
        with self.assertRaises(RuntimeError):
            impl.advance_iteration()


class TestRLAIFJudge(unittest.TestCase):
    def test_score_judge_labels_pairs(self):
        from techniques.rlaif import ScoreJudge, label_pairs
        judge = ScoreJudge(lambda prompt, response: len(response))
        pairs = label_pairs(judge, "q", ["short", "a longer response", "mid one"])
        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs[0].chosen, "a longer response")
        self.assertEqual(pairs[0].rejected, "short")
        self.assertEqual(pairs[0].metadata["labeler"], "ai")
        self.assertEqual(pairs[0].metadata["judge"], "ScoreJudge")

    def test_label_pairs_needs_two_candidates(self):
        from techniques.rlaif import ScoreJudge, label_pairs
        judge = ScoreJudge(lambda prompt, response: 0.0)
        with self.assertRaises(ValueError):
            label_pairs(judge, "q", ["only one"])

    def test_bad_judge_return_value_rejected(self):
        from techniques.rlaif import label_pairs
        with self.assertRaises(ValueError):
            label_pairs(lambda p, a, b: 2, "q", ["a", "b"])

    def test_technique_without_judge_raises_on_label(self):
        from techniques.rlaif import RLAIFTechnique
        with self.assertRaises(RuntimeError):
            RLAIFTechnique().label("q", ["a", "b"])


if __name__ == "__main__":
    unittest.main()
