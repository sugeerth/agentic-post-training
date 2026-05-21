"""Tests for post-training techniques."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from techniques import PRIORITY_1, PRIORITY_2, PRIORITY_3, TECHNIQUE_REGISTRY
from techniques.base_technique import BaseTechnique


class TestTechniqueRegistry(unittest.TestCase):
    def test_all_techniques_registered(self):
        expected = ["ppo", "grpo", "spo", "dpo", "kto", "orpo", "rlhf", "rlaif", "spin", "simpo", "ipo"]
        for name in expected:
            self.assertIn(name, TECHNIQUE_REGISTRY, f"Missing technique: {name}")

    def test_priority_groups(self):
        self.assertEqual(len(PRIORITY_1), 5)
        self.assertEqual(len(PRIORITY_2), 5)
        self.assertEqual(len(PRIORITY_3), 1)

    def test_all_inherit_base(self):
        for name, cls in TECHNIQUE_REGISTRY.items():
            self.assertTrue(issubclass(cls, BaseTechnique), f"{name} doesn't inherit BaseTechnique")


class TestTechniqueInstantiation(unittest.TestCase):
    def test_instantiate_all(self):
        for name, cls in TECHNIQUE_REGISTRY.items():
            instance = cls()
            self.assertIsNotNone(instance, f"Failed to instantiate {name}")
            self.assertIsNotNone(instance.name)
            self.assertIsNotNone(instance.description)

    def test_info_method(self):
        for _name, cls in TECHNIQUE_REGISTRY.items():
            instance = cls()
            info = instance.info()
            self.assertIn("name", info)
            self.assertIn("description", info)
            self.assertIn("priority", info)


class TestTechniqueTraining(unittest.TestCase):
    def test_train_step_all(self):
        for name, cls in TECHNIQUE_REGISTRY.items():
            instance = cls()
            metrics = instance.train_step(epoch=1)
            self.assertIsInstance(metrics, dict, f"{name} train_step didn't return dict")
            self.assertIn("loss", metrics, f"{name} missing 'loss' in metrics")

    def test_compute_loss_simulation(self):
        for name, cls in TECHNIQUE_REGISTRY.items():
            instance = cls()
            loss = instance.compute_loss(epoch=1)
            self.assertIsNotNone(loss, f"{name} compute_loss returned None")

    def test_metrics_update(self):
        for _name, cls in TECHNIQUE_REGISTRY.items():
            instance = cls()
            instance.train_step(epoch=1)
            instance.train_step(epoch=2)
            metrics = instance.get_metrics()
            self.assertEqual(metrics["step"], 2)


if __name__ == "__main__":
    unittest.main()
