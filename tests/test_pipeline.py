"""Tests for pipeline configuration and orchestration."""

import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.config import PipelineConfig


class TestPipelineConfig(unittest.TestCase):
    def test_default_config(self):
        config = PipelineConfig()
        self.assertEqual(config.technique, "grpo")
        self.assertEqual(config.epochs, 3)

    def test_validation(self):
        config = PipelineConfig(technique="invalid")
        errors = config.validate()
        self.assertTrue(len(errors) > 0)

    def test_valid_config(self):
        config = PipelineConfig(technique="grpo", epochs=3)
        errors = config.validate()
        self.assertEqual(len(errors), 0)

    def test_presets(self):
        for name in ["quick_dpo", "full_rlhf", "efficient_grpo", "research_spo", "production"]:
            config = PipelineConfig.preset(name)
            self.assertIsNotNone(config)
            self.assertEqual(len(config.validate()), 0)

    def test_to_dict(self):
        config = PipelineConfig()
        d = config.to_dict()
        self.assertIn("technique", d)
        self.assertIn("model_name", d)


if __name__ == "__main__":
    unittest.main()
