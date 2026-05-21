"""Phase 2 contract tests: split coordinator, reporters, optimization reabsorption."""
from __future__ import annotations

import io
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.dashboard import render_dashboard
from agents.planner import PipelineStage, plan
from agents.reporters import (
    ConsoleReporter,
    JSONLReporter,
    NoopReporter,
)
from optimization.apply import (
    OptimizationPlan,
    OptimizationResult,
    apply_optimizations,
    plan_from_legacy_kwargs,
)
from optimization.quantization import QuantizationConfig

# ---------------------------------------------------------------------------- #
# Planner
# ---------------------------------------------------------------------------- #


class TestPlanner(unittest.TestCase):
    def test_default_plan_has_five_stages(self):
        stages = plan()
        self.assertEqual(len(stages), 5)
        names = [s.name for s in stages]
        self.assertEqual(
            names,
            ["data_prep", "technique_selection", "training", "optimization", "evaluation"],
        )

    def test_per_stage_config_overrides_merge_in(self):
        stages = plan({"training": {"epochs": 9, "lr": 1e-5}})
        train = next(s for s in stages if s.name == "training")
        self.assertEqual(train.config["epochs"], 9)
        self.assertEqual(train.config["lr"], 1e-5)

    def test_plan_is_idempotent_across_calls(self):
        a = plan({"training": {"epochs": 9}})
        b = plan({"training": {"epochs": 9}})
        self.assertEqual([s.config for s in a], [s.config for s in b])
        # Mutating one shouldn't bleed into the next call.
        a[0].config["mutated"] = True
        c = plan()
        self.assertNotIn("mutated", c[0].config)

    def test_plan_no_asyncio_needed(self):
        # The pure planner must not require an event loop. This test will
        # fail if `plan` is ever made async or imports asyncio at module top.
        stages = plan()
        self.assertIsInstance(stages, list)
        self.assertIsInstance(stages[0], PipelineStage)


# ---------------------------------------------------------------------------- #
# Dashboard
# ---------------------------------------------------------------------------- #


class _FakeAgent:
    """Minimal stand-in matching the duck-typed surface render_dashboard needs."""
    def __init__(self, name: str, role: str) -> None:
        from agents.base_agent import AgentStatus
        self.name = name
        self.role = role
        self.status = AgentStatus.IDLE
        self.capabilities: list = []
        self._color = ""


class TestDashboard(unittest.TestCase):
    def test_render_returns_string(self):
        agents = {"Coordinator": _FakeAgent("Coordinator", "coordinator")}
        stages = plan()
        out = render_dashboard(agents, stages)
        self.assertIsInstance(out, str)
        self.assertIn("Pipeline Dashboard", out)
        self.assertIn("training", out)
        self.assertIn("Coordinator", out)

    def test_render_handles_empty_stages(self):
        agents = {"x": _FakeAgent("x", "trainer")}
        out = render_dashboard(agents, [])
        self.assertIn("Pipeline Dashboard", out)
        self.assertNotIn("Pipeline Stages:", out)


# ---------------------------------------------------------------------------- #
# Reporters
# ---------------------------------------------------------------------------- #


class TestReporters(unittest.TestCase):
    def test_noop_reporter_is_silent(self):
        # Even if we pass garbage, nothing should raise.
        NoopReporter().emit("anything", {"k": "v", "nested": [1, 2, 3]})

    def test_console_reporter_writes_event_name(self):
        buf = io.StringIO()
        ConsoleReporter(stream=buf).emit("stage_start", {"stage": "training"})
        out = buf.getvalue()
        self.assertIn("stage_start", out)
        self.assertIn("training", out)

    def test_jsonl_reporter_to_file(self):
        import json
        import tempfile
        with tempfile.NamedTemporaryFile("w+", suffix=".jsonl", delete=False) as fh:
            path = fh.name
        try:
            r = JSONLReporter(path)
            r.emit("a", {"x": 1})
            r.emit("b", {"y": "hello"})
            r.close()
            lines = Path(path).read_text().strip().splitlines()
            self.assertEqual(len(lines), 2)
            first = json.loads(lines[0])
            self.assertEqual(first["event"], "a")
            self.assertEqual(first["payload"]["x"], 1)
            self.assertIn("ts", first)
        finally:
            Path(path).unlink()


# ---------------------------------------------------------------------------- #
# Optimization reabsorption
# ---------------------------------------------------------------------------- #


class TestOptimizationReabsorption(unittest.TestCase):
    def test_apply_optimizations_handles_quantization(self):
        plan_ = OptimizationPlan(quantization=QuantizationConfig(method="nf4", bits=4))
        _, result = apply_optimizations("model_handle", plan_)
        self.assertIsInstance(result, OptimizationResult)
        self.assertIsNotNone(result.quantization)
        self.assertEqual(result.quantization["method"], "nf4")
        self.assertEqual(result.quantization["bits"], 4)

    def test_empty_plan_is_a_no_op(self):
        _, result = apply_optimizations("model", OptimizationPlan())
        self.assertEqual(result.summary(), {})

    def test_legacy_kwargs_to_plan(self):
        p = plan_from_legacy_kwargs(method="quantization", quant_type="gptq")
        self.assertIsNotNone(p.quantization)
        self.assertEqual(p.quantization.method, "gptq")

        p2 = plan_from_legacy_kwargs(method="pruning", pruning_type="wanda", sparsity=0.7)
        self.assertIsNotNone(p2.pruning)
        self.assertEqual(p2.pruning.method, "wanda")
        self.assertEqual(p2.pruning.sparsity, 0.7)

    def test_optimization_agent_uses_apply(self):
        # The whole point of the refactor: agent delegates to the package.
        import inspect

        from agents import optimization_agent
        src = inspect.getsource(optimization_agent)
        self.assertIn("apply_optimizations", src)
        # Inline class-level dispatch dict is gone (the name may still appear
        # in the docstring describing what was removed — that's fine).
        self.assertNotIn("QUANTIZATION_METHODS = {", src)
        self.assertNotIn("PRUNING_METHODS = {", src)


if __name__ == "__main__":
    unittest.main()
