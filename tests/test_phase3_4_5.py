"""Phase 3 / 4 / 5 contract tests: YAML, LocalBackend, CLI, demo run-spec."""
from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Side-effect imports populate the registry.
import backends  # noqa: F401
import techniques  # noqa: F401
from backends.local import LocalBackend, make_job
from core import Backend, get_backend, list_backends
from core.types import JobStatus
from pipeline.cli import main as cli_main
from pipeline.config import PipelineConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO_YAML = REPO_ROOT / "examples" / "orpo_quickstart" / "run.yaml"


class TestYamlRoundTrip(unittest.TestCase):
    def test_to_yaml_then_from_yaml(self):
        cfg = PipelineConfig(technique="orpo", epochs=5, batch_size=2)
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
            fh.write(cfg.to_yaml())
            path = fh.name
        try:
            loaded = PipelineConfig.from_yaml(path)
            self.assertEqual(loaded.technique, "orpo")
            self.assertEqual(loaded.epochs, 5)
            self.assertEqual(loaded.batch_size, 2)
        finally:
            Path(path).unlink()

    def test_from_yaml_rejects_typos(self):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
            fh.write("technique: dpo\ntypo_field: 99\n")
            path = fh.name
        try:
            with self.assertRaises(ValueError) as ctx:
                PipelineConfig.from_yaml(path)
            self.assertIn("typo_field", str(ctx.exception))
        finally:
            Path(path).unlink()


class TestLocalBackend(unittest.TestCase):
    def test_backend_self_registers(self):
        self.assertIn("local", list_backends())
        self.assertIs(get_backend("local"), LocalBackend)

    def test_conforms_to_protocol(self):
        self.assertIsInstance(LocalBackend(), Backend)

    def test_dry_run_returns_zero_cost_plan(self):
        b = LocalBackend()
        job = make_job(technique="orpo", model_name="tiny")
        plan = b.dry_run(job)
        self.assertEqual(plan["cost_estimate_usd"], 0.0)
        self.assertEqual(plan["technique"], "orpo")
        self.assertEqual(plan["technique_class"], "ORPOTechnique")

    def test_launch_completes_simulation(self):
        b = LocalBackend()
        job = make_job(technique="grpo", technique_config={"epochs": 2})
        handle = b.launch(job)
        self.assertEqual(handle.status, JobStatus.COMPLETED)
        self.assertEqual(handle.backend, "local")
        self.assertIn("final_loss", handle.metrics)
        self.assertEqual(handle.metrics["epochs_completed"], 2.0)

    def test_launch_unknown_technique_fails_cleanly(self):
        b = LocalBackend()
        job = make_job(technique="doesnotexist")
        handle = b.launch(job)
        self.assertEqual(handle.status, JobStatus.FAILED)
        self.assertIn("doesnotexist", handle.error or "")


class TestCLI(unittest.TestCase):
    def _capture(self, argv):
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
        try:
            rc = cli_main(argv)
            return rc, sys.stdout.getvalue(), sys.stderr.getvalue()
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

    def test_list_techniques(self):
        rc, out, _ = self._capture(["list", "techniques"])
        self.assertEqual(rc, 0)
        # Only the 4 Protocol-conforming techniques (Phase 1 + Phase 2 migrations)
        # appear here; legacy-only `BaseTechnique` stubs live in the legacy
        # `techniques.TECHNIQUE_REGISTRY` dict but not the new registry.
        for name in ("grpo", "dpo", "orpo", "ppo"):
            self.assertIn(name, out)

    def test_list_backends(self):
        rc, out, _ = self._capture(["list", "backends"])
        self.assertEqual(rc, 0)
        self.assertIn("local", out)

    def test_inspect_known_technique(self):
        rc, out, _ = self._capture(["inspect", "grpo"])
        self.assertEqual(rc, 0)
        data = json.loads(out)
        self.assertEqual(data["name"], "grpo")
        self.assertFalse(data["experimental"])

    def test_inspect_unknown_technique_fails(self):
        rc, _, err = self._capture(["inspect", "nope"])
        self.assertEqual(rc, 2)
        self.assertIn("nope", err)

    def test_dry_run_demo_yaml(self):
        rc, out, _ = self._capture([
            "run", str(DEMO_YAML), "--backend", "local", "--dry-run",
        ])
        self.assertEqual(rc, 0)
        plan = json.loads(out)
        self.assertEqual(plan["technique"], "orpo")
        self.assertEqual(plan["cost_estimate_usd"], 0.0)

    def test_run_demo_yaml_succeeds(self):
        rc, out, _ = self._capture([
            "run", str(DEMO_YAML), "--backend", "local",
        ])
        self.assertEqual(rc, 0)
        data = json.loads(out)
        self.assertEqual(data["status"], "completed")
        self.assertIn("final_loss", data["metrics"])


class TestDemoArtifactsExist(unittest.TestCase):
    """The Phase 5 demo files must exist and be in sync with the code."""

    def test_run_yaml_loads(self):
        cfg = PipelineConfig.from_yaml(DEMO_YAML)
        self.assertEqual(cfg.technique, "orpo")
        self.assertGreater(cfg.epochs, 0)

    def test_readme_exists(self):
        readme = REPO_ROOT / "examples" / "orpo_quickstart" / "README.md"
        self.assertTrue(readme.exists())
        self.assertIn("agentic-train", readme.read_text())

    def test_train_script_exists(self):
        train = REPO_ROOT / "examples" / "orpo_quickstart" / "train.py"
        self.assertTrue(train.exists(), "examples/orpo_quickstart/train.py must exist")

    def test_makefile_exists(self):
        mk = REPO_ROOT / "Makefile"
        self.assertTrue(mk.exists())
        body = mk.read_text()
        for target in ("install", "test", "demo", "demo-dry"):
            self.assertIn(target + ":", body)


class TestStandaloneTrainScript(unittest.TestCase):
    """The standalone `train.py` is the programmatic counterpart of the CLI.
    It must produce the same shape of result and write `metrics.json`."""

    def test_run_returns_completed_handle(self):
        from examples.orpo_quickstart.train import run

        with tempfile.TemporaryDirectory() as tmp:
            spec = Path(tmp) / "spec.yaml"
            cfg = PipelineConfig.from_yaml(DEMO_YAML)
            cfg.output_dir = tmp  # avoid writing into repo
            spec.write_text(cfg.to_yaml())
            result = run(spec, dry_run=False)

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["backend"], "local")
        self.assertIn("final_loss", result["metrics"])

    def test_run_dry_run_returns_plan(self):
        from examples.orpo_quickstart.train import run

        result = run(DEMO_YAML, dry_run=True)
        self.assertEqual(result["cost_estimate_usd"], 0.0)
        self.assertEqual(result["technique"], "orpo")

    def test_run_writes_metrics_json(self):
        from examples.orpo_quickstart.train import run

        with tempfile.TemporaryDirectory() as tmp:
            spec = Path(tmp) / "spec.yaml"
            cfg = PipelineConfig.from_yaml(DEMO_YAML)
            cfg.output_dir = tmp
            spec.write_text(cfg.to_yaml())
            run(spec, dry_run=False)

            metrics_path = Path(tmp) / "metrics.json"
            self.assertTrue(metrics_path.exists())
            payload = json.loads(metrics_path.read_text())
            self.assertEqual(payload["status"], "completed")


if __name__ == "__main__":
    unittest.main()
