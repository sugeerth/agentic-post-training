"""Standalone ORPO quickstart — programmatic entry point.

Mirrors `agentic-train run examples/orpo_quickstart/run.yaml --backend local`
but does it via the public Python API so the moving parts are visible in
one screen of code:

    PipelineConfig.from_yaml  ──►  TrainingJob  ──►  LocalBackend.launch
                                                          │
                                                          ▼
                                                  JobHandle (status, metrics)

Run it as a module so the side-effect imports populate the registries:

    python3 -m examples.orpo_quickstart.train
    python3 -m examples.orpo_quickstart.train --dry-run
    python3 -m examples.orpo_quickstart.train --run-spec path/to/other.yaml

The backend is `LocalBackend` and currently exercises ORPO's deterministic
simulation path (so the script runs without a GPU, without HF Hub access,
and without `torch`). Wiring this to real `transformers + peft` training is
the next milestone; the contract — `LocalBackend.launch(TrainingJob) →
JobHandle` — does not change when that lands.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Side-effect imports register backends + techniques. Order matters: the
# registries are empty until each module loads.
import backends  # noqa: F401  - registers LocalBackend under "local"
import techniques  # noqa: F401  - registers every priority-1/2/3 technique
from backends.local import make_job
from core.registry import get_backend
from pipeline.config import PipelineConfig

DEFAULT_SPEC = Path(__file__).with_name("run.yaml")


def run(spec_path: Path, *, dry_run: bool = False, backend_name: str = "local") -> dict:
    """Load `spec_path`, dispatch to `backend_name`, return the result dict.

    On success the result includes `metrics`. On dry-run it includes the
    full plan instead. Either way the dict is JSON-serializable so the
    caller can write it out or pipe it to another tool.
    """
    cfg = PipelineConfig.from_yaml(spec_path)
    errs = cfg.validate()
    if errs:
        raise ValueError(f"Run-spec failed validation: {errs}")

    backend = get_backend(backend_name)()

    job = make_job(
        technique=cfg.technique,
        model_name=cfg.model_name,
        output_dir=cfg.output_dir,
        technique_config={
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "learning_rate": cfg.learning_rate,
            "max_length": cfg.max_length,
            **cfg.technique_config,
        },
    )

    if dry_run:
        return backend.dry_run(job)

    handle = backend.launch(job)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"

    result = {
        "job_id": handle.job_id,
        "status": handle.status.value,
        "backend": handle.backend,
        "last_checkpoint": handle.last_checkpoint,
        "metrics": handle.metrics,
        "error": handle.error,
    }
    metrics_path.write_text(json.dumps(result, indent=2, default=str))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ORPO quickstart")
    parser.add_argument(
        "--run-spec",
        type=Path,
        default=DEFAULT_SPEC,
        help=f"Path to the run-spec YAML (default: {DEFAULT_SPEC})",
    )
    parser.add_argument("--backend", default="local", help="Backend to dispatch to (default: local)")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan; spend nothing")
    args = parser.parse_args(argv)

    result = run(args.run_spec, dry_run=args.dry_run, backend_name=args.backend)
    print(json.dumps(result, indent=2, default=str))
    return 0 if result.get("status", "completed") == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
