"""Local in-process backend.

The default backend — runs a `TrainingJob` in the current Python process
without spawning subprocesses or hitting any external service. Used for
tests, the Phase 5 quickstart demo, and any case where the user already
has the right hardware in front of them.

Today this backend exercises the technique's simulation path so it's safe
to run anywhere (no GPU, no HF Hub access). The hooks for real training
(model loading, optimizer, autograd) are clearly marked TODOs — wiring
them up is the Phase 5 demo's job once we know the demo's exact model
+ dataset shape.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from core.registry import get_technique, register_backend
from core.types import (
    JobHandle,
    JobStatus,
    RolloutBatch,
    StepMetrics,
    TrainingJob,
)


@register_backend("local")
class LocalBackend:
    """Conforms to `core.Backend`."""

    name = "local"

    def dry_run(self, job: TrainingJob) -> dict[str, Any]:
        """Print the plan without spending any compute. Always free.

        Returns a structured plan dict so callers (CLI, tests) can verify
        what would run without parsing stdout.
        """
        plan = {
            "backend": self.name,
            "cost_estimate_usd": 0.0,  # local runs are always free
            "job_id": job.job_id,
            "technique": job.technique,
            "model": job.model_name,
            "dataset": job.dataset,
            "output_dir": job.output_dir,
            "technique_config": dict(job.technique_config),
        }
        # Best-effort: confirm the technique exists in the registry.
        try:
            plan["technique_class"] = get_technique(job.technique).__name__
        except KeyError as e:
            plan["technique_class"] = f"<unknown: {e}>"
        return plan

    def launch(self, job: TrainingJob) -> JobHandle:
        """Run the job. Returns a JobHandle with final status.

        Uses the technique's simulation path for now (no real model
        download, no GPU dependency). This is enough to exercise the
        whole pipeline end-to-end and produce a metrics trace; a follow-up
        wires in real `transformers` + `peft` once the demo dataset is
        finalized.
        """
        handle = JobHandle(
            job_id=job.job_id,
            status=JobStatus.RUNNING,
            backend=self.name,
        )
        try:
            tech_cls = get_technique(job.technique)
        except KeyError as e:
            handle.status = JobStatus.FAILED
            handle.error = str(e)
            return handle

        technique = tech_cls()
        # `cfg=None` → technique uses its defaults; pipeline-level overrides
        # will be threaded through via `job.technique_config` once the
        # demo lands.
        technique.prepare(model=object(), tokenizer=object(), cfg=None)

        # Simulated rollout loop — replaced with real training in Phase 5.
        epochs = int(job.technique_config.get("epochs", 3))
        metrics_log: list[StepMetrics] = []
        empty_batch = RolloutBatch(prompts=[], responses=[], rewards=[])
        start = time.time()
        for _ in range(epochs):
            metrics = technique.step(empty_batch)
            metrics_log.append(metrics)

        handle.last_checkpoint = f"{job.output_dir}/sim-{job.job_id[:8]}.pt"
        handle.metrics = {
            "final_loss": metrics_log[-1].loss,
            "epochs_completed": float(epochs),
            "wall_seconds": time.time() - start,
        }
        handle.metrics.update({f"final_{k}": float(v) for k, v in metrics_log[-1].extras.items()})
        handle.status = JobStatus.COMPLETED
        return handle


def make_job(
    technique: str,
    *,
    model_name: str = "sshleifer/tiny-gpt2",
    dataset: str = "ultrafeedback",
    output_dir: str = "./output",
    technique_config: dict[str, Any] | None = None,
) -> TrainingJob:
    """Convenience: build a `TrainingJob` from common fields.

    Saves callers from importing `uuid` and `TrainingJob` directly.
    """
    return TrainingJob(
        job_id=uuid.uuid4().hex,
        technique=technique,
        technique_config=dict(technique_config or {}),
        model_name=model_name,
        dataset=dataset,
        output_dir=output_dir,
    )
