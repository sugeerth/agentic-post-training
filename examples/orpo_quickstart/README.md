# ORPO Quickstart

End-to-end smoke test of the agentic-post-training library, runnable on any
laptop in under a second.

## What this demos

- The `core.Technique` Protocol + `LocalBackend`.
- A YAML run-spec (`run.yaml`) as the single source of truth for a run.
- Two equivalent entry points: the `agentic-train` CLI and a standalone
  `train.py` script you can read top-to-bottom.

## 30-second tour

```bash
# Sanity check the plan (free, no compute spent)
python3 -m pipeline.cli run examples/orpo_quickstart/run.yaml --backend local --dry-run

# Run it (currently exercises ORPO's simulation path — no GPU, no torch needed)
python3 -m pipeline.cli run examples/orpo_quickstart/run.yaml --backend local

# Same thing, programmatically (one screen of code; useful as a template)
python3 -m examples.orpo_quickstart.train --dry-run
python3 -m examples.orpo_quickstart.train

# Inspect the technique registry
python3 -m pipeline.cli list techniques
python3 -m pipeline.cli inspect orpo
```

After `pip install -e .` the same commands work via the `agentic-train`
console script (e.g. `agentic-train run examples/orpo_quickstart/run.yaml --backend local`).

## Expected output

`--dry-run` returns the resolved plan (no compute spent):

```json
{
  "backend": "local",
  "cost_estimate_usd": 0.0,
  "technique": "orpo",
  "technique_class": "ORPOTechnique",
  "model": "sshleifer/tiny-gpt2",
  "dataset": "ultrafeedback",
  "output_dir": "./output/orpo_quickstart",
  "technique_config": {
    "epochs": 3, "batch_size": 1, "learning_rate": 1e-05,
    "max_length": 512, "lambda_or": 0.1, "beta": 0.1
  }
}
```

A full run returns a `JobHandle`-shaped result and writes the same payload
to `<output_dir>/metrics.json`:

```json
{
  "status": "completed",
  "backend": "local",
  "last_checkpoint": "./output/orpo_quickstart/sim-<job_id>.pt",
  "metrics": {
    "final_loss": 0.7277,
    "epochs_completed": 3.0,
    "wall_seconds": 1.1e-05,
    "final_sft_loss": 0.5094,
    "final_or_loss": 0.2183,
    "final_log_odds_chosen_mean": -0.05,
    "final_log_odds_rejected_mean": -0.56,
    "final_accuracy": 0.65
  }
}
```

Numbers are deterministic — the simulation seeds itself from the technique
defaults so they match the values in `tests/test_phase3_4_5.py`.

## What this does NOT do (yet)

`LocalBackend` currently calls `ORPOTechnique.step(empty_batch)` in a loop,
not real torch training. The contract — `LocalBackend.launch(TrainingJob) →
JobHandle` — is fixed; only the body of `launch` changes when real training
lands. To upgrade in place:

1. Swap `model_name` in `run.yaml` to `Qwen/Qwen2.5-0.5B-Instruct`.
2. Add an `ultrafeedback` loader under `data/loaders/` (greenfield; see
   C2 in `REFACTOR_PLAN.md`).
3. Run the same command on a GPU box (Colab T4 / Vast.ai / local CUDA).

Colab, Kaggle, and Vast.ai backends will share this exact `run.yaml` and
only differ in how they get the job to hardware.
