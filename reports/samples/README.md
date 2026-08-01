# Sample outputs

One file per agent. Each shows what that agent produces *in isolation* so
you can tell at a glance what the agent is for. All samples are ≤ 30
lines — the framework's attention-budget rule applies to its own docs.

| Agent | File | What it produces |
|-------|------|------------------|
| Supervisor | [`supervisor.md`](supervisor.md) | Plan choice + rationale + interventions |
| Trajectory | [`trajectory.md`](trajectory.md) | Rollout stats + curated set metrics |
| RewardModel | [`reward_model.md`](reward_model.md) | RM accuracy + calibration |
| AgenticTrainer | [`agentic_trainer.md`](agentic_trainer.md) | Iter-by-iter loss / reward / success |
| Evaluator | [`evaluator.md`](evaluator.md) | Benchmark table + recommendation |
| Reporter | [`reporter.md`](reporter.md) | The TL;DR the reporter emits |
