# Agentic Post-Training

Why a whole framework? Because post-training an **agent** is not the same
problem as post-training a base LLM, and treating it that way is the
number-one reason RL runs collapse.

## The core problem

A base LLM produces one response to one prompt. An agent produces a
**trajectory** — a sequence of `(assistant, tool, tool_result)` turns
that eventually succeeds or fails at a task. Everything about the
training loop changes:

| Axis | Base LLM PT | Agentic PT |
|------|-------------|------------|
| Unit of training | (prompt, response) pair | full trajectory |
| Reward density | per response | one terminal bit per trajectory (mostly) |
| Credit assignment | none needed | across many turns |
| Off-policy risk | low | high — the policy edits its own future rollouts |
| Failure mode | style drift | tool loops, hallucinated tool_results, mode collapse |
| Compute per gradient | one forward | many rollouts × many turns each |

If you ignore any of these, the run looks fine on loss curves and dies
on the eval.

## What each agent is for

Each agent owns exactly one hard problem. The supervisor picks a plan,
the coordinator schedules, the specialists execute.

| Agent | Problem it owns | Real-world analog |
|-------|-----------------|-------------------|
| **Supervisor** | Which recipe to run, and when to stop | The RL team lead |
| **Coordinator** | Stage plumbing, dispatch, dashboards | The training-infra harness |
| **TrajectoryAgent** | Rollout sparsity & noise | Data engineering / rejection sampling |
| **RewardModelAgent** | Reward hacking & mis-calibration | RM team (ORM + PRM) |
| **AgenticTrainingAgent** | Credit assignment across turns | The RL trainer |
| **EvaluationAgent** | Distribution shift & regression | Eval team |
| **ReporterAgent** | Attention overload | The weekly status doc |

## Recipes the supervisor knows

| Plan | Stages | Why |
|------|--------|-----|
| `agentic-tool-use` | curate → trajectory → rm → multi_turn_grpo → eval | Kimi K2 / tool-agent style. Trajectory-level GRPO on multi-turn tool use, scored by a process reward model. |
| `reasoning-r1` | curate → sft → rm → grpo → distill → eval | DeepSeek-R1: SFT cold-start, outcome-reward GRPO, distill reasoning traces down. |
| `preference-alignment` | curate → sft → dpo → eval | Cheap DPO when you already have preference pairs. |
| `rejection-sampling-loop` | trajectory → rft → eval | STaR / RFT. Sample, keep the good ones, SFT, repeat. |

## Techniques (agentic layer)

Files under `techniques/agentic/`. Each one is deliberately narrow — the
right technique depends on how sparse your reward is and how much
on-policy compute you can spend.

- **`multi_turn_grpo.py`** — GRPO where the "response" is a whole
  trajectory. Group-normalized advantages across G rollouts per task.
  Multi-turn credit assignment via per-turn log-prob aggregation.
  KL to a *reference agent policy* (not the base LM).
- **`trajectory_dpo.py`** — DPO over (winning-trajectory,
  losing-trajectory) pairs. No reward model needed. Offline. Weaker
  regularization than on-policy RL.
- **`rejection_sampling_ft.py`** — Sample K rollouts, keep successes,
  SFT on them, repeat. Collapses when initial success rate is near
  zero; mix in expert trajectories to fix.
- **`process_reward_model.py`** — Trains the PRM (step-level scores)
  used by `multi_turn_grpo` when the terminal signal is too sparse.

## Human-friendly output

The reporter is not decorative. Attention is the scarcest resource on
a training run.

- **6 lines above the fold**: headline (delta first), one metric per
  stage, "needs attention" list, one next action.
- **Anomalies-only detail**: if all stages are within the expected
  envelope, the reporter says so in one sentence and hides the log
  behind a `<details>` block.
- **Both formats**: `output/report.md` for humans, `output/report.json`
  for CI.

Example above-the-fold:

```
# TL;DR — agentic-tool-use

**Task success 76% (Δ +38%) via multi_turn_grpo.**

curated `32/64` · rm-acc `0.82` · reward `0.71` · success `76%`

_All stages within expected envelope. No manual review needed._

## Next action
- Ship the checkpoint. Nothing else to tune.
```

That's what a good run looks like. A bad run replaces the "Ship the
checkpoint" line with a specific stage to fix.

## CI/CD

`.github/workflows/post-training.yml` runs the pipeline on push and
uploads `output/report.md` as a build artifact — the PR reviewer sees
the TL;DR without pulling logs.

## Extension

Add a plan: append a `RunPlan(...)` in `agents/supervisor_agent.py::SupervisorAgent.PLANS`.
Add a stage: extend `STAGE_SPEC` in `pipeline/agentic_pipeline.py`.
Add a technique: drop a class in `techniques/agentic/`, register it in
`AGENTIC_TECHNIQUES`.
Add an agent role: subclass `BaseAgent`, register with the coordinator.

Everything else is machinery; those are the only files that need to
change to support new research.
