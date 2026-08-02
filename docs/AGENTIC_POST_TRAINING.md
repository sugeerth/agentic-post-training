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

## The autonomous closed loop — the core

`pipeline/autonomous_loop.py` is what everything else stages for. Fixed
pipelines run stages and stop; this loop closes the feedback that defines
agent post-training (expert iteration / iterated RFT / R1's multi-round
recipe):

```
collect(skill s) ──▶ curate ──▶ train ──▶ s'
     ▲                                    │
     └────── next round collects with s' ─┘
              probe s' on fresh env rollouts
              audit reward vs measured success
              supervisor decides — and its patch mutates the next round
```

Three properties make it agentic rather than a script:

1. **Closed feedback.** The policy that collects round *r+1*'s data is the
   policy improved in round *r*. Skill is state, not a schedule.
2. **Goal-driven termination.** `run(target_success=0.85, budget_rollouts=2000)`
   stops on target met, budget spent (checked *before* each round — never
   overspends), or genuine convergence. Never "ran out of stages."
3. **Real interventions.** Every non-continue supervisor decision carries a
   `config_patch` the loop applies: KL tightening (hard-capped at 0.4),
   rollback restores the best-known policy, exploration boosts change
   collection itself. Escalation is bounded — after 2 failed interventions
   on one no-gain streak the supervisor concludes "converged" and stops
   instead of thrashing. Measured behavior on a deliberately broken config
   (`learning_gain=0`): stops at round 11 of 30 with 80% of budget unspent.

```bash
python3 examples/run_autonomous.py --target-success 0.85 --budget 2000
# 🎯 target met: 88% ≥ 85% — best success 88% in 5 rounds, 440 rollouts
```

Ground truth discipline: decisions only ever read the probe (fresh env
rollouts at the updated skill) — never the training curve.

## Agent registry — swap any specialist without forking

`agents/registry.py` builds on `core.registry.Registry`. Every agent
self-registers under its role; pipelines construct by role name:

```python
from agents.registry import register_agent, build

@register_agent("agentic_trainer", replace=True)
class MyTrainer(BaseAgent): ...

trainer = build("agentic_trainer")   # returns MyTrainer
```

`AgenticPipeline` and `AutonomousLoop` both build their crews this way —
replacing the trainer, the reward model, or the supervisor is one
decorator, zero pipeline edits.

## Environment (real trajectories, not simulated dicts)

`environments/tool_env.py` is a small deterministic tool-use env:

- 4 tools — `search`, `calculate`, `lookup`, `finish(answer)`
- 4 task kinds — `sum`, `lookup`, `twohop`, `distract`
- Sparse terminal reward (1 if `finish(answer)` matches truth, else 0)
  plus small per-turn shaping (+0.05 for useful tool calls, −0.02 for
  dead-end queries)

Reference policies (`environments/policies.py`) give the trajectory agent
a knob: `p_expert=0.0` → 0% success, `p_expert=1.0` → 100% success. The
trainer's "policy improving" arc across iterations is a schedule on that
knob, not a math trick.

## Reward-hacking detector

New agent: `agents/reward_hacking_detector.py`. Watches for the classic
failure of agentic RL — the reward-model score climbs monotonically while
the eval score plateaus or drops. Two signals, deliberately auditable in
ten lines:

- **Spearman ρ** between per-iteration reward and per-iteration eval
- **gap** = (Δ reward) − (Δ eval)

ρ < 0 or gap > 0.25 → hacking flagged. The supervisor promotes a flagged
audit into a real intervention (`adjust` for medium, `rollback` for high),
which lands in the TL;DR's *Needs attention* section.

## Recipe bake-off

`agents/bakeoff_agent.py` + `pipeline/bakeoff_pipeline.py` + `examples/run_bakeoff.py`.

Bake-offs pick the winner by highest success rate *among plans that stayed
under a KL ceiling*. This is more honest than raw success rate — a plan
that hit 90% by drifting off-distribution is not the recipe you want to
ship. The reporter also prints the pareto frontier so you see the trade
between "chased reward" and "stayed close to the reference policy".

## Run history + regression detection

Each run persists to `.runs/{plan}-{n:04d}.json`. The reporter reads the
latest prior run of the same plan and prints a `## vs last run` block
with per-metric deltas. If task-success regressed by >5%, the reporter's
next-action becomes *"revert or investigate"* instead of *"ship the
checkpoint"* — the framework catches its own regressions.

## ASCII sparklines (attention-conscious visuals)

The reporter renders one-line unicode sparklines for reward / loss /
success / KL across iterations:

```
## Curves
- `reward  ` ▁▃▅▇█
- `loss    ` ▇▅▃▂▁
- `success ` ▁▂▅▆█
- `kl      ` ▇▆▄▂▁
```

No plot library, no browser, no scroll. The shape of the curve tells you
in one glance what a page of numbers can't.

## Extension

Add a plan: append a `RunPlan(...)` in `agents/supervisor_agent.py::SupervisorAgent.PLANS`.
Add a stage: extend `STAGE_SPEC` in `pipeline/agentic_pipeline.py`.
Add a technique: drop a class in `techniques/agentic/`, register it in
`AGENTIC_TECHNIQUES`.
Add an agent role: subclass `BaseAgent`, register with the coordinator.
Add an environment: subclass `ToolEnv`, or drop a new `env.py` alongside;
update `TrajectoryAgent` to instantiate it.

Everything else is machinery; those are the only files that need to
change to support new research.
