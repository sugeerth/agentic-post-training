# Sample: AgenticTrainingAgent

**Owns:** credit assignment across turns and KL to the reference agent
policy. Consumes curated trajectories + RM scores, emits one metric
line per RL iteration.

```
[AgenticTrainer] Starting multi_turn_grpo — 3 iters, group=8, kl_coef=0.05, curated_rollouts=32
iter 1/3 | loss=1.421 | reward=0.540 | success=46% | kl=0.135
iter 2/3 | loss=1.117 | reward=0.740 | success=57% | kl=0.110
iter 3/3 | loss=0.898 | reward=0.940 | success=68% | kl=0.085
multi_turn_grpo complete — success 68% (+33%)
```

Result payload:

```json
{
  "technique": "multi_turn_grpo",
  "iterations": 3,
  "final_loss": 0.898,
  "final_reward": 0.94,
  "kl_divergence": 0.085,
  "task_success_rate": 0.68,
  "success_rate_lift": 0.33
}
```

**What to notice:** the trainer reports `task_success_rate` — the
metric an *agent* is judged by — not just `loss`. Loss going down while
success stays flat is the classic reward-hacking failure mode.
