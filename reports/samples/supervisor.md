# Sample: SupervisorAgent

**Owns:** which recipe to run, when to intervene.

```
Supervisor plan: agentic-tool-use
  Why:    Kimi K2 / tool-agent recipe: sample multi-turn tool-use trajectories,
          reward on task success, PPO-style clipped update.
  Stages: curate → trajectory → rm → multi_turn_grpo → eval
  Config: technique=multi_turn_grpo, reward_kind=process
```

After the run, the supervisor logs one Intervention per stage:

```json
[
  {"stage": "curate",          "reason": "metrics within expected envelope", "action": "continue"},
  {"stage": "rm",              "reason": "metrics within expected envelope", "action": "continue"},
  {"stage": "multi_turn_grpo", "reason": "KL=0.61 exceeds 0.5 — tighten kl_coef", "action": "adjust"},
  {"stage": "eval",            "reason": "metrics within expected envelope", "action": "continue"}
]
```

**What to notice:** an anomaly at `multi_turn_grpo` triggers an
`adjust` — the reporter surfaces this as a next action instead of
burying it in the log.
