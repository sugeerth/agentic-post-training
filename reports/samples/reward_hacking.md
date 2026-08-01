# Sample: RewardHackingDetector

**Owns:** the biggest failure mode of agentic RL — reward climbing while
eval doesn't. Two signals, auditable in ten lines.

Aligned run:

```
{
  "spearman_rho": 0.949,
  "reward_eval_gap": 0.02,
  "severity": "low",
  "verdict": "reward/eval aligned",
  "n_iterations": 4
}
```

Hacked run — reward up 0.75, eval down 0.25 across four iterations:

```
{
  "spearman_rho": -1.0,
  "reward_eval_gap": 1.0,
  "severity": "high",
  "verdict": "reward hacking suspected",
  "n_iterations": 4
}
```

**What to notice:** the supervisor promotes `severity: high` to a
`rollback` intervention, which the reporter surfaces in the *Needs
attention* section of the TL;DR. The detector itself is not learned —
a learned hack detector is another RM that can itself be gamed.
