# Sample: TrajectoryAgent

**Owns:** rollout sparsity & noise. Samples multi-turn agent trajectories,
then curates via rejection sampling.

```
[TrajectoryCurator] Sampling 64 rollouts (baseline success ≈ 35%)
[TrajectoryCurator] Curated 32/64 rollouts (success 78%)
```

Result payload:

```json
{
  "sampled": 64,
  "success_rate": 0.344,
  "curated": 32,
  "curated_success_rate": 0.781,
  "avg_turns": 4.9,
  "avg_tokens_per_rollout": 392.0
}
```

**What to notice:** curation more than doubles the effective success
rate seen by the trainer (34% → 78%), which is why the downstream RL
step actually learns instead of chasing noise.
