# Sample: AutonomousLoop

**Owns:** the closed loop itself — collect with the current policy, train,
probe on fresh env rollouts, let the supervisor decide the next round.

```
Autonomous loop · target 85% · budget 2000 rollouts

  round  1 | skill 0.15→0.27 | success 38% (best 38%) | reward 0.56 | kl_coef 0.05 | budget 88/2000  | continue: improving — continue
  round  2 | skill 0.27→0.45 | success 62% (best 62%) | reward 1.00 | kl_coef 0.05 | budget 176/2000 | continue: improving — continue
  round  3 | skill 0.45→0.59 | success 79% (best 79%) | reward 1.00 | kl_coef 0.05 | budget 264/2000 | continue: improving — continue
  round  4 | skill 0.59→0.69 | success 79% (best 79%) | reward 1.00 | kl_coef 0.05 | budget 352/2000 | continue: improving — continue
  round  5 | skill 0.69→0.77 | success 88% (best 88%) | reward 1.00 | kl_coef 0.05 | budget 440/2000 | stop: target met: 88% ≥ 85%

🎯 target met: 88% ≥ 85% — best success 88% in 5 rounds, 440 rollouts
```

And on a deliberately broken config (`learning_gain=0` — the policy
cannot improve):

```
  round 11 | ... | stop: no improvement in 5 rounds despite 2 interventions — converged at 38%
```

**What to notice:**
- `skill_before[r+1] == skill_after[r]` — the policy that collects each
  round's data is the policy the previous round improved. Closed feedback.
- Decisions read the probe (fresh env rollouts), never the training curve.
- The broken run stops at round 11 of 30 with 80% of budget unspent —
  bounded escalation (rollback → exploration boost → stop), not thrashing.
