# Sample: RewardModelAgent

**Owns:** the reward signal itself. ORM (one scalar / trajectory) or PRM
(one scalar / step). Calibration matters more than raw accuracy — a
mis-calibrated RM guarantees reward hacking downstream.

```
[RewardModel] Fitting PROCESS reward model on 4096 pairs
[RewardModel] PROCESS RM ready — acc 0.74, ECE 0.11
```

Result payload:

```json
{
  "reward_kind": "process",
  "pairs_seen": 4096,
  "held_out_accuracy": 0.74,
  "expected_calibration_error": 0.11,
  "scored_rollouts": 512
}
```

**What to notice:** the expected calibration error (ECE) is reported
alongside accuracy. An RM that says "0.95" when the true rate is 0.60
will train your policy to game exactly that gap.
