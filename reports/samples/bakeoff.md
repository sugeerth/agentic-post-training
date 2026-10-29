# Sample: BakeoffAgent

**Owns:** picking a winner across recipes on a fair, budgeted comparison.
Highest success rate *among plans that stayed under the KL ceiling* — not
raw success, because a plan that hit 90% by drifting off-distribution
isn't the one you ship.

```
══════════════════════════════════════════════════════════════════════
  🥊 Bake-off results
══════════════════════════════════════════════════════════════════════
     plan                          success  reward     kl    sec
     ──────────────────────────────────────────────────────────────
  🏆 agentic-tool-use                  74%    0.93   0.09    2.5
     reasoning-r1                      68%    0.85   0.07    2.4
     rejection-sampling-loop           55%    0.68   0.00    1.9
══════════════════════════════════════════════════════════════════════

Winner: agentic-tool-use
Pareto: agentic-tool-use · rejection-sampling-loop
```

**What to notice:**
- The winner is picked under a KL ceiling (default 0.2) — success without
  drift.
- The pareto frontier drops `reasoning-r1` because
  `agentic-tool-use` beats it on both axes.
- `rejection-sampling-loop` stays on the frontier despite lower success
  because its KL is zero (pure SFT).
