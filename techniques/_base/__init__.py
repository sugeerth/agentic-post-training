"""Shared math primitives extracted from individual techniques.

Single source of truth for loss helpers that 3+ techniques duplicate.
Currently:

  - `ratio_loss.py`  — β·(chosen_lp − rejected_lp) family (DPO, SPO, KTO, SimPO)
  - `advantages.py`  — group-relative advantage (GRPO) and GAE (PPO)
  - `simulator.py`   — the deterministic "no-torch" fallback so techniques
                        don't each invent their own decay formula
"""

from techniques._base.advantages import compute_gae, compute_group_advantages
from techniques._base.ratio_loss import RatioLossType, pairwise_ratio_loss
from techniques._base.simulator import simulate_decay

__all__ = [
    "RatioLossType",
    "compute_gae",
    "compute_group_advantages",
    "pairwise_ratio_loss",
    "simulate_decay",
]
