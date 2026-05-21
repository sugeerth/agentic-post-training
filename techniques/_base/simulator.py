"""Deterministic 'no-torch' fallback used by every legacy technique.

Each technique used to invent its own decay formula and constants. That
pattern duplicated 12 times made the loss math hard to read and the
simulation behavior inconsistent. This helper unifies it.

Real (torch-enabled) loss computation lives in the technique files. This
module is for examples, tests, and notebook demos that need representative
numbers without spinning up CUDA.
"""

from __future__ import annotations

import math


def simulate_decay(
    epoch: int,
    *,
    start: float = 1.8,
    floor: float = 0.3,
    decay: float = 0.35,
) -> float:
    """Monotonically decaying scalar — a plausible-looking 'loss' for demos.

    `start * exp(-decay * epoch) + floor` was the formula reverse-engineered
    from the legacy GRPO/DPO/PPO simulation branches. Keeping it identical
    so existing test expectations still pass.
    """
    return start * math.exp(-decay * epoch) + floor
