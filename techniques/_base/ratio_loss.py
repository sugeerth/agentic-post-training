"""Pairwise log-probability ratio loss.

Five techniques (DPO, SPO, KTO, SimPO, IPO) compute their loss from the same
two-tensor input — chosen and rejected log-probs — combined with a per-method
shaping function. Pulling the shared piece out kills the duplicated
`β·(chosen_lp − rejected_lp)` algebra and makes the per-technique file just a
choice of shaping function.
"""

from __future__ import annotations

from enum import Enum
from typing import Any


class RatioLossType(str, Enum):
    """Which sigmoid-style transform to apply to the policy/ref ratio.

    - SIGMOID:  −log σ(β·logits)              ← classic DPO (Rafailov et al.)
    - HINGE:    relu(1 − β·logits)             ← SLiC-HF
    - IPO:      (logits − 1/(2β))²             ← IPO (Azar et al.)
    - SIMPO:    −log σ(β·logits − γ)           ← length-normalized SimPO
    """

    SIGMOID = "sigmoid"
    HINGE = "hinge"
    IPO = "ipo"
    SIMPO = "simpo"


def pairwise_ratio_loss(
    chosen_logps: Any,
    rejected_logps: Any,
    ref_chosen_logps: Any | None = None,
    ref_rejected_logps: Any | None = None,
    *,
    beta: float = 0.1,
    loss_type: RatioLossType = RatioLossType.SIGMOID,
    label_smoothing: float = 0.0,
    simpo_gamma: float = 1.4,
) -> Any:
    """Compute the pairwise preference loss.

    Reference log-probs are optional — when omitted (e.g. ORPO, SimPO), the
    ratio collapses to the raw policy log-prob difference, which is the
    "ref-free" variant.

    Returns a scalar tensor when torch is available, else a Python float.
    """
    try:
        import torch
        import torch.nn.functional as F
    except ImportError:  # pragma: no cover  - simulation path tested separately
        return 0.0

    logits = chosen_logps - rejected_logps
    if ref_chosen_logps is not None and ref_rejected_logps is not None:
        logits = logits - (ref_chosen_logps - ref_rejected_logps)

    if loss_type is RatioLossType.SIGMOID:
        # Classic DPO. label_smoothing is the cDPO trick.
        losses = (
            -F.logsigmoid(beta * logits) * (1 - label_smoothing)
            - F.logsigmoid(-beta * logits) * label_smoothing
        )
    elif loss_type is RatioLossType.HINGE:
        losses = torch.relu(1 - beta * logits)
    elif loss_type is RatioLossType.IPO:
        # IPO regularizes the log-ratio toward 1/(2β) instead of pushing
        # it to infinity. See Azar et al. 2023, Eq. 17.
        losses = (logits - 1 / (2 * beta)) ** 2
    elif loss_type is RatioLossType.SIMPO:
        # Length-normalized; caller is expected to pass length-normalized
        # log-probs. `simpo_gamma` is the target margin.
        losses = -F.logsigmoid(beta * logits - simpo_gamma)
    else:  # pragma: no cover - exhaustive
        raise ValueError(f"unknown loss_type: {loss_type!r}")

    return losses.mean()
