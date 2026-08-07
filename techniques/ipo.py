"""Identity Preference Optimization (IPO).

Priority: 3 (Research variant)
Paper: "A General Theoretical Paradigm to Understand Learning from Human Feedback"
       (Azar et al., 2023)

IPO addresses DPO's overfitting by regressing the policy/reference log-ratio
toward the constant `1/(2τ)` instead of pushing it to infinity:
`(logits − 1/(2τ))²` (Azar et al., Eq. 17). The squared loss keeps the
policy close to the reference even on preference pairs the model already
gets right.

This file follows the DPO/GRPO migration template: a typed Pydantic config
(`IPOConfigV2`), a `core.Technique`-conforming `IPOTechnique` self-registered
via `@register_technique`, and the legacy `IPO` / `IPOConfig` dataclass
surface kept as a thin adapter so existing callers keep working.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, ClassVar

from pydantic import Field

from core.config import BaseConfig
from core.registry import register_technique
from core.types import PreferencePair, StepMetrics
from techniques._base.ratio_loss import RatioLossType, pairwise_ratio_loss
from techniques._base.simulator import simulate_decay
from techniques.base_technique import BaseTechnique, TechniqueConfig

try:
    import torch  # noqa: F401  (availability probe; real use is inside IPOTechnique methods)
    HAS_TORCH = True
except ImportError:  # pragma: no cover
    HAS_TORCH = False


# --------------------------------------------------------------------------- #
# New typed config (Phase 2 contract).
# --------------------------------------------------------------------------- #


class IPOConfigV2(BaseConfig):
    """Pydantic config for IPO. Validated at construction time.

    `tau` plays the role DPO's β plays, but the loss shape is quadratic:
    the log-ratio is regressed toward `1/(2τ)`, so smaller τ demands a
    larger (but still finite) margin.
    """

    learning_rate: float = Field(2e-5, ge=0, le=1, description="Optimizer LR")
    batch_size: int = Field(4, ge=1, description="Per-device batch size")
    gradient_accumulation_steps: int = Field(4, ge=1)
    epochs: int = Field(3, ge=1)
    max_length: int = Field(512, ge=1)
    seed: int = Field(42, ge=0)
    bf16: bool = True

    # IPO-specific.
    tau: float = Field(
        0.1, gt=0,
        description="Regularization strength τ. The log-ratio is regressed "
                    "toward 1/(2τ); smaller τ ⇒ larger target margin.",
    )


# --------------------------------------------------------------------------- #
# Protocol-conforming implementation.
# --------------------------------------------------------------------------- #


@register_technique("ipo")
class IPOTechnique:
    """Conforms to `core.Technique`.

    Consumes one `PreferencePair` per `step()` call, like DPO. The pair's
    `metadata` carries `chosen_logps` / `rejected_logps` (and optionally
    `ref_chosen_logps` / `ref_rejected_logps`) tensors for the real path;
    without them (or without torch) the step falls through to the
    deterministic simulation so demos run anywhere.
    """

    name = "ipo"

    def __init__(self, config: IPOConfigV2 | None = None) -> None:
        self.config = config or IPOConfigV2()
        self._model: Any = None
        self._tokenizer: Any = None
        self._step: int = 0

    # ---- core.Technique --------------------------------------------------- #

    def prepare(self, model: Any, tokenizer: Any, cfg: Any) -> None:
        self._model = model
        self._tokenizer = tokenizer
        if cfg is not None:
            self.config = cfg if isinstance(cfg, IPOConfigV2) else IPOConfigV2.model_validate(cfg)
        self._step = 0

    def step(self, batch: PreferencePair) -> StepMetrics:
        self._step += 1

        md = batch.metadata
        if HAS_TORCH and "chosen_logps" in md and "rejected_logps" in md:
            return self._step_real(batch)

        # Simulation path — replicate the legacy IPO decay constants exactly.
        loss = simulate_decay(self._step, start=1.3, floor=0.27, decay=0.38)
        epoch = self._step
        extras = {"reward": min(0.86, 0.32 + 0.16 * epoch)}
        return StepMetrics(loss=loss, step=self._step, extras=extras)

    def save(self, path: str) -> None:
        if self._model is None:
            raise RuntimeError("IPOTechnique.save called before .prepare()")
        saver = getattr(self._model, "save_pretrained", None)
        if saver is not None:
            saver(path)
            return
        import torch as _torch
        _torch.save(self._model.state_dict(), path)

    # ---- real-tensor step ------------------------------------------------- #

    def _step_real(self, batch: PreferencePair) -> StepMetrics:
        cfg = self.config
        md = batch.metadata
        chosen_logps = md["chosen_logps"]
        rejected_logps = md["rejected_logps"]
        ref_chosen_logps = md.get("ref_chosen_logps")
        ref_rejected_logps = md.get("ref_rejected_logps")

        loss = pairwise_ratio_loss(
            chosen_logps,
            rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
            beta=cfg.tau,
            loss_type=RatioLossType.IPO,
        )

        if ref_chosen_logps is not None and ref_rejected_logps is not None:
            chosen_reward = (cfg.tau * (chosen_logps - ref_chosen_logps)).detach()
            rejected_reward = (cfg.tau * (rejected_logps - ref_rejected_logps)).detach()
        else:
            chosen_reward = (cfg.tau * chosen_logps).detach()
            rejected_reward = (cfg.tau * rejected_logps).detach()
        margin = chosen_reward - rejected_reward

        return StepMetrics(
            loss=float(loss.detach()),
            step=self._step,
            extras={
                "chosen_reward": float(chosen_reward.mean()),
                "rejected_reward": float(rejected_reward.mean()),
                "reward_margin": float(margin.mean()),
                "accuracy": float((margin > 0).float().mean()),
            },
        )


# --------------------------------------------------------------------------- #
# Legacy surface — thin adapter over the real implementation.
# --------------------------------------------------------------------------- #


@dataclass
class IPOConfig(TechniqueConfig):
    """Legacy dataclass config. Prefer `IPOConfigV2` (Pydantic) for new code."""

    tau: float = 0.1  # Regularization strength


class IPO(BaseTechnique):
    """Legacy `BaseTechnique` adapter.

    `compute_loss(chosen_log_probs=…, rejected_log_probs=…)` runs the real
    IPO squared loss when torch tensors are supplied; otherwise it reproduces
    the legacy simulation numbers bit-for-bit.
    """

    name = "ipo"
    description = "Identity Preference Optimization — regularized DPO variant"
    paper_reference = "Azar et al., 2023 — A General Theoretical Paradigm for Learning from HF"
    priority = 3
    recommended_for: ClassVar[list[str]] = ["when DPO overfits", "regularized alignment", "research"]
    pros: ClassVar[list[str]] = ["Better regularization than DPO", "Theoretically grounded"]
    cons: ClassVar[list[str]] = ["Marginal gains in practice", "Less adopted"]

    def __init__(self, config: IPOConfig | None = None):
        super().__init__(config or IPOConfig())
        self._impl = IPOTechnique(
            IPOConfigV2(
                learning_rate=self.config.learning_rate,
                batch_size=self.config.batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                epochs=self.config.epochs,
                max_length=self.config.max_length,
                seed=self.config.seed,
                bf16=self.config.bf16,
                tau=getattr(self.config, "tau", 0.1),
            )
        )

    def compute_loss(self, **kwargs: Any) -> Any:
        cfg = self._impl.config

        if HAS_TORCH and "chosen_log_probs" in kwargs and "rejected_log_probs" in kwargs:
            chosen_log_probs = kwargs["chosen_log_probs"]
            rejected_log_probs = kwargs["rejected_log_probs"]
            ref_chosen_log_probs = kwargs.get("ref_chosen_log_probs")
            ref_rejected_log_probs = kwargs.get("ref_rejected_log_probs")

            loss = pairwise_ratio_loss(
                chosen_log_probs,
                rejected_log_probs,
                ref_chosen_log_probs,
                ref_rejected_log_probs,
                beta=cfg.tau,
                loss_type=RatioLossType.IPO,
            )

            if ref_chosen_log_probs is not None and ref_rejected_log_probs is not None:
                chosen_reward = cfg.tau * (chosen_log_probs - ref_chosen_log_probs).detach()
                rejected_reward = cfg.tau * (rejected_log_probs - ref_rejected_log_probs).detach()
            else:
                chosen_reward = cfg.tau * chosen_log_probs.detach()
                rejected_reward = cfg.tau * rejected_log_probs.detach()
            margin = chosen_reward - rejected_reward

            self.metrics.update({
                "loss": float(loss.detach()),
                "chosen_reward": float(chosen_reward.mean()),
                "rejected_reward": float(rejected_reward.mean()),
                "reward_margin": float(margin.mean()),
                "accuracy": float((margin > 0).float().mean()),
            })
            return loss

        # Simulation path — preserve legacy numbers exactly.
        epoch = kwargs.get("epoch", self.step_count)
        loss = 1.3 * math.exp(-0.38 * epoch) + 0.27
        self.metrics.update({
            "loss": loss,
            "reward": min(0.86, 0.32 + 0.16 * epoch),
        })
        return loss
