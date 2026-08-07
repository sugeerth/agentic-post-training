"""Simple Preference Optimization (SimPO).

Priority: 2 (Advanced technique)
Paper: "SimPO: Simple Preference Optimization with a Reference-Free Reward" (Meng et al., 2024)

SimPO uses length-normalized log probabilities as implicit reward,
eliminating the need for a reference model entirely. The loss is
`−log σ(β·(logp_w − logp_l) − γ)` where the log-probs are averaged over
sequence length and γ is a fixed target margin.

This file follows the DPO/GRPO migration template: a typed Pydantic config
(`SimPOConfigV2`), a `core.Technique`-conforming `SimPOTechnique` that is
self-registered via `@register_technique`, and the legacy `SimPO` /
`SimPOConfig` dataclass surface kept as a thin adapter so existing callers
keep working.
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
    import torch  # noqa: F401  (availability probe; real use is inside SimPOTechnique methods)
    HAS_TORCH = True
except ImportError:  # pragma: no cover
    HAS_TORCH = False


# --------------------------------------------------------------------------- #
# New typed config (Phase 2 contract).
# --------------------------------------------------------------------------- #


class SimPOConfigV2(BaseConfig):
    """Pydantic config for SimPO. Validated at construction time.

    SimPO is reference-free: there is no `ref_*` anything. Callers are
    expected to pass **length-normalized** log-probs (sum logp / num tokens)
    — that normalization is the technique's core idea, and it happens where
    the log-probs are computed, not here.
    """

    learning_rate: float = Field(2e-5, ge=0, le=1, description="Optimizer LR")
    batch_size: int = Field(4, ge=1, description="Per-device batch size")
    gradient_accumulation_steps: int = Field(4, ge=1)
    epochs: int = Field(3, ge=1)
    max_length: int = Field(512, ge=1)
    seed: int = Field(42, ge=0)
    bf16: bool = True

    # SimPO-specific.
    beta: float = Field(
        2.0, ge=0,
        description="Scale on the length-normalized log-prob margin. SimPO "
                    "uses a much larger β than DPO (2.0–2.5 in the paper) "
                    "because the normalized margin is small.",
    )
    gamma: float = Field(
        0.5, ge=0,
        description="Target reward margin γ. The loss only saturates once "
                    "β·margin exceeds γ, forcing a real gap between chosen "
                    "and rejected.",
    )


# --------------------------------------------------------------------------- #
# Protocol-conforming implementation.
# --------------------------------------------------------------------------- #


@register_technique("simpo")
class SimPOTechnique:
    """Conforms to `core.Technique`.

    Consumes one `PreferencePair` per `step()` call, like DPO. The pair's
    `metadata` must carry length-normalized `chosen_logps` / `rejected_logps`
    tensors for the real path; without them (or without torch) the step
    falls through to the deterministic simulation so demos run anywhere.
    """

    name = "simpo"

    def __init__(self, config: SimPOConfigV2 | None = None) -> None:
        self.config = config or SimPOConfigV2()
        self._model: Any = None
        self._tokenizer: Any = None
        self._step: int = 0

    # ---- core.Technique --------------------------------------------------- #

    def prepare(self, model: Any, tokenizer: Any, cfg: Any) -> None:
        self._model = model
        self._tokenizer = tokenizer
        if cfg is not None:
            self.config = cfg if isinstance(cfg, SimPOConfigV2) else SimPOConfigV2.model_validate(cfg)
        self._step = 0

    def step(self, batch: PreferencePair) -> StepMetrics:
        self._step += 1

        md = batch.metadata
        if HAS_TORCH and "chosen_logps" in md and "rejected_logps" in md:
            return self._step_real(batch)

        # Simulation path — replicate the legacy SimPO decay constants exactly.
        loss = simulate_decay(self._step, start=1.2, floor=0.22, decay=0.42)
        epoch = self._step
        extras = {
            "reward": min(0.87, 0.33 + 0.17 * epoch),
            "reward_margin": 0.3 + 0.1 * epoch,
        }
        return StepMetrics(loss=loss, step=self._step, extras=extras)

    def save(self, path: str) -> None:
        if self._model is None:
            raise RuntimeError("SimPOTechnique.save called before .prepare()")
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

        # Reference-free: no ref log-probs, by design.
        loss = pairwise_ratio_loss(
            chosen_logps,
            rejected_logps,
            beta=cfg.beta,
            loss_type=RatioLossType.SIMPO,
            simpo_gamma=cfg.gamma,
        )

        # Implicit reward is β · length-normalized logp.
        chosen_reward = (cfg.beta * chosen_logps).detach()
        rejected_reward = (cfg.beta * rejected_logps).detach()
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
class SimPOConfig(TechniqueConfig):
    """Legacy dataclass config. Prefer `SimPOConfigV2` (Pydantic) for new code."""

    beta: float = 2.0
    gamma: float = 0.5  # Reward margin


class SimPO(BaseTechnique):
    """Legacy `BaseTechnique` adapter.

    `compute_loss(chosen_log_probs=…, rejected_log_probs=…)` runs the real
    SimPO loss when torch tensors are supplied; otherwise it reproduces the
    legacy simulation numbers bit-for-bit.
    """

    name = "simpo"
    description = "SimPO — reference-free, length-normalized preference optimization"
    paper_reference = "Meng et al., 2024 — SimPO: Simple Preference Optimization"
    priority = 2
    recommended_for: ClassVar[list[str]] = ["reference-free alignment", "memory-constrained setups", "simplicity"]
    pros: ClassVar[list[str]] = ["No reference model", "Length-normalized (fair)", "Very simple"]
    cons: ClassVar[list[str]] = ["Less expressive", "Newer technique"]

    def __init__(self, config: SimPOConfig | None = None):
        super().__init__(config or SimPOConfig())
        self._impl = SimPOTechnique(
            SimPOConfigV2(
                learning_rate=self.config.learning_rate,
                batch_size=self.config.batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                epochs=self.config.epochs,
                max_length=self.config.max_length,
                seed=self.config.seed,
                bf16=self.config.bf16,
                beta=getattr(self.config, "beta", 2.0),
                gamma=getattr(self.config, "gamma", 0.5),
            )
        )

    def compute_loss(self, **kwargs: Any) -> Any:
        cfg = self._impl.config

        if HAS_TORCH and "chosen_log_probs" in kwargs and "rejected_log_probs" in kwargs:
            chosen_log_probs = kwargs["chosen_log_probs"]
            rejected_log_probs = kwargs["rejected_log_probs"]

            loss = pairwise_ratio_loss(
                chosen_log_probs,
                rejected_log_probs,
                beta=cfg.beta,
                loss_type=RatioLossType.SIMPO,
                simpo_gamma=cfg.gamma,
            )

            chosen_reward = cfg.beta * chosen_log_probs.detach()
            rejected_reward = cfg.beta * rejected_log_probs.detach()
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
        loss = 1.2 * math.exp(-0.42 * epoch) + 0.22
        self.metrics.update({
            "loss": loss,
            "reward": min(0.87, 0.33 + 0.17 * epoch),
            "reward_margin": 0.3 + 0.1 * epoch,
        })
        return loss
