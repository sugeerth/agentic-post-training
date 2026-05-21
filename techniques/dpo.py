"""Direct Preference Optimization (DPO).

Priority: 1 (Core technique)
Paper: "Direct Preference Optimization: Your Language Model is Secretly a
Reward Model" (Rafailov et al., 2023)

DPO directly optimizes the policy from preference pairs without training a
separate reward model. It reparameterizes the RLHF objective to derive a
simple classification loss on preference pairs.

This file is the **Phase 2 migration**: it mirrors the GRPO template
(typed Pydantic config, `core.Technique` Protocol, shared `_base` math,
self-registration via `@register_technique`) while keeping the legacy
`DPO`/`DPOConfig` dataclass surface as a deprecation shim for one minor
version so the existing tests, notebooks, and `TECHNIQUE_REGISTRY` dict
keep working.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any, ClassVar

from pydantic import Field, field_validator

from core.config import BaseConfig
from core.registry import register_technique
from core.types import PreferencePair, StepMetrics
from techniques._base.ratio_loss import RatioLossType, pairwise_ratio_loss
from techniques._base.simulator import simulate_decay
from techniques.base_technique import BaseTechnique, TechniqueConfig

try:
    import torch  # noqa: F401  (availability probe; real use is inside DPOTechnique methods)
    HAS_TORCH = True
except ImportError:  # pragma: no cover
    HAS_TORCH = False


# --------------------------------------------------------------------------- #
# New typed config (Phase 2 contract).
# --------------------------------------------------------------------------- #


class DPOConfigV2(BaseConfig):
    """Pydantic config for DPO. Validated at construction time.

    Shares the base learning-loop fields with every other technique config
    (Phase 2 will hoist these to a `TrainerConfig`). DPO-specific fields
    cover the temperature `beta`, conservative-DPO `label_smoothing`, and
    the loss shape selector `loss_type`.
    """

    learning_rate: float = Field(2e-5, ge=0, le=1, description="Optimizer LR")
    batch_size: int = Field(4, ge=1, description="Per-device batch size")
    gradient_accumulation_steps: int = Field(4, ge=1)
    epochs: int = Field(3, ge=1)
    max_length: int = Field(512, ge=1)
    seed: int = Field(42, ge=0)
    bf16: bool = True

    # DPO-specific.
    beta: float = Field(
        0.1, ge=0,
        description="Temperature on the policy/ref log-ratio. Higher = stays "
                    "closer to the reference model.",
    )
    label_smoothing: float = Field(
        0.0, ge=0, le=0.5,
        description="Conservative-DPO label smoothing in [0, 0.5]. 0 = vanilla DPO.",
    )
    loss_type: str = Field(
        "sigmoid",
        description="Pairwise loss shape. One of 'sigmoid' | 'hinge' | 'ipo'.",
    )

    @field_validator("loss_type")
    @classmethod
    def _check_loss_type(cls, v: str) -> str:
        allowed = {"sigmoid", "hinge", "ipo"}
        if v not in allowed:
            raise ValueError(f"loss_type must be one of {sorted(allowed)}")
        return v


# --------------------------------------------------------------------------- #
# Protocol-conforming implementation.
# --------------------------------------------------------------------------- #


@register_technique("dpo")
class DPOTechnique:
    """Conforms to `core.Technique`.

    Lifecycle:
      1. `prepare(model, tokenizer, cfg)` — store handles, init step counter.
      2. `step(batch: PreferencePair)` — one optimization step, returns
         `StepMetrics`. Unlike GRPO (which sees a `RolloutBatch`), DPO
         consumes a single `PreferencePair` per call; batching across pairs
         is the trainer's job.
      3. `save(path)` — delegates to the model's `save_pretrained`.
    """

    name = "dpo"

    def __init__(self, config: DPOConfigV2 | None = None) -> None:
        self.config = config or DPOConfigV2()
        self._model: Any = None
        self._tokenizer: Any = None
        self._step: int = 0

    # ---- core.Technique --------------------------------------------------- #

    def prepare(self, model: Any, tokenizer: Any, cfg: Any) -> None:
        """Wire the model, tokenizer, and (optionally) override the config."""
        self._model = model
        self._tokenizer = tokenizer
        if cfg is not None:
            self.config = cfg if isinstance(cfg, DPOConfigV2) else DPOConfigV2.model_validate(cfg)
        self._step = 0

    def step(self, batch: PreferencePair) -> StepMetrics:
        """One DPO update.

        If `batch.metadata` carries pre-computed log-prob tensors
        (`chosen_logps`, `rejected_logps`, optionally `ref_chosen_logps` /
        `ref_rejected_logps`) we hand them to `pairwise_ratio_loss`.
        Otherwise we fall through to the deterministic simulation path so
        demos and tests run without a GPU.
        """
        self._step += 1

        md = batch.metadata
        if HAS_TORCH and "chosen_logps" in md and "rejected_logps" in md:
            return self._step_real(batch)

        # Simulation path — replicate the legacy DPO decay constants exactly.
        loss = simulate_decay(self._step, start=1.5, floor=0.25, decay=0.4)
        epoch = self._step
        extras = {
            "chosen_reward": 0.5 + 0.15 * epoch,
            "rejected_reward": -0.3 - 0.1 * epoch,
            "reward_margin": 0.8 + 0.25 * epoch,
            "accuracy": min(0.95, 0.6 + 0.1 * epoch),
        }
        return StepMetrics(loss=loss, step=self._step, extras=extras)

    def save(self, path: str) -> None:
        """Persist adapter / full weights via the model's saver."""
        if self._model is None:
            raise RuntimeError("DPOTechnique.save called before .prepare()")
        saver = getattr(self._model, "save_pretrained", None)
        if saver is not None:
            saver(path)
            return
        import torch as _torch
        _torch.save(self._model.state_dict(), path)

    # ---- real-tensor step ------------------------------------------------- #

    def _step_real(self, batch: PreferencePair) -> StepMetrics:
        import torch

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
            beta=cfg.beta,
            loss_type=RatioLossType(cfg.loss_type),
            label_smoothing=cfg.label_smoothing,
        )

        # Implicit rewards for logging (β · log-ratio).
        if ref_chosen_logps is not None and ref_rejected_logps is not None:
            chosen_reward = (cfg.beta * (chosen_logps - ref_chosen_logps)).detach()
            rejected_reward = (cfg.beta * (rejected_logps - ref_rejected_logps)).detach()
        else:
            chosen_reward = (cfg.beta * chosen_logps).detach()
            rejected_reward = (cfg.beta * rejected_logps).detach()

        margin = (chosen_reward - rejected_reward)
        accuracy = (margin > 0).float().mean() if hasattr(margin, "float") else torch.tensor(0.0)

        return StepMetrics(
            loss=float(loss.detach()),
            step=self._step,
            extras={
                "chosen_reward": float(chosen_reward.mean()),
                "rejected_reward": float(rejected_reward.mean()),
                "reward_margin": float(margin.mean()),
                "accuracy": float(accuracy),
            },
        )


# --------------------------------------------------------------------------- #
# Deprecation shims — preserve the legacy public API for one minor version.
# --------------------------------------------------------------------------- #


@dataclass
class DPOConfig(TechniqueConfig):
    """Legacy dataclass config. Deprecated — use `DPOConfigV2` (Pydantic).

    Kept so existing code (`tests/test_techniques.py`, the two notebooks,
    `examples/run_pipeline.py`) keeps working. Will be removed in 0.3.
    """

    beta: float = 0.1
    label_smoothing: float = 0.0
    reference_free: bool = False
    loss_type: str = "sigmoid"  # sigmoid, hinge, ipo


class DPO(BaseTechnique):
    """Legacy `BaseTechnique` adapter that delegates to `DPOTechnique`.

    Public surface preserved: callers can still do
    `DPO().compute_loss(**kwargs)`. The kwarg path emits `DeprecationWarning`
    pointing at the new `DPOTechnique.step(PreferencePair(...))` API.
    """

    name = "dpo"
    description = "Direct Preference Optimization — simple preference alignment without RL"
    paper_reference = "Rafailov et al., 2023 — Direct Preference Optimization"
    priority = 1
    recommended_for: ClassVar[list[str]] = ["preference alignment", "instruction following", "simple setups"]
    pros: ClassVar[list[str]] = ["No reward model", "Simple", "Stable training", "Single stage"]
    cons: ClassVar[list[str]] = ["Needs paired preferences", "Less flexible than RL", "Reference model overhead"]

    def __init__(self, config: DPOConfig | None = None) -> None:
        super().__init__(config or DPOConfig())
        loss_type = getattr(self.config, "loss_type", "sigmoid")
        # Legacy allowed unknown strings to fall through to sigmoid; normalize
        # so the strict Pydantic validator doesn't reject historical configs.
        if loss_type not in {"sigmoid", "hinge", "ipo"}:
            loss_type = "sigmoid"
        self._impl = DPOTechnique(
            DPOConfigV2(
                learning_rate=self.config.learning_rate,
                batch_size=self.config.batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                epochs=self.config.epochs,
                max_length=self.config.max_length,
                seed=self.config.seed,
                bf16=self.config.bf16,
                beta=getattr(self.config, "beta", 0.1),
                label_smoothing=getattr(self.config, "label_smoothing", 0.0),
                loss_type=loss_type,
            )
        )

    def compute_loss(self, **kwargs: Any) -> Any:
        """Legacy entry point. Returns a torch.Tensor or a Python float.

        Real torch path: builds a `PreferencePair` (with empty prompt/chosen/
        rejected — those aren't on the tensor path) and delegates to the new
        impl via `pairwise_ratio_loss` so the caller can still backprop on
        the returned tensor.
        Simulation path: matches the legacy decay formula bit-for-bit so
        `tests/test_techniques.py` stays green.
        """
        cfg = self.config

        if HAS_TORCH and "chosen_log_probs" in kwargs:
            warnings.warn(
                "DPO.compute_loss(**kwargs) is deprecated. "
                "Use DPOTechnique(...).step(PreferencePair(...)) instead.",
                DeprecationWarning, stacklevel=2,
            )
            chosen_log_probs = kwargs["chosen_log_probs"]
            rejected_log_probs = kwargs["rejected_log_probs"]
            ref_chosen_log_probs = kwargs.get("ref_chosen_log_probs")
            ref_rejected_log_probs = kwargs.get("ref_rejected_log_probs")

            # Build a PreferencePair on the tensor path (prompt/chosen/rejected
            # text is unknown to the legacy caller).
            _ = PreferencePair(
                prompt="",
                chosen="",
                rejected="",
                metadata={
                    "chosen_logps": chosen_log_probs,
                    "rejected_logps": rejected_log_probs,
                    **({"ref_chosen_logps": ref_chosen_log_probs}
                       if ref_chosen_log_probs is not None else {}),
                    **({"ref_rejected_logps": ref_rejected_log_probs}
                       if ref_rejected_log_probs is not None else {}),
                },
            )

            loss_type_str = getattr(cfg, "loss_type", "sigmoid")
            if loss_type_str not in {"sigmoid", "hinge", "ipo"}:
                loss_type_str = "sigmoid"

            loss = pairwise_ratio_loss(
                chosen_log_probs,
                rejected_log_probs,
                ref_chosen_log_probs,
                ref_rejected_log_probs,
                beta=cfg.beta,
                loss_type=RatioLossType(loss_type_str),
                label_smoothing=getattr(cfg, "label_smoothing", 0.0),
            )

            # Implicit reward bookkeeping for the legacy metrics dict.
            if ref_chosen_log_probs is not None and ref_rejected_log_probs is not None:
                chosen_reward = cfg.beta * (chosen_log_probs - ref_chosen_log_probs).detach()
                rejected_reward = cfg.beta * (rejected_log_probs - ref_rejected_log_probs).detach()
            else:
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
        loss = 1.5 * math.exp(-0.4 * epoch) + 0.25
        self.metrics.update({
            "loss": loss,
            "chosen_reward": 0.5 + 0.15 * epoch,
            "rejected_reward": -0.3 - 0.1 * epoch,
            "reward_margin": 0.8 + 0.25 * epoch,
            "accuracy": min(0.95, 0.6 + 0.1 * epoch),
        })
        return loss
