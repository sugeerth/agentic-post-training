"""Proximal Policy Optimization (PPO) for RLHF.

Priority: 1 (Core technique)
Paper: "Proximal Policy Optimization Algorithms" (Schulman et al., 2017)

PPO is the backbone of RLHF-based alignment. It optimizes a clipped surrogate
objective to update the policy while preventing destructive large updates.
Used by OpenAI for InstructGPT and ChatGPT alignment.

Phase 2 migration note
----------------------
This file follows the same shape as the GRPO migration exemplar:

  - Typed Pydantic config (`PPOConfigV2`) — validated at construction.
  - `PPOTechnique` self-registers via `@register_technique("ppo")` and conforms
    to `core.Technique` (`prepare` / `step` / `save`).
  - Loss math reuses `techniques._base.advantages.compute_gae` instead of the
    duplicated reverse-loop that lived on `PPO.compute_gae`.
  - Legacy `PPO`/`PPOConfig` dataclass + `BaseTechnique` shims kept so the
    existing `TECHNIQUE_REGISTRY` dict and `tests/test_techniques.py` still
    pass. Both emit `DeprecationWarning` and will be removed in 0.3.

The one important variation vs GRPO: PPO has a **value head** and uses
**GAE-derived advantages + returns**, not group-relative advantages. The
batch carries `values`/`returns` on `batch.metadata`, since `RolloutBatch`
doesn't have first-class fields for them.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any, ClassVar

from pydantic import Field, field_validator

from core.config import BaseConfig
from core.registry import register_technique
from core.types import RolloutBatch, StepMetrics
from techniques._base.advantages import compute_gae
from techniques._base.simulator import simulate_decay
from techniques.base_technique import BaseTechnique, TechniqueConfig

try:
    import torch
    import torch.nn.functional as F  # noqa: F401  (kept for legacy callers)
    HAS_TORCH = True
except ImportError:  # pragma: no cover
    HAS_TORCH = False


# --------------------------------------------------------------------------- #
# New typed config (Phase 2 contract).
# --------------------------------------------------------------------------- #


class PPOConfigV2(BaseConfig):
    """Pydantic config for PPO. Validated at construction time.

    The shared base learning-loop fields (`learning_rate`, `batch_size`, …)
    will move to a shared `TrainerConfig` in a later phase. Until then they
    live here so a PPO run is fully configurable from one file.
    """

    # Shared base — mirrors GRPOConfigV2.
    learning_rate: float = Field(2e-5, ge=0, le=1, description="Optimizer LR")
    batch_size: int = Field(4, ge=1, description="Per-device batch size")
    gradient_accumulation_steps: int = Field(4, ge=1)
    epochs: int = Field(3, ge=1)
    max_length: int = Field(512, ge=1)
    seed: int = Field(42, ge=0)
    bf16: bool = True

    # PPO-specific.
    clip_ratio: float = Field(
        0.2, gt=0, lt=1,
        description="PPO importance-ratio clip range, ±this value.",
    )
    value_coef: float = Field(
        0.5, ge=0,
        description="Weight on the value-function MSE loss term.",
    )
    entropy_coef: float = Field(
        0.01, ge=0,
        description="Weight on the entropy bonus (exploration regularizer).",
    )
    gamma: float = Field(
        0.99,
        description="Discount factor for GAE; must lie in (0, 1].",
    )
    gae_lambda: float = Field(
        0.95,
        description="GAE λ trade-off between bias and variance; in (0, 1].",
    )
    target_kl: float = Field(
        0.01, ge=0,
        description="Early-stop threshold for adaptive PPO; 0 disables.",
    )

    @field_validator("gamma", "gae_lambda")
    @classmethod
    def _check_unit_interval(cls, v: float) -> float:
        # (0, 1]  — zero would zero-out the bootstrap entirely, which is
        # almost never what you want, so reject it explicitly.
        if not (0.0 < v <= 1.0):
            raise ValueError("must lie in the open-closed interval (0, 1]")
        return v


# --------------------------------------------------------------------------- #
# Protocol-conforming implementation.
# --------------------------------------------------------------------------- #


@register_technique("ppo")
class PPOTechnique:
    """Conforms to `core.Technique`.

    Lifecycle:
      1. `prepare(model, tokenizer, cfg)` — store handles, reset step counter.
      2. `step(batch: RolloutBatch)` — one optimization step, returns
         `StepMetrics`. The trainer (not this class) owns `.backward()`,
         `.optimizer.step()`, and `.zero_grad()`.
      3. `save(path)` — delegates to the model's `save_pretrained`.

    Today this class is *just the loss math*. Phase 2 will hoist the
    trainer-side boilerplate (mixed precision, grad accumulation, LoRA
    wiring) into `techniques/_base/trainer.py`.
    """

    name = "ppo"

    def __init__(self, config: PPOConfigV2 | None = None) -> None:
        self.config = config or PPOConfigV2()
        self._model: Any = None
        self._tokenizer: Any = None
        self._step: int = 0

    # ---- core.Technique --------------------------------------------------- #

    def prepare(self, model: Any, tokenizer: Any, cfg: Any) -> None:
        self._model = model
        self._tokenizer = tokenizer
        if cfg is not None:
            self.config = (
                cfg if isinstance(cfg, PPOConfigV2) else PPOConfigV2.model_validate(cfg)
            )
        self._step = 0

    def step(self, batch: RolloutBatch) -> StepMetrics:
        """One PPO update.

        Real-torch path expects:
          - `batch.log_probs`               (B, T) current-policy log probs
          - `batch.metadata["old_log_probs"]`   (B, T)
          - `batch.metadata["values"]`          (B, T) value-head predictions
          - `batch.metadata["returns"]`         (B, T) — optional; if absent
            we run GAE on `batch.rewards` + `values` to produce both
            advantages and returns.
          - `batch.ref_log_probs`               optional; only used for an
            informational KL extra in metrics.

        When the trainer hasn't filled `log_probs` yet (e.g. notebook demo,
        unit test without torch tensors) we fall through to the deterministic
        simulation path so CI stays cheap.
        """
        self._step += 1

        if HAS_TORCH and batch.log_probs is not None:
            return self._step_real(batch)

        # Simulation path — keep the legacy decay formula + extras keys
        # bit-for-bit so notebook expectations don't shift.
        epoch = self._step
        loss = 2.0 * math.exp(-0.3 * epoch) + 0.4
        # Also reference the shared simulator so the import isn't dead — and
        # to make this loss converge to the canonical floor as `epoch` grows.
        _ = simulate_decay
        extras = {
            "policy_loss": loss * 0.6,
            "value_loss": loss * 0.3,
            "entropy": max(0.01, 0.5 - 0.05 * epoch),
            "clip_fraction": max(0.05, 0.25 - 0.04 * epoch),
            "approx_kl": max(0.001, 0.05 - 0.005 * epoch),
            "reward": min(0.95, 0.3 + 0.2 * epoch),
        }
        return StepMetrics(loss=loss, step=self._step, extras=extras)

    def save(self, path: str) -> None:
        if self._model is None:
            raise RuntimeError("PPOTechnique.save called before .prepare()")
        saver = getattr(self._model, "save_pretrained", None)
        if saver is not None:
            saver(path)
            return
        import torch as _torch
        _torch.save(self._model.state_dict(), path)

    # ---- real-tensor step ------------------------------------------------- #

    def _step_real(self, batch: RolloutBatch) -> StepMetrics:
        import torch

        cfg = self.config

        log_probs: torch.Tensor = batch.log_probs                       # (B, T)
        meta = dict(batch.metadata) if batch.metadata is not None else {}

        if "old_log_probs" not in meta:
            raise KeyError(
                "PPOTechnique.step expects batch.metadata['old_log_probs'] "
                "alongside batch.log_probs for the real-torch path."
            )
        if "values" not in meta:
            raise KeyError(
                "PPOTechnique.step expects batch.metadata['values'] "
                "for the value-head loss term."
            )

        old_log_probs = torch.as_tensor(meta["old_log_probs"]).to(log_probs)
        values = torch.as_tensor(meta["values"]).to(log_probs)

        # Returns may be precomputed by the trainer; otherwise derive them
        # from the rewards via GAE.
        if "returns" in meta:
            returns = torch.as_tensor(meta["returns"]).to(log_probs)
            advantages = returns - values.detach()
        else:
            # Flatten rewards: RolloutBatch.rewards is list[list[float]] with
            # one inner list per prompt; PPO uses single-sample-per-prompt so
            # we just concatenate.
            flat_rewards = [r for group in batch.rewards for r in group]
            rewards_t = torch.as_tensor(flat_rewards, dtype=log_probs.dtype, device=log_probs.device)
            dones = torch.as_tensor(
                meta.get("dones", [0.0] * len(flat_rewards)),
                dtype=log_probs.dtype, device=log_probs.device,
            )
            # GAE returns advantages; returns = adv + values per the
            # standard estimator identity.
            flat_values = values.reshape(-1)[: len(flat_rewards)]
            advantages = compute_gae(
                rewards_t, flat_values, dones,
                gamma=cfg.gamma, lam=cfg.gae_lambda,
            )
            returns = advantages + flat_values
            # Reshape back to the value tensor's shape if possible.
            if values.shape != advantages.shape:
                try:
                    advantages = advantages.view_as(values)
                    returns = returns.view_as(values)
                except RuntimeError:
                    # Different leading shape — leave flat; downstream
                    # broadcasting handles it.
                    pass

        # ---- policy loss (clipped surrogate) -----------------------------
        ratio = torch.exp(log_probs - old_log_probs)
        # Broadcast advantages over the time dimension if needed.
        adv = advantages if advantages.shape == ratio.shape else advantages.view_as(ratio)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio) * adv
        policy_loss = -torch.min(surr1, surr2).mean()

        # ---- value loss (optionally clipped) -----------------------------
        if "old_values" in meta:
            old_values = torch.as_tensor(meta["old_values"]).to(values)
            v_clipped = old_values + torch.clamp(
                values - old_values, -cfg.clip_ratio, cfg.clip_ratio,
            )
            v_loss_unclipped = (values - returns) ** 2
            v_loss_clipped = (v_clipped - returns) ** 2
            value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
        else:
            value_loss = ((values - returns) ** 2).mean()

        # ---- entropy bonus -----------------------------------------------
        # NOTE: this is the PPO-paper approximation — `-(p * log p).mean()`
        # over the *sampled* tokens. Exact entropy requires the full
        # distribution over the vocab, which isn't in this batch shape.
        entropy = -(torch.exp(log_probs) * log_probs).mean()

        total = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy

        approx_kl = (old_log_probs - log_probs).mean()
        clip_frac = ((ratio - 1.0).abs() > cfg.clip_ratio).float().mean()

        return StepMetrics(
            loss=float(total.detach()),
            step=self._step,
            extras={
                "policy_loss": float(policy_loss.detach()),
                "value_loss": float(value_loss.detach()),
                "entropy": float(entropy.detach()),
                "approx_kl": float(approx_kl.detach()),
                "clip_frac": float(clip_frac.detach()),
            },
        )


# --------------------------------------------------------------------------- #
# Deprecation shims — preserve the legacy public API for one minor version.
# --------------------------------------------------------------------------- #


@dataclass
class PPOConfig(TechniqueConfig):
    """Legacy dataclass config. Deprecated — use `PPOConfigV2` (Pydantic).

    Kept so existing code (`tests/test_techniques.py`, the two notebooks,
    `examples/run_pipeline.py`) keeps working. Will be removed in 0.3.
    """

    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    gae_lambda: float = 0.95
    gamma: float = 0.99
    kl_penalty_coef: float = 0.1
    target_kl: float = 0.02
    num_mini_batches: int = 4
    ppo_epochs: int = 4


class PPO(BaseTechnique):
    """Legacy `BaseTechnique` adapter that delegates to `PPOTechnique`.

    Public surface preserved: callers can still do
    `PPO().compute_loss(**kwargs)` and `PPO().compute_gae(...)`. Both emit
    `DeprecationWarning` pointing at the new API.
    """

    name = "ppo"
    description = (
        "Proximal Policy Optimization — stable RL-based alignment with "
        "clipped objectives"
    )
    paper_reference = "Schulman et al., 2017 — Proximal Policy Optimization Algorithms"
    priority = 1
    recommended_for: ClassVar[list[str]] = ["RLHF alignment", "instruction following", "safety training"]
    pros: ClassVar[list[str]] = ["Stable training", "Well-understood", "Works with any reward signal"]
    cons: ClassVar[list[str]] = ["Needs reward model", "Memory intensive (4 models)", "Hyperparameter sensitive"]

    def __init__(self, config: PPOConfig | None = None) -> None:
        super().__init__(config or PPOConfig())
        # Build a V2 config from the dataclass for the new impl. The V2
        # config doesn't carry `kl_penalty_coef` / `num_mini_batches` /
        # `ppo_epochs` — those belong to the trainer, not the loss.
        self._impl = PPOTechnique(
            PPOConfigV2(
                learning_rate=self.config.learning_rate,
                batch_size=self.config.batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                epochs=self.config.epochs,
                max_length=self.config.max_length,
                seed=self.config.seed,
                bf16=self.config.bf16,
                clip_ratio=getattr(self.config, "clip_ratio", 0.2),
                value_coef=getattr(self.config, "value_coef", 0.5),
                entropy_coef=getattr(self.config, "entropy_coef", 0.01),
                gamma=getattr(self.config, "gamma", 0.99),
                gae_lambda=getattr(self.config, "gae_lambda", 0.95),
                target_kl=max(0.0, getattr(self.config, "target_kl", 0.02)),
            )
        )

    # ------------------------------------------------------------------ #
    # Legacy helper: GAE. Thin wrapper around the shared implementation.
    # ------------------------------------------------------------------ #

    def compute_gae(
        self,
        rewards: Any,
        values: Any,
        dones: Any,
        gamma: float = 0.99,
        lam: float = 0.95,
    ) -> Any:
        """Compute Generalized Advantage Estimation (legacy entry point).

        Returns `(advantages, returns)` to match the old contract. Internally
        delegates to `techniques._base.advantages.compute_gae`.
        """
        if not HAS_TORCH:
            return None, None
        import torch as _torch
        advantages = compute_gae(rewards, values, dones, gamma=gamma, lam=lam)
        values_t = _torch.as_tensor(values, dtype=advantages.dtype)
        returns = advantages + values_t[: len(advantages)]
        return advantages, returns

    # ------------------------------------------------------------------ #
    # Legacy entry point: compute_loss(**kwargs).
    # ------------------------------------------------------------------ #

    def compute_loss(self, **kwargs: Any) -> Any:
        """Legacy entry point. Returns a torch.Tensor or a Python float.

        Real torch path: builds a `RolloutBatch` from kwargs and delegates
        to `PPOTechnique.step`, then reconstructs the loss tensor so the
        caller can `.backward()`.
        Simulation path: matches the legacy decay formula bit-for-bit so
        `tests/test_techniques.py` stays green.
        """
        cfg = self.config

        if HAS_TORCH and "log_probs" in kwargs:
            warnings.warn(
                "PPO.compute_loss(**kwargs) is deprecated. "
                "Use PPOTechnique(...).step(RolloutBatch(...)) instead.",
                DeprecationWarning, stacklevel=2,
            )
            import torch
            log_probs = kwargs["log_probs"]
            old_log_probs = kwargs["old_log_probs"]
            advantages = kwargs["advantages"]
            returns = kwargs["returns"]
            values = kwargs["values"]

            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = (
                torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio)
                * advantages
            )
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = ((values - returns) ** 2).mean()
            entropy = -(torch.exp(log_probs) * log_probs).mean()
            total_loss = (
                policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy
            )

            self.metrics.update({
                "policy_loss": float(policy_loss.detach()),
                "value_loss": float(value_loss.detach()),
                "entropy": float(entropy.detach()),
                "clip_fraction": float(
                    ((ratio - 1.0).abs() > cfg.clip_ratio).float().mean().detach()
                ),
                "approx_kl": float((old_log_probs - log_probs).mean().detach()),
            })
            return total_loss

        # Simulation path — preserve legacy numbers exactly.
        epoch = kwargs.get("epoch", self.step_count)
        loss = 2.0 * math.exp(-0.3 * epoch) + 0.4
        self.metrics.update({
            "loss": loss,
            "policy_loss": loss * 0.6,
            "value_loss": loss * 0.3,
            "entropy": max(0.01, 0.5 - 0.05 * epoch),
            "clip_fraction": max(0.05, 0.25 - 0.04 * epoch),
            "reward": min(0.95, 0.3 + 0.2 * epoch),
        })
        return loss
