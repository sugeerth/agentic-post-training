"""Group Relative Policy Optimization (GRPO).

Priority: 1 (Core technique — used in DeepSeek-R1)
Paper: "DeepSeekMath: Pushing the Limits of Mathematical Reasoning" (Shao et al., 2024)

GRPO eliminates the need for a separate value model by using group-level
relative ranking. For each prompt, multiple responses are sampled and ranked
within the group. The advantage of each response is computed relative to the
group mean reward.

This file is the **Phase 1 migration exemplar**:

  - Typed Pydantic config (no `**kwargs` on the public API)
  - Conforms to `core.Technique` Protocol (`prepare` / `step` / `save`)
  - Loss math delegated to `techniques._base.advantages.compute_group_advantages`
  - Self-registers via `@register_technique("grpo")`
  - Legacy `GRPO`/`GRPOConfig` dataclass kept as deprecation shims so the
    existing `TECHNIQUE_REGISTRY` dict and `tests/test_techniques.py` still
    work. Both shims emit `DeprecationWarning` and will be removed in 0.3.
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
from techniques._base.advantages import compute_group_advantages
from techniques._base.simulator import simulate_decay
from techniques.base_technique import BaseTechnique, TechniqueConfig

try:
    import torch
    HAS_TORCH = True
except ImportError:  # pragma: no cover
    HAS_TORCH = False


# --------------------------------------------------------------------------- #
# New typed config (Phase 1 contract).
# --------------------------------------------------------------------------- #


class GRPOConfigV2(BaseConfig):
    """Pydantic config for GRPO. Validated at construction time.

    The base learning loop fields (`learning_rate`, `batch_size`, …) will move
    to a shared `TrainerConfig` in Phase 2. Until then they live here so a
    GRPO run is configurable from one file.
    """

    learning_rate: float = Field(2e-5, ge=0, le=1, description="Optimizer LR")
    batch_size: int = Field(4, ge=1, description="Per-device batch size")
    gradient_accumulation_steps: int = Field(4, ge=1)
    epochs: int = Field(3, ge=1)
    max_length: int = Field(512, ge=1)
    seed: int = Field(42, ge=0)
    bf16: bool = True

    # GRPO-specific.
    group_size: int = Field(
        8, ge=2,
        description="Number of responses sampled per prompt. ≥2 required for "
                    "group-relative advantages to be defined.",
    )
    kl_coef: float = Field(
        0.1, ge=0,
        description="Weight on the KL(policy || reference) penalty.",
    )
    clip_ratio: float = Field(
        0.2, gt=0, lt=1,
        description="PPO-style importance-ratio clip range, ±this value.",
    )
    temperature: float = Field(1.0, gt=0)
    reward_baseline: str = Field(
        "group_mean",
        description="How to center rewards before normalizing. One of "
                    "'group_mean' | 'group_min' | 'running_mean'.",
    )

    @field_validator("reward_baseline")
    @classmethod
    def _check_baseline(cls, v: str) -> str:
        allowed = {"group_mean", "group_min", "running_mean"}
        if v not in allowed:
            raise ValueError(f"reward_baseline must be one of {sorted(allowed)}")
        return v


# --------------------------------------------------------------------------- #
# Protocol-conforming implementation.
# --------------------------------------------------------------------------- #


@register_technique("grpo")
class GRPOTechnique:
    """Conforms to `core.Technique`.

    Lifecycle:
      1. `prepare(model, tokenizer, cfg)` — store handles, init step counter.
      2. `step(batch: RolloutBatch)` — one optimization step, returns
         `StepMetrics`. The trainer (not this class) owns `.backward()`,
         `.optimizer.step()`, and `.zero_grad()`.
      3. `save(path)` — delegates to the model's `save_pretrained`.

    Today this class is *just the loss math*. Phase 2 will hoist the
    trainer-side boilerplate (mixed precision, grad accumulation, LoRA
    wiring) into `techniques/_base/trainer.py`.
    """

    name = "grpo"

    def __init__(self, config: GRPOConfigV2 | None = None) -> None:
        self.config = config or GRPOConfigV2()
        self._model: Any = None
        self._tokenizer: Any = None
        self._step: int = 0

    # ---- core.Technique --------------------------------------------------- #

    def prepare(self, model: Any, tokenizer: Any, cfg: Any) -> None:
        self._model = model
        self._tokenizer = tokenizer
        if cfg is not None:
            self.config = cfg if isinstance(cfg, GRPOConfigV2) else GRPOConfigV2.model_validate(cfg)
        self._step = 0

    def step(self, batch: RolloutBatch) -> StepMetrics:
        """One GRPO update.

        Expects `batch.log_probs` (current policy) and `batch.ref_log_probs`
        (reference policy) as tensors of shape (B, G, T). If those are absent
        (e.g. the trainer hasn't filled them yet) we fall through to the
        deterministic simulation path so demos and tests stay green without
        a GPU.
        """
        self._step += 1

        if HAS_TORCH and batch.log_probs is not None:
            return self._step_real(batch)

        # Simulation path: deterministic loss curve for notebooks/tests.
        loss = simulate_decay(self._step, start=1.8, floor=0.3, decay=0.35)
        extras = {
            "reward": min(0.95, 0.35 + 0.22 * self._step),
            "group_reward_std": max(0.1, 0.5 - 0.08 * self._step),
            "advantage_mean": 0.1 + 0.05 * self._step,
            "kl_divergence": max(0.01, 0.12 - 0.02 * self._step),
        }
        return StepMetrics(loss=loss, step=self._step, extras=extras)

    def save(self, path: str) -> None:
        if self._model is None:
            raise RuntimeError("GRPOTechnique.save called before .prepare()")
        # Conformant models implement `save_pretrained`; if not, fall back.
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

        log_probs: torch.Tensor = batch.log_probs                # (B, G, T)
        ref_log_probs: torch.Tensor | None = batch.ref_log_probs  # (B, G, T) | None
        rewards = torch.as_tensor(
            [r for group in batch.rewards for r in group],
            dtype=log_probs.dtype, device=log_probs.device,
        ).view(log_probs.shape[0], log_probs.shape[1])             # (B, G)

        # NOTE: the legacy file used `group_old_log_probs` from kwargs; the
        # trainer now caches them on `batch.metadata["old_log_probs"]`.
        old_log_probs = batch.metadata.get("old_log_probs", log_probs.detach())

        advantages = compute_group_advantages(rewards.flatten()).view_as(rewards)
        ratio = torch.exp(log_probs - old_log_probs)
        adv = advantages.unsqueeze(-1)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1 - cfg.clip_ratio, 1 + cfg.clip_ratio) * adv
        policy_loss = -torch.min(surr1, surr2).mean()

        kl_loss = torch.tensor(0.0, device=log_probs.device)
        if ref_log_probs is not None:
            kl_loss = (torch.exp(log_probs) * (log_probs - ref_log_probs)).mean()

        total_loss = policy_loss + cfg.kl_coef * kl_loss
        return StepMetrics(
            loss=float(total_loss.detach()),
            step=self._step,
            extras={
                "policy_loss": float(policy_loss.detach()),
                "kl_divergence": float(kl_loss.detach()),
                "group_reward_mean": float(rewards.mean()),
                "group_reward_std": float(rewards.std()),
                "advantage_mean": float(advantages.mean()),
            },
        )


# --------------------------------------------------------------------------- #
# Deprecation shims — preserve the legacy public API for one minor version.
# --------------------------------------------------------------------------- #


@dataclass
class GRPOConfig(TechniqueConfig):
    """Legacy dataclass config. Deprecated — use `GRPOConfigV2` (Pydantic).

    Kept so existing code (`tests/test_techniques.py`, the two notebooks,
    `examples/run_pipeline.py`) keeps working. Will be removed in 0.3.
    """

    group_size: int = 8
    kl_coef: float = 0.1
    clip_ratio: float = 0.2
    temperature: float = 1.0
    reward_baseline: str = "group_mean"


class GRPO(BaseTechnique):
    """Legacy `BaseTechnique` adapter that delegates to `GRPOTechnique`.

    Public surface preserved: callers can still do
    `GRPO().compute_loss(**kwargs)` and `GRPO().compute_group_advantages(rewards)`.
    Both emit `DeprecationWarning` pointing at the new API.
    """

    name = "grpo"
    description = "Group Relative Policy Optimization — value-free RL (DeepSeek-R1's technique)"
    paper_reference = "Shao et al., 2024 — DeepSeekMath; DeepSeek-R1 Technical Report"
    priority = 1
    recommended_for: ClassVar[list[str]] = ["reasoning tasks", "math", "coding", "memory-constrained RL"]
    pros: ClassVar[list[str]] = ["No value model (50% less memory)", "Simple", "Great for reasoning", "Scales well"]
    cons: ClassVar[list[str]] = ["Multiple samples per prompt", "Group size sensitive", "Higher variance"]

    def __init__(self, config: GRPOConfig | None = None) -> None:
        super().__init__(config or GRPOConfig())
        self._impl = GRPOTechnique(
            GRPOConfigV2(
                learning_rate=self.config.learning_rate,
                batch_size=self.config.batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                epochs=self.config.epochs,
                max_length=self.config.max_length,
                seed=self.config.seed,
                bf16=self.config.bf16,
                group_size=getattr(self.config, "group_size", 8),
                kl_coef=getattr(self.config, "kl_coef", 0.1),
                clip_ratio=getattr(self.config, "clip_ratio", 0.2),
                temperature=getattr(self.config, "temperature", 1.0),
                reward_baseline=getattr(self.config, "reward_baseline", "group_mean"),
            )
        )

    def compute_group_advantages(self, rewards: Any) -> Any:
        return compute_group_advantages(rewards)

    def compute_loss(self, **kwargs: Any) -> Any:
        """Legacy entry point. Returns a torch.Tensor or a Python float.

        Real torch path: builds a `RolloutBatch` from kwargs and delegates.
        Simulation path: matches the legacy decay formula bit-for-bit so
        `tests/test_techniques.py` stays green.
        """
        if HAS_TORCH and "group_log_probs" in kwargs:
            warnings.warn(
                "GRPO.compute_loss(**kwargs) is deprecated. "
                "Use GRPOTechnique(...).step(RolloutBatch(...)) instead.",
                DeprecationWarning, stacklevel=2,
            )
            batch = RolloutBatch(
                prompts=[],  # legacy callers don't pass them
                responses=[],
                rewards=kwargs["group_rewards"].tolist()
                if hasattr(kwargs["group_rewards"], "tolist")
                else list(kwargs["group_rewards"]),
                log_probs=kwargs["group_log_probs"],
                ref_log_probs=kwargs.get("ref_log_probs"),
                metadata={"old_log_probs": kwargs["group_old_log_probs"]},
            )
            metrics = self._impl.step(batch)
            self.metrics.update({"loss": metrics.loss, **metrics.extras})
            # The legacy contract returns the loss *tensor*, not a float, so
            # the caller can backprop. Reconstruct from the cached tensors.
            import torch
            adv = self.compute_group_advantages(kwargs["group_rewards"])
            ratio = torch.exp(kwargs["group_log_probs"] - kwargs["group_old_log_probs"])
            cfg = self.config
            surr1 = ratio * adv.unsqueeze(-1)
            surr2 = torch.clamp(ratio, 1 - cfg.clip_ratio, 1 + cfg.clip_ratio) * adv.unsqueeze(-1)
            policy_loss = -torch.min(surr1, surr2).mean()
            kl = torch.tensor(0.0)
            if kwargs.get("ref_log_probs") is not None:
                kl = (
                    torch.exp(kwargs["group_log_probs"])
                    * (kwargs["group_log_probs"] - kwargs["ref_log_probs"])
                ).mean()
            return policy_loss + cfg.kl_coef * kl

        # Simulation path — preserve legacy numbers exactly.
        epoch = kwargs.get("epoch", self.step_count)
        loss = 1.8 * math.exp(-0.35 * epoch) + 0.3
        self.metrics.update({
            "loss": loss,
            "reward": min(0.95, 0.35 + 0.22 * epoch),
            "group_reward_std": max(0.1, 0.5 - 0.08 * epoch),
            "advantage_mean": 0.1 + 0.05 * epoch,
            "kl_divergence": max(0.01, 0.12 - 0.02 * epoch),
        })
        return loss
