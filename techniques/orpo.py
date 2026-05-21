"""Odds Ratio Preference Optimization (ORPO).

Priority: 2 (Advanced; powering the Phase 5 end-to-end demo)
Paper: "ORPO: Monolithic Preference Optimization without Reference Model"
       (Hong, Lee, & Thorne, 2024 — https://arxiv.org/abs/2403.07691)

Key insight
-----------
DPO and friends need a frozen reference model and a *separate* SFT stage.
ORPO folds preference alignment into the SFT loss with a single odds-ratio
regularizer, so:

    L_ORPO(x, y_w, y_l) = L_SFT(y_w | x)  +  λ · L_OR(x, y_w, y_l)
    L_OR              = -log σ( log_odds(y_w|x)  -  log_odds(y_l|x) )
    log_odds(y|x)     = log p(y|x) - log( 1 - p(y|x) )

Sums of token log-probs stand in for `log p(y|x)` in the implementation; the
`log(1 - p)` half is the numerically delicate part (see `_log1mexp` below).

When to use
-----------
- You want a single-stage post-training run on preference pairs and cannot
  afford the memory of carrying both policy and ref model in VRAM.
- Your base model is small-to-mid (≤7B); ORPO's regularizer is gentler than
  DPO's and tends to behave on weaker priors.
- You're shipping the Phase 5 demo on Qwen2.5-0.5B + UltraFeedback — that's
  exactly the setting Hong et al. validate.

Pros
----
- No reference model => roughly DPO/2 memory.
- No multi-stage pipeline; you train one loss, period.
- Empirically competitive with DPO + SFT on AlpacaEval / MT-Bench.

Cons
----
- Newer, smaller body of replication studies than DPO.
- Sensitive to `λ`: too small => degrades to pure SFT, too large => SFT
  signal drowned out.
- log(1-p) tricks make low-temperature decoding fragile if implemented naïvely.

Phase 2 migration
-----------------
This module exposes the new Phase-2 contract (`ORPOConfigV2` Pydantic config,
`ORPOTechnique` Protocol-conforming class self-registered as ``"orpo"``)
while keeping the legacy ``ORPOConfig`` dataclass and ``ORPO`` BaseTechnique
adapter alive as deprecation shims. Shims will be removed in 0.3.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any, ClassVar

from pydantic import Field, model_validator

from core.config import BaseConfig
from core.registry import register_technique
from core.types import PreferencePair, StepMetrics
from techniques._base.simulator import simulate_decay
from techniques.base_technique import BaseTechnique, TechniqueConfig

try:
    import torch
    import torch.nn.functional as F

    HAS_TORCH = True
except ImportError:  # pragma: no cover
    HAS_TORCH = False


# --------------------------------------------------------------------------- #
# New typed config (Phase 2 contract).
# --------------------------------------------------------------------------- #


class ORPOConfigV2(BaseConfig):
    """Pydantic config for ORPO. Validated at construction time.

    The base learning loop fields (`learning_rate`, `batch_size`, …) will move
    to a shared `TrainerConfig` in a later phase. Until then they live here so
    an ORPO run is fully described by one file.

    ORPO-specific fields
    --------------------
    `lambda_or`
        Weight on the odds-ratio regularizer term. Hong et al. use 0.1 on
        UltraFeedback and 1.0 on HH-RLHF; we default to 0.1 to match the
        UltraFeedback recipe the Phase 5 demo will run.
    `beta`
        Back-compat alias for `lambda_or`. Some early ORPO drafts (and our
        own pre-Phase-2 notebooks) used "beta" as the term weight; passing
        both fields is an error.
    """

    # Shared base.
    learning_rate: float = Field(2e-5, ge=0, le=1, description="Optimizer LR")
    batch_size: int = Field(4, ge=1, description="Per-device batch size")
    gradient_accumulation_steps: int = Field(4, ge=1, description="Grad accumulation steps")
    epochs: int = Field(3, ge=1, description="Number of training epochs")
    max_length: int = Field(512, ge=1, description="Max sequence length (tokens)")
    seed: int = Field(42, ge=0, description="Random seed for reproducibility")
    bf16: bool = Field(True, description="Use bf16 mixed precision when supported")

    # ORPO-specific.
    lambda_or: float = Field(
        0.1,
        ge=0,
        description="Weight on the odds-ratio regularizer term. 0 disables the "
        "preference signal, reducing ORPO to vanilla SFT on the chosen response.",
    )
    beta: float | None = Field(
        None,
        ge=0,
        description="Deprecated alias for `lambda_or`. Supplying both is an error; "
        "supplying only this is accepted with a DeprecationWarning.",
    )

    @model_validator(mode="after")
    def _reconcile_beta_alias(self) -> ORPOConfigV2:
        # Pydantic's `frozen=True` blocks direct assignment after construction.
        # We use object.__setattr__ to handle the alias migration during the
        # post-init validator, then leave the model immutable afterwards.
        if self.beta is not None:
            # Detect "both supplied explicitly". `lambda_or` has a non-None
            # default, so we treat any user-supplied beta together with a
            # non-default lambda_or as a conflict.
            if self.lambda_or != 0.1:
                raise ValueError(
                    "ORPOConfigV2: pass either `lambda_or` (preferred) or `beta` "
                    "(deprecated alias) — not both."
                )
            warnings.warn(
                "ORPOConfigV2.beta is deprecated; use `lambda_or` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            object.__setattr__(self, "lambda_or", float(self.beta))
            object.__setattr__(self, "beta", None)
        return self


# --------------------------------------------------------------------------- #
# Numerically stable log(1 - exp(x)) helper.
# --------------------------------------------------------------------------- #


def _log1mexp(x: torch.Tensor) -> torch.Tensor:
    """Compute ``log(1 - exp(x))`` for x <= 0, stably.

    The two-branch identity from Mächler (2012, "Accurately Computing
    log(1 - exp(-a))"):

        log(1 - exp(x)) = log(-expm1(x))     if x > -log(2)
                       = log1p(-exp(x))      otherwise

    The first branch dodges catastrophic cancellation when ``exp(x)`` is close
    to 1 (x near 0); the second handles the tail where ``exp(x)`` is tiny.
    Inputs are clamped to a small negative ceiling because at x == 0 the true
    answer is -inf and most callers want a finite gradient.
    """
    # Clamp to avoid log(0) = -inf blowing up the loss; -1e-7 is the smallest
    # negative value safe under fp32 expm1 without going to -inf.
    x_safe = torch.clamp(x, max=-1e-7)
    threshold = -math.log(2.0)
    return torch.where(
        x_safe > threshold,
        torch.log(-torch.expm1(x_safe)),
        torch.log1p(-torch.exp(x_safe)),
    )


# --------------------------------------------------------------------------- #
# Shared loss computation — single source of truth used by both the new
# `ORPOTechnique.step` and the legacy `ORPO.compute_loss` shim.
# --------------------------------------------------------------------------- #


def _orpo_loss_components(
    chosen_logps: torch.Tensor,
    rejected_logps: torch.Tensor,
    *,
    lambda_or: float,
    sft_loss: torch.Tensor | None = None,
    chosen_token_count: torch.Tensor | float | None = None,
) -> dict[str, torch.Tensor]:
    """Compute every piece of the ORPO loss in one place.

    Paper formula (Hong et al., 2024, Eq. 6):

        L_ORPO = L_SFT(y_w) + λ * (-log σ( log_odds(y_w) - log_odds(y_l) ))

    with ``log_odds(y) = log p(y) - log(1 - p(y))``. Here ``log p(y)`` is the
    sequence log-prob (sum over response tokens), which is what
    `chosen_logps` / `rejected_logps` carry.

    Args:
        chosen_logps, rejected_logps: per-example summed token log-probs,
            shape ``(B,)`` each. Values must be non-positive.
        lambda_or: weight on the odds-ratio regularizer.
        sft_loss: optionally precomputed SFT loss tensor (scalar). When omitted
            we fall back to ``-chosen_logps.mean()`` normalized by
            ``chosen_token_count`` if provided.
        chosen_token_count: per-example token count for SFT normalization.
            Either a scalar or shape ``(B,)``. Only used when `sft_loss` is None.

    Returns:
        dict with keys ``loss``, ``sft_loss``, ``or_loss``,
        ``log_odds_chosen``, ``log_odds_rejected``, ``accuracy``. All tensors;
        ``accuracy`` is a scalar in [0, 1].
    """
    # log_odds(y|x) = log p - log(1 - p)
    log_odds_chosen = chosen_logps - _log1mexp(chosen_logps)
    log_odds_rejected = rejected_logps - _log1mexp(rejected_logps)

    # L_OR = -log sigmoid( log_odds_chosen - log_odds_rejected )
    or_loss = -F.logsigmoid(log_odds_chosen - log_odds_rejected).mean()

    # SFT term. If the trainer precomputed a real cross-entropy on the chosen
    # response, prefer that. Otherwise approximate via summed logps.
    if sft_loss is None:
        if chosen_token_count is None:
            sft = -chosen_logps.mean()
        else:
            tok = (
                chosen_token_count
                if isinstance(chosen_token_count, torch.Tensor)
                else torch.as_tensor(
                    chosen_token_count, dtype=chosen_logps.dtype, device=chosen_logps.device
                )
            )
            sft = -(chosen_logps / torch.clamp(tok, min=1.0)).mean()
    else:
        sft = sft_loss

    total = sft + lambda_or * or_loss

    with torch.no_grad():
        accuracy = (log_odds_chosen > log_odds_rejected).float().mean()

    return {
        "loss": total,
        "sft_loss": sft,
        "or_loss": or_loss,
        "log_odds_chosen": log_odds_chosen,
        "log_odds_rejected": log_odds_rejected,
        "accuracy": accuracy,
    }


# --------------------------------------------------------------------------- #
# Protocol-conforming implementation.
# --------------------------------------------------------------------------- #


@register_technique("orpo")
class ORPOTechnique:
    """Conforms to `core.Technique`.

    Lifecycle:
      1. `prepare(model, tokenizer, cfg)` — store handles, init step counter.
      2. `step(batch: PreferencePair)` — one optimization step, returns
         `StepMetrics`. The trainer (not this class) owns `.backward()`,
         `.optimizer.step()`, and `.zero_grad()`.
      3. `save(path)` — delegates to the model's `save_pretrained`.

    Today this class is *just the loss math*. Trainer-side boilerplate (mixed
    precision, grad accumulation, LoRA wiring) will move into
    `techniques/_base/trainer.py` in a later phase.

    The real-torch path expects the trainer to have already computed per-
    example summed log-probs for chosen/rejected responses and attached them
    to ``batch.metadata`` as `"chosen_logps"` / `"rejected_logps"`. Optional
    keys: `"sft_loss"` (precomputed CE on chosen), `"chosen_token_count"`.
    """

    name = "orpo"

    def __init__(self, config: ORPOConfigV2 | None = None) -> None:
        self.config = config or ORPOConfigV2()
        self._model: Any = None
        self._tokenizer: Any = None
        self._step: int = 0

    # ---- core.Technique --------------------------------------------------- #

    def prepare(self, model: Any, tokenizer: Any, cfg: Any) -> None:
        self._model = model
        self._tokenizer = tokenizer
        if cfg is not None:
            self.config = (
                cfg if isinstance(cfg, ORPOConfigV2) else ORPOConfigV2.model_validate(cfg)
            )
        self._step = 0

    def step(self, batch: PreferencePair) -> StepMetrics:
        """One ORPO update.

        Real path: read tensors from `batch.metadata` and compute the full
        ORPO loss (Hong et al. 2024, Eq. 6). Sim path: deterministic decay
        with a plausible accuracy curve.
        """
        self._step += 1

        if HAS_TORCH and self._has_real_tensors(batch):
            return self._step_real(batch)

        # Simulation path: deterministic loss curve for notebooks/tests.
        # Decay constants match the legacy `ORPO.compute_loss` simulation so
        # downstream baselines don't shift when the new path is exercised.
        loss = simulate_decay(self._step, start=1.4, floor=0.28, decay=0.38)
        accuracy = min(0.9, 0.5 + 0.05 * self._step)
        extras = {
            "sft_loss": loss * 0.7,
            "or_loss": loss * 0.3,
            "log_odds_chosen_mean": -0.2 + 0.05 * self._step,
            "log_odds_rejected_mean": -0.5 - 0.02 * self._step,
            "accuracy": accuracy,
        }
        return StepMetrics(loss=loss, step=self._step, extras=extras)

    def save(self, path: str) -> None:
        if self._model is None:
            raise RuntimeError("ORPOTechnique.save called before .prepare()")
        saver = getattr(self._model, "save_pretrained", None)
        if saver is not None:
            saver(path)
            return
        import torch as _torch

        _torch.save(self._model.state_dict(), path)

    # ---- real-tensor step ------------------------------------------------- #

    @staticmethod
    def _has_real_tensors(batch: PreferencePair) -> bool:
        md = batch.metadata or {}
        return "chosen_logps" in md and "rejected_logps" in md

    def _step_real(self, batch: PreferencePair) -> StepMetrics:
        cfg = self.config
        md = batch.metadata

        chosen_logps = md["chosen_logps"]
        rejected_logps = md["rejected_logps"]
        sft_loss = md.get("sft_loss")
        chosen_token_count = md.get("chosen_token_count")

        parts = _orpo_loss_components(
            chosen_logps,
            rejected_logps,
            lambda_or=cfg.lambda_or,
            sft_loss=sft_loss,
            chosen_token_count=chosen_token_count,
        )

        return StepMetrics(
            loss=float(parts["loss"].detach()),
            step=self._step,
            extras={
                "sft_loss": float(parts["sft_loss"].detach()),
                "or_loss": float(parts["or_loss"].detach()),
                "log_odds_chosen_mean": float(parts["log_odds_chosen"].detach().mean()),
                "log_odds_rejected_mean": float(parts["log_odds_rejected"].detach().mean()),
                "accuracy": float(parts["accuracy"].detach()),
            },
        )


# --------------------------------------------------------------------------- #
# Deprecation shims — preserve the legacy public API for one minor version.
# --------------------------------------------------------------------------- #


@dataclass
class ORPOConfig(TechniqueConfig):
    """Legacy dataclass config. Deprecated — use `ORPOConfigV2` (Pydantic).

    Kept so existing code (`tests/test_techniques.py`, the two notebooks,
    `examples/`) keeps working. Will be removed in 0.3.

    `lambda_orpo` is the original field name; `lambda_or` is the new
    canonical spelling. Both refer to the same weight on the odds-ratio
    regularizer.
    """

    lambda_orpo: float = 0.1  # Weight of odds ratio loss (legacy spelling)


class ORPO(BaseTechnique):
    """Legacy `BaseTechnique` adapter that delegates to `ORPOTechnique`.

    Public surface preserved: callers can still do
    `ORPO().compute_loss(**kwargs)` and `ORPO().train_step(epoch=…)`. The
    torch-kwarg path emits `DeprecationWarning` pointing at the new API.
    """

    name = "orpo"
    description = "ORPO — combined SFT + preference alignment, no reference model"
    paper_reference = "Hong et al., 2024 — ORPO: Monolithic Preference Optimization"
    priority = 2
    recommended_for: ClassVar[list[str]] = ["single-stage alignment", "no reference model", "efficient training"]
    pros: ClassVar[list[str]] = ["No reference model", "Combined SFT + alignment", "Memory efficient"]
    cons: ClassVar[list[str]] = ["Less flexible", "Newer, less battle-tested"]

    def __init__(self, config: ORPOConfig | None = None) -> None:
        super().__init__(config or ORPOConfig())
        self._impl = ORPOTechnique(
            ORPOConfigV2(
                learning_rate=self.config.learning_rate,
                batch_size=self.config.batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                epochs=self.config.epochs,
                max_length=self.config.max_length,
                seed=self.config.seed,
                bf16=self.config.bf16,
                lambda_or=getattr(self.config, "lambda_orpo", 0.1),
            )
        )

    def compute_loss(self, **kwargs: Any) -> Any:
        """Legacy entry point. Returns a torch.Tensor or a Python float.

        Real torch path: rebuilds the ORPO loss via the shared helper so the
        legacy and new implementations stay bit-identical.
        Simulation path: matches the legacy decay formula bit-for-bit.
        """
        cfg = self.config

        if HAS_TORCH and "chosen_log_probs" in kwargs:
            warnings.warn(
                "ORPO.compute_loss(**kwargs) is deprecated. "
                "Use ORPOTechnique(...).step(PreferencePair(...)) instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            chosen_lp = kwargs["chosen_log_probs"]
            rejected_lp = kwargs["rejected_log_probs"]
            parts = _orpo_loss_components(
                chosen_lp,
                rejected_lp,
                lambda_or=getattr(cfg, "lambda_orpo", 0.1),
                sft_loss=kwargs.get("sft_loss"),
                chosen_token_count=kwargs.get("chosen_token_count"),
            )
            self.metrics.update(
                {
                    "loss": float(parts["loss"].detach()),
                    "sft_loss": float(parts["sft_loss"].detach()),
                    "or_loss": float(parts["or_loss"].detach()),
                    "accuracy": float(parts["accuracy"].detach()),
                }
            )
            return parts["loss"]

        # Simulation path — preserve legacy numbers exactly.
        epoch = kwargs.get("epoch", self.step_count)
        loss = 1.4 * math.exp(-0.38 * epoch) + 0.28
        self.metrics.update({"loss": loss, "reward": min(0.88, 0.32 + 0.17 * epoch)})
        return loss
