"""Self-Play Fine-Tuning (SPIN).

Priority: 2 (Advanced technique)
Paper: "Self-Play Fine-Tuning Converts Weak Language Models to Strong" (Chen et al., 2024)

SPIN trains the model to distinguish human-written responses from its own
generations, iteratively. At iteration `t` the loss is the DPO sigmoid loss
with a specific pair construction:

  - chosen   = the human/SFT response,
  - rejected = the model's own generation for the same prompt, sampled from
               the *previous* iterate ("opponent"),
  - reference model = that same previous iterate.

so `L = −log σ(β·[(logπ_θ(y_h) − logπ_opp(y_h)) − (logπ_θ(y_g) − logπ_opp(y_g))])`.
When the discriminator can no longer tell its own outputs from human text,
the policy has converged onto the SFT distribution's strengths.

The generation/snapshot cadence (sample from opponent, train, promote the
trained model to be the next opponent) belongs to the trainer loop; this
module owns the loss, the pair construction (`build_spin_pairs`), and the
iteration bookkeeping (`advance_iteration`).

This file follows the DPO/GRPO migration template: a typed Pydantic config
(`SPINConfigV2`), a `core.Technique`-conforming `SPINTechnique` self-registered
via `@register_technique`, and the legacy `SPIN` / `SPINConfig` dataclass
surface kept as a thin adapter so existing callers keep working.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
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
    import torch  # noqa: F401  (availability probe; real use is inside SPINTechnique methods)
    HAS_TORCH = True
except ImportError:  # pragma: no cover
    HAS_TORCH = False


# --------------------------------------------------------------------------- #
# Pair construction — the part of SPIN that is not "just DPO".
# --------------------------------------------------------------------------- #


def build_spin_pairs(
    prompts: Sequence[str],
    human_responses: Sequence[str],
    generated_responses: Sequence[str],
    *,
    iteration: int = 0,
) -> list[PreferencePair]:
    """Turn (prompt, human, self-generated) triples into SPIN preference pairs.

    chosen is always the human/SFT response; rejected is always the model's
    own generation. The iteration index rides along in metadata so training
    logs can distinguish self-play rounds.
    """
    if not (len(prompts) == len(human_responses) == len(generated_responses)):
        raise ValueError(
            "prompts, human_responses, generated_responses must have equal "
            f"lengths, got {len(prompts)}/{len(human_responses)}/{len(generated_responses)}"
        )
    return [
        PreferencePair(
            prompt=p,
            chosen=h,
            rejected=g,
            metadata={"spin_iteration": iteration},
        )
        for p, h, g in zip(prompts, human_responses, generated_responses, strict=True)
    ]


# --------------------------------------------------------------------------- #
# New typed config (Phase 2 contract).
# --------------------------------------------------------------------------- #


class SPINConfigV2(BaseConfig):
    """Pydantic config for SPIN. Validated at construction time."""

    learning_rate: float = Field(2e-5, ge=0, le=1, description="Optimizer LR")
    batch_size: int = Field(4, ge=1, description="Per-device batch size")
    gradient_accumulation_steps: int = Field(4, ge=1)
    epochs: int = Field(3, ge=1)
    max_length: int = Field(512, ge=1)
    seed: int = Field(42, ge=0)
    bf16: bool = True

    # SPIN-specific.
    beta: float = Field(
        0.1, ge=0,
        description="Temperature on the policy/opponent log-ratio, exactly "
                    "DPO's β with the opponent as reference.",
    )
    num_iterations: int = Field(
        3, ge=1,
        description="Self-play rounds. After each round the trainer promotes "
                    "the current policy to be the next opponent.",
    )


# --------------------------------------------------------------------------- #
# Protocol-conforming implementation.
# --------------------------------------------------------------------------- #


@register_technique("spin")
class SPINTechnique:
    """Conforms to `core.Technique`.

    Consumes one `PreferencePair` per `step()` call, built by
    `build_spin_pairs` (chosen = human, rejected = self-generated). For the
    real path the pair's `metadata` must carry `chosen_logps` /
    `rejected_logps` under the current policy and `ref_chosen_logps` /
    `ref_rejected_logps` under the opponent (previous iterate). Without
    tensors (or without torch) the step falls back to the deterministic
    simulation so demos run anywhere.
    """

    name = "spin"

    def __init__(self, config: SPINConfigV2 | None = None) -> None:
        self.config = config or SPINConfigV2()
        self._model: Any = None
        self._tokenizer: Any = None
        self._step: int = 0
        self._iteration: int = 0

    # ---- core.Technique --------------------------------------------------- #

    def prepare(self, model: Any, tokenizer: Any, cfg: Any) -> None:
        self._model = model
        self._tokenizer = tokenizer
        if cfg is not None:
            self.config = cfg if isinstance(cfg, SPINConfigV2) else SPINConfigV2.model_validate(cfg)
        self._step = 0
        self._iteration = 0

    @property
    def iteration(self) -> int:
        """Current self-play round (0-based)."""
        return self._iteration

    def advance_iteration(self) -> int:
        """Move to the next self-play round.

        Called by the trainer *after* it snapshots the current policy as the
        new opponent and regenerates the rejected responses. Raises once the
        configured number of rounds is exhausted so a driver loop can't
        silently overrun.
        """
        if self._iteration + 1 >= self.config.num_iterations:
            raise RuntimeError(
                f"SPIN configured for {self.config.num_iterations} iterations; "
                f"round {self._iteration} was the last."
            )
        self._iteration += 1
        return self._iteration

    def step(self, batch: PreferencePair) -> StepMetrics:
        self._step += 1

        md = batch.metadata
        if HAS_TORCH and "chosen_logps" in md and "rejected_logps" in md:
            return self._step_real(batch)

        # Simulation path — replicate the legacy SPIN decay constants exactly.
        loss = simulate_decay(self._step, start=1.1, floor=0.25, decay=0.4)
        epoch = self._step
        extras = {
            "reward": min(0.85, 0.35 + 0.16 * epoch),
            "discrimination_accuracy": min(0.92, 0.55 + 0.11 * epoch),
            "spin_iteration": float(self._iteration),
        }
        return StepMetrics(loss=loss, step=self._step, extras=extras)

    def save(self, path: str) -> None:
        if self._model is None:
            raise RuntimeError("SPINTechnique.save called before .prepare()")
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
        chosen_logps = md["chosen_logps"]          # human response, current policy
        rejected_logps = md["rejected_logps"]      # self-generated, current policy
        ref_chosen_logps = md.get("ref_chosen_logps")      # human, opponent
        ref_rejected_logps = md.get("ref_rejected_logps")  # self-generated, opponent

        loss = pairwise_ratio_loss(
            chosen_logps,
            rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
            beta=cfg.beta,
            loss_type=RatioLossType.SIGMOID,
        )

        if ref_chosen_logps is not None and ref_rejected_logps is not None:
            chosen_reward = (cfg.beta * (chosen_logps - ref_chosen_logps)).detach()
            rejected_reward = (cfg.beta * (rejected_logps - ref_rejected_logps)).detach()
        else:
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
                # In SPIN terms: how often the implicit discriminator ranks
                # human text above the model's own generation.
                "discrimination_accuracy": float((margin > 0).float().mean()),
                "spin_iteration": float(md.get("spin_iteration", self._iteration)),
            },
        )


# --------------------------------------------------------------------------- #
# Legacy surface — thin adapter over the real implementation.
# --------------------------------------------------------------------------- #


@dataclass
class SPINConfig(TechniqueConfig):
    """Legacy dataclass config. Prefer `SPINConfigV2` (Pydantic) for new code."""

    beta: float = 0.1
    num_iterations: int = 3


class SPIN(BaseTechnique):
    """Legacy `BaseTechnique` adapter.

    `compute_loss(chosen_log_probs=…, rejected_log_probs=…)` runs the real
    SPIN loss when torch tensors are supplied (chosen = human response,
    rejected = self-generated, ref_* = opponent model); otherwise it
    reproduces the legacy simulation numbers bit-for-bit.
    """

    name = "spin"
    description = "Self-Play Fine-Tuning — distinguish model outputs from human text"
    paper_reference = "Chen et al., 2024 — Self-Play Fine-Tuning Converts Weak LMs to Strong"
    priority = 2
    recommended_for: ClassVar[list[str]] = ["self-improvement", "SFT data only", "iterative refinement"]
    pros: ClassVar[list[str]] = ["Only needs SFT data", "Self-improving", "Simple iterative process"]
    cons: ClassVar[list[str]] = ["Convergence can be slow", "May plateau early"]

    def __init__(self, config: SPINConfig | None = None):
        super().__init__(config or SPINConfig())
        self._impl = SPINTechnique(
            SPINConfigV2(
                learning_rate=self.config.learning_rate,
                batch_size=self.config.batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                epochs=self.config.epochs,
                max_length=self.config.max_length,
                seed=self.config.seed,
                bf16=self.config.bf16,
                beta=getattr(self.config, "beta", 0.1),
                num_iterations=getattr(self.config, "num_iterations", 3),
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
                beta=cfg.beta,
                loss_type=RatioLossType.SIGMOID,
            )

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
                "discrimination_accuracy": float((margin > 0).float().mean()),
            })
            return loss

        # Simulation path — preserve legacy numbers exactly.
        epoch = kwargs.get("epoch", self.step_count)
        loss = 1.1 * math.exp(-0.4 * epoch) + 0.25
        self.metrics.update({
            "loss": loss,
            "reward": min(0.85, 0.35 + 0.16 * epoch),
            "discrimination_accuracy": min(0.92, 0.55 + 0.11 * epoch),
        })
        return loss
