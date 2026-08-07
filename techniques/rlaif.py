"""RL from AI Feedback (RLAIF).

Priority: 2 (Advanced technique)
Papers: "Constitutional AI: Harmlessness from AI Feedback" (Bai et al., 2022);
        "RLAIF: Scaling Reinforcement Learning from Human Feedback with AI
        Feedback" (Lee et al., 2023)

RLAIF replaces the human preference labeler with an AI judge: candidate
responses are sampled per prompt, the judge picks chosen/rejected, and the
policy is then optimized on those AI-labeled pairs with a standard
preference loss. The constitution enters through the judge — its scoring
prompt/criteria — not through this module.

The module owns three pieces:

  1. `Judge` — the labeler protocol. Anything callable as
     `judge(prompt, response_a, response_b) -> 0 | 1` (index of the
     *preferred* response) works: an LLM API call, a reward model, a rubric.
  2. `label_pairs(judge, prompt, responses)` — round-robin candidate
     comparison producing `PreferencePair`s with judge metadata attached.
  3. `RLAIFTechnique` — DPO-style optimization over the labeled pairs
     (`pairwise_ratio_loss`, sigmoid shape), same migration template as
     DPO/GRPO/SimPO/IPO/SPIN.

`ScoreJudge` adapts a plain scoring function `(prompt, response) -> float`
into a `Judge`. The full Constitutional-AI critique→revision loop is a data
*generation* concern and lives with the trainer/data layer, not here.

This file follows the DPO/GRPO migration template: a typed Pydantic config
(`RLAIFConfigV2`), a `core.Technique`-conforming `RLAIFTechnique`
self-registered via `@register_technique`, and the legacy `RLAIF` /
`RLAIFConfig` dataclass surface kept as a thin adapter so existing callers
keep working.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol, runtime_checkable

from pydantic import Field

from core.config import BaseConfig
from core.registry import register_technique
from core.types import PreferencePair, StepMetrics
from techniques._base.ratio_loss import RatioLossType, pairwise_ratio_loss
from techniques._base.simulator import simulate_decay
from techniques.base_technique import BaseTechnique, TechniqueConfig

try:
    import torch  # noqa: F401  (availability probe; real use is inside RLAIFTechnique methods)
    HAS_TORCH = True
except ImportError:  # pragma: no cover
    HAS_TORCH = False


# --------------------------------------------------------------------------- #
# The AI labeler — the part of RLAIF that is not "just DPO".
# --------------------------------------------------------------------------- #


@runtime_checkable
class Judge(Protocol):
    """AI preference labeler.

    Returns the index (0 or 1) of the preferred response. Implementations
    range from an LLM prompted with a constitution to a trained reward
    model to a hand-written rubric.
    """

    def __call__(self, prompt: str, response_a: str, response_b: str) -> int: ...


class ScoreJudge:
    """Adapts a scoring function `(prompt, response) -> float` into a `Judge`.

    Ties break toward the first response so labeling is deterministic.
    This is the easiest on-ramp: point it at a reward model's forward pass
    or an LLM "rate this 1–10" call and you have an RLAIF labeler.
    """

    def __init__(self, score_fn: Callable[[str, str], float]) -> None:
        self._score_fn = score_fn

    def __call__(self, prompt: str, response_a: str, response_b: str) -> int:
        return 0 if self._score_fn(prompt, response_a) >= self._score_fn(prompt, response_b) else 1


def label_pairs(
    judge: Judge,
    prompt: str,
    responses: Sequence[str],
) -> list[PreferencePair]:
    """Compare candidate responses pairwise and emit AI-labeled preference pairs.

    Adjacent round-robin (r0 vs r1, r1 vs r2, …) rather than all-pairs:
    n−1 judge calls instead of n·(n−1)/2, which matters when the judge is a
    paid API. Judge identity is recorded in pair metadata for provenance.
    """
    if len(responses) < 2:
        raise ValueError(f"need at least 2 candidate responses, got {len(responses)}")
    pairs: list[PreferencePair] = []
    judge_name = type(judge).__name__
    for a, b in itertools.pairwise(responses):
        preferred = judge(prompt, a, b)
        if preferred not in (0, 1):
            raise ValueError(f"judge must return 0 or 1, got {preferred!r}")
        chosen, rejected = (a, b) if preferred == 0 else (b, a)
        pairs.append(
            PreferencePair(
                prompt=prompt,
                chosen=chosen,
                rejected=rejected,
                metadata={"labeler": "ai", "judge": judge_name},
            )
        )
    return pairs


# --------------------------------------------------------------------------- #
# New typed config (Phase 2 contract).
# --------------------------------------------------------------------------- #


class RLAIFConfigV2(BaseConfig):
    """Pydantic config for RLAIF. Validated at construction time."""

    learning_rate: float = Field(2e-5, ge=0, le=1, description="Optimizer LR")
    batch_size: int = Field(4, ge=1, description="Per-device batch size")
    gradient_accumulation_steps: int = Field(4, ge=1)
    epochs: int = Field(3, ge=1)
    max_length: int = Field(512, ge=1)
    seed: int = Field(42, ge=0)
    bf16: bool = True

    # RLAIF-specific.
    beta: float = Field(
        0.1, ge=0,
        description="Temperature on the policy/ref log-ratio for the "
                    "preference loss over AI-labeled pairs.",
    )
    num_candidates: int = Field(
        2, ge=2,
        description="Candidate responses sampled per prompt for the judge "
                    "to compare.",
    )
    sampling_temperature: float = Field(
        0.7, ge=0,
        description="Temperature for candidate sampling (used by the "
                    "trainer's generation stage).",
    )


# --------------------------------------------------------------------------- #
# Protocol-conforming implementation.
# --------------------------------------------------------------------------- #


@register_technique("rlaif")
class RLAIFTechnique:
    """Conforms to `core.Technique`.

    Consumes one AI-labeled `PreferencePair` per `step()` call — typically
    produced by `label_pairs(judge, …)`. For the real path the pair's
    `metadata` must carry `chosen_logps` / `rejected_logps` (and optionally
    `ref_chosen_logps` / `ref_rejected_logps`) tensors. Without tensors (or
    without torch) the step falls back to the deterministic simulation so
    demos run anywhere.
    """

    name = "rlaif"

    def __init__(
        self,
        config: RLAIFConfigV2 | None = None,
        *,
        judge: Judge | None = None,
    ) -> None:
        self.config = config or RLAIFConfigV2()
        self.judge = judge
        self._model: Any = None
        self._tokenizer: Any = None
        self._step: int = 0

    # ---- core.Technique --------------------------------------------------- #

    def prepare(self, model: Any, tokenizer: Any, cfg: Any) -> None:
        self._model = model
        self._tokenizer = tokenizer
        if cfg is not None:
            self.config = cfg if isinstance(cfg, RLAIFConfigV2) else RLAIFConfigV2.model_validate(cfg)
        self._step = 0

    def label(self, prompt: str, responses: Sequence[str]) -> list[PreferencePair]:
        """Label candidate responses with the attached judge."""
        if self.judge is None:
            raise RuntimeError(
                "RLAIFTechnique has no judge attached. Pass judge= at "
                "construction (e.g. ScoreJudge(reward_model_score)) before "
                "labeling."
            )
        return label_pairs(self.judge, prompt, responses)

    def step(self, batch: PreferencePair) -> StepMetrics:
        self._step += 1

        md = batch.metadata
        if HAS_TORCH and "chosen_logps" in md and "rejected_logps" in md:
            return self._step_real(batch)

        # Simulation path — replicate the legacy RLAIF decay constants exactly.
        loss = simulate_decay(self._step, start=1.6, floor=0.35, decay=0.32)
        epoch = self._step
        extras = {
            "reward": min(0.88, 0.28 + 0.19 * epoch),
            "harmlessness": min(0.95, 0.5 + 0.12 * epoch),
            "helpfulness": min(0.9, 0.4 + 0.15 * epoch),
        }
        return StepMetrics(loss=loss, step=self._step, extras=extras)

    def save(self, path: str) -> None:
        if self._model is None:
            raise RuntimeError("RLAIFTechnique.save called before .prepare()")
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
                # Agreement between the policy's implicit reward and the AI
                # judge's label.
                "judge_agreement": float((margin > 0).float().mean()),
            },
        )


# --------------------------------------------------------------------------- #
# Legacy surface — thin adapter over the real implementation.
# --------------------------------------------------------------------------- #


@dataclass
class RLAIFConfig(TechniqueConfig):
    """Legacy dataclass config. Prefer `RLAIFConfigV2` (Pydantic) for new code.

    The critique/revision knobs are kept for backward compatibility; the
    Constitutional-AI revision loop is a data-generation concern handled
    upstream of this technique.
    """

    num_principles: int = 16
    critique_temperature: float = 0.7
    revision_temperature: float = 0.5
    num_revisions: int = 2


class RLAIF(BaseTechnique):
    """Legacy `BaseTechnique` adapter.

    `compute_loss(chosen_log_probs=…, rejected_log_probs=…)` runs the real
    preference loss over AI-labeled pairs when torch tensors are supplied;
    otherwise it reproduces the legacy simulation numbers bit-for-bit.
    """

    name = "rlaif"
    description = "RL from AI Feedback — AI-judge-labeled preference optimization"
    paper_reference = "Bai et al., 2022 — Constitutional AI; Lee et al., 2023 — RLAIF"
    priority = 2
    recommended_for: ClassVar[list[str]] = ["safety alignment", "scalable oversight", "principle-based training"]
    pros: ClassVar[list[str]] = ["No human labelers needed", "Scalable", "Principle-driven"]
    cons: ClassVar[list[str]] = ["AI judge quality limits ceiling", "Constitutional principles need careful design"]

    def __init__(self, config: RLAIFConfig | None = None):
        super().__init__(config or RLAIFConfig())
        self._impl = RLAIFTechnique(
            RLAIFConfigV2(
                learning_rate=self.config.learning_rate,
                batch_size=self.config.batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                epochs=self.config.epochs,
                max_length=self.config.max_length,
                seed=self.config.seed,
                bf16=self.config.bf16,
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
                "judge_agreement": float((margin > 0).float().mean()),
            })
            return loss

        # Simulation path — preserve legacy numbers exactly.
        epoch = kwargs.get("epoch", self.step_count)
        loss = 1.6 * math.exp(-0.32 * epoch) + 0.35
        self.metrics.update({
            "loss": loss,
            "reward": min(0.88, 0.28 + 0.19 * epoch),
            "harmlessness": min(0.95, 0.5 + 0.12 * epoch),
            "helpfulness": min(0.9, 0.4 + 0.15 * epoch),
        })
        return loss
