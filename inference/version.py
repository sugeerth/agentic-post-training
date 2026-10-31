"""Which policy produced this rollout, and does the objective still hold.

GRPO and PPO are on-policy: the advantage of a sampled action is only
meaningful relative to the policy that sampled it. The moment a trainer takes
a step, every rollout still in flight was produced by a policy that no longer
exists, and the gradient computed from it is answering a question about the
wrong distribution.

Nothing crashes when this happens. The loss still falls. The reward curve
still rises. The run is simply optimizing something other than what the
objective says, and the only symptom is that it works less well than it
should — which is indistinguishable from a bad hyperparameter.

The usual defences are to pause sampling during the update, or to accept the
drift and correct for it with an importance ratio. Both are legitimate. What
is not legitimate is not knowing which one you are doing, so this module
makes the version a property of the sample rather than of the wall clock:

* every `PolicyWeights` snapshot carries the trainer step that produced it;
* every completion is stamped with the version that generated it;
* a batch assembled from mixed versions is a stated choice, made by passing
  `max_staleness`, and never a silent default.

`StalenessError` is raised rather than warned because a warning in a training
loop is a line of log nobody reads at 3am, and the failure it describes is
one that produces a plausible-looking curve either way.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

__all__ = [
    "RatioReport",
    "StalenessError",
    "StalenessReport",
    "check_staleness",
    "importance_ratios",
]


class StalenessError(RuntimeError):
    """A batch mixes policy versions further apart than the caller allowed."""


@dataclass(frozen=True)
class StalenessReport:
    """How far the samples in a batch lag the policy about to be updated."""

    trainer_version: int
    versions: tuple[int, ...]
    #: Per-sample lag, in trainer steps.
    lags: tuple[int, ...]

    @property
    def max_lag(self) -> int:
        return max(self.lags) if self.lags else 0

    @property
    def mean_lag(self) -> float:
        return sum(self.lags) / len(self.lags) if self.lags else 0.0

    @property
    def on_policy(self) -> bool:
        """True when every sample came from the weights about to be updated."""
        return self.max_lag == 0

    def to_dict(self) -> dict[str, float | int | bool]:
        return {
            "trainer_version": self.trainer_version,
            "max_lag": self.max_lag,
            "mean_lag": round(self.mean_lag, 3),
            "on_policy": self.on_policy,
            "distinct_versions": len(set(self.versions)),
            "samples": len(self.versions),
        }


def check_staleness(
    versions: Sequence[int], trainer_version: int, *, max_staleness: int = 0
) -> StalenessReport:
    """Measure the lag in a batch, and refuse it if it exceeds what was allowed.

    `max_staleness=0` — the default — means strictly on-policy: every sample
    must come from the current weights. Raising it is how a caller says "I am
    running asynchronously and I accept the drift", which is a real and common
    choice. It just has to be a choice.
    """
    if not versions:
        raise ValueError("no samples to check: an empty batch has no policy")
    if max_staleness < 0:
        raise ValueError(f"max_staleness must be >= 0, got {max_staleness}")

    lags = tuple(trainer_version - v for v in versions)
    if any(lag < 0 for lag in lags):
        ahead = max(-lag for lag in lags if lag < 0)
        raise StalenessError(
            f"a sample claims policy version {trainer_version + ahead}, ahead of "
            f"the trainer at {trainer_version}. A sampler running ahead of its "
            "trainer means the version stamps are not coming from one clock."
        )

    report = StalenessReport(trainer_version, tuple(versions), lags)
    if report.max_lag > max_staleness:
        raise StalenessError(
            f"batch lags the trainer by {report.max_lag} step(s), over the "
            f"{max_staleness} allowed. These rollouts came from a policy that "
            "no longer exists, so the on-policy objective does not hold for "
            "them. Either re-sample against the current weights, or pass "
            f"max_staleness={report.max_lag} to say the drift is intended and "
            "correct for it."
        )
    return report


# --------------------------------------------------------------------------- #
# The importance ratio, and why it has two halves
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RatioReport:
    """Importance-ratio diagnostics for one batch of sampled tokens.

    The asynchronous-RL literature decomposes the total ratio a PPO-style
    objective needs into two semantically different factors, and warns that
    entangling them makes clipping thresholds interact in ways nobody
    intended:

    1. **Training–inference discrepancy** — the sampler and the trainer
       disagree about the probability of the *same token* under the *same
       weights*, because they are different implementations. This is not
       benign numerical noise; it is documented as a first-order cause of
       collapse in real runs, which is why bitwise-consistent sampling is
       something frameworks now build on purpose.
    2. **Policy staleness** — the weights genuinely moved between sampling
       and updating. That is the drift `check_staleness` measures.

    This package's sampler is bit-identical to its trainer by construction and
    by test, so factor 1 is exactly 1.0 and needs no repair. That is a real
    property of running both halves on the same arithmetic rather than a claim
    about being careful, and `discrepancy_free` reports it from the numbers
    rather than asserting it.
    """

    #: exp(current - behaviour), per token, over unmasked positions only.
    ratios: tuple[float, ...]
    clip: float

    @property
    def mean(self) -> float:
        return sum(self.ratios) / len(self.ratios) if self.ratios else 1.0

    @property
    def clipped_fraction(self) -> float:
        """Share of tokens outside the trust region.

        The number to watch. A ratio distribution that drifts wide means the
        gradient is increasingly made of clipped, and therefore truncated,
        contributions — the run is quietly training on less than it thinks.
        """
        if not self.ratios:
            return 0.0
        outside = sum(
            1 for r in self.ratios if r > self.clip or r < 1.0 / self.clip
        )
        return outside / len(self.ratios)

    @property
    def discrepancy_free(self) -> bool:
        """True when every ratio is exactly 1 — sampler and trainer agree bitwise."""
        return all(r == 1.0 for r in self.ratios)

    def to_dict(self) -> dict[str, float | bool | int]:
        return {
            "tokens": len(self.ratios),
            "mean_ratio": round(self.mean, 6),
            "max_ratio": round(max(self.ratios), 6) if self.ratios else 1.0,
            "min_ratio": round(min(self.ratios), 6) if self.ratios else 1.0,
            "clipped_fraction": round(self.clipped_fraction, 4),
            "discrepancy_free": self.discrepancy_free,
        }


def importance_ratios(
    behaviour: Sequence[Sequence[float]],
    current: Sequence[Sequence[float]],
    *,
    mask: Sequence[Sequence[float]] | None = None,
    clip: float = 5.0,
) -> RatioReport:
    """Per-token `exp(current - behaviour)` over the positions that are real.

    `behaviour` is what the sampler reported when it drew each token — which
    is why the engine returns log-probs instead of leaving them to be
    recomputed. Recomputing them is the failure this guards against: a second
    forward pass measures the *current* policy, so the ratio it produces is
    identically 1 and the correction it feeds silently does nothing.

    `mask` must be supplied whenever the log-probs were padded. Padding is 0.0
    in log space, so an unmasked padded position contributes a ratio of
    exactly 1 and drags the diagnostics toward looking healthier than the run
    is — most for the shortest samples, which is a bias with a direction.
    """
    if clip <= 1.0:
        raise ValueError(f"clip must be > 1, got {clip}")
    if len(behaviour) != len(current):
        raise ValueError(
            f"{len(behaviour)} behaviour rows against {len(current)} current "
            "rows: these are not the same batch"
        )

    kept: list[float] = []
    for index, (old_row, new_row) in enumerate(zip(behaviour, current, strict=True)):
        if len(old_row) != len(new_row):
            raise ValueError(
                f"row {index} has {len(old_row)} behaviour log-probs against "
                f"{len(new_row)} current ones. Ragged rows mean the batch was "
                "assembled without padding; see `rollout.to_rollout_batch`"
            )
        weights = mask[index] if mask is not None else [1.0] * len(old_row)
        for old, new, keep in zip(old_row, new_row, weights, strict=True):
            if keep:
                kept.append(math.exp(new - old))
    return RatioReport(ratios=tuple(kept), clip=clip)
