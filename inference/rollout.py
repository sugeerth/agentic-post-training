"""Sampling a GRPO group, and handing the trainer something it can learn from.

`RolloutBatch` is the shape every RL technique in this repo consumes, and
until now nothing produced one — a batch arrived fully formed and the question
of where it came from was out of scope. That is the gap this closes: an
environment, a policy, and a verifier go in; a `RolloutBatch` with real
rewards and real log-probs comes out.

The group is the unit, because that is what GRPO's advantage is defined over.
It normalizes a sample's reward against the *other samples of the same prompt*,
which is what lets it drop the value network — and it means a group of size 1
carries no signal at all, since a sample compared only to itself has zero
advantage by construction. `sample_group` refuses that rather than returning a
batch whose gradient is silently zero.

Two properties are enforced here rather than left to the caller:

**Every sample in a group comes from one policy version.** Sampling the group
is a single call against a single weight snapshot, so a mid-group weight swap
cannot happen. The version is carried onto the batch metadata for the trainer
to check against its own step.

**Temperature is required to be non-zero.** A group sampled greedily is *k*
identical trajectories, so every reward is the group mean, every advantage is
zero, and the step is a no-op that looks like a step. This is an easy mistake
to make when the surrounding evaluation code deliberately uses `temperature=0`
for reproducibility, so it is refused with a reason.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from core.types import RolloutBatch
from inference.engine import Completion, Engine, Request

__all__ = ["Group", "GroupSpec", "sample_group", "to_rollout_batch"]

#: Scores one completion. Returns a scalar reward — a verifier's verdict,
#: a shaped score, anything the caller can defend.
Scorer = Callable[[Sequence[int], str], float]


@dataclass(frozen=True)
class GroupSpec:
    """One prompt, to be sampled `size` times."""

    prompt: list[int]
    size: int = 8
    max_new: int = 32
    stop: tuple[int, ...] = ()
    temperature: float = 0.8
    tag: str = ""

    def __post_init__(self) -> None:
        if self.size < 2:
            raise ValueError(
                f"group_size={self.size} carries no learning signal: GRPO's "
                "advantage is a sample's reward relative to its group, so a "
                "group of one has zero advantage for every sample and the "
                "update is a no-op. Use 2 or more."
            )
        if self.temperature <= 0.0:
            raise ValueError(
                "temperature=0 samples the same trajectory `size` times, so "
                "every reward equals the group mean and every advantage is "
                "zero. Greedy decoding is right for scoring a policy and wrong "
                "for sampling a group to learn from."
            )


@dataclass(frozen=True)
class Group:
    """One prompt's samples, with what they scored."""

    spec: GroupSpec
    completions: tuple[Completion, ...]
    rewards: tuple[float, ...]
    version: int

    @property
    def advantages(self) -> tuple[float, ...]:
        """Reward, centred on the group and scaled by its spread.

        This is GRPO's estimator, computed here so the batch can be inspected
        before it reaches a trainer. A group whose samples all scored the same
        has zero spread and therefore zero advantage — correctly, since it
        contains no evidence that any sample was better than any other.
        """
        if len(self.rewards) < 2:
            return tuple(0.0 for _ in self.rewards)
        mean = statistics.fmean(self.rewards)
        spread = statistics.pstdev(self.rewards)
        if spread == 0.0:
            return tuple(0.0 for _ in self.rewards)
        return tuple((r - mean) / spread for r in self.rewards)

    @property
    def degenerate(self) -> bool:
        """True when the group cannot teach anything: every sample scored alike."""
        return len(set(self.rewards)) <= 1

    @property
    def prefix_saved(self) -> int:
        """Prompt tokens the cache served rather than recomputed, across the group."""
        return sum(c.prefix_hit for c in self.completions)


def sample_group(engine: Engine, spec: GroupSpec, score: Scorer) -> Group:
    """Sample one prompt `size` times and score each continuation.

    All `size` requests go to the engine in one call, which is what lets the
    prefix cache compute the shared prompt once — and what guarantees a single
    policy version across the group.
    """
    requests = [
        Request(
            prompt=list(spec.prompt), max_new=spec.max_new, stop=spec.stop,
            temperature=spec.temperature, tag=f"{spec.tag}#{i}",
        )
        for i in range(spec.size)
    ]
    completions = engine.generate(requests)
    versions = {c.version for c in completions}
    if len(versions) != 1:
        raise RuntimeError(
            f"one group spans policy versions {sorted(versions)}. The weights "
            "changed mid-group, so these samples are not comparable to each "
            "other and the advantage between them is meaningless."
        )
    rewards = tuple(score(c.tokens, c.tag) for c in completions)
    return Group(
        spec=spec,
        completions=tuple(completions),
        rewards=rewards,
        version=next(iter(versions)),
    )


@dataclass
class BatchReport:
    """What a sampled batch cost and how much of it can teach anything."""

    groups: int = 0
    samples: int = 0
    degenerate_groups: int = 0
    prefix_tokens_saved: int = 0
    versions: tuple[int, ...] = field(default_factory=tuple)

    @property
    def usable_groups(self) -> int:
        return self.groups - self.degenerate_groups

    def to_dict(self) -> dict[str, object]:
        return {
            "groups": self.groups,
            "samples": self.samples,
            "degenerate_groups": self.degenerate_groups,
            "usable_groups": self.usable_groups,
            "prefix_tokens_saved": self.prefix_tokens_saved,
            "versions": sorted(set(self.versions)),
        }


def to_rollout_batch(
    groups: Sequence[Group], *, decode: Callable[[Sequence[int]], str] | None = None
) -> tuple[RolloutBatch, BatchReport]:
    """Groups into the shape `techniques.grpo` consumes, plus what it cost.

    `log_probs` is filled from the sampler rather than left `None`. That is the
    difference between GRPO's real path and its simulation fallback: with
    log-probs absent the technique drops to a deterministic decay curve, which
    is useful for a demo and is not training. With them present it computes an
    actual ratio against the reference policy.

    Degenerate groups are kept rather than dropped. They contribute zero
    gradient either way, and silently removing them would make the batch size
    a function of the policy's current spread — which is a moving denominator
    in every metric computed from it.

    **Log-probs are padded, and the mask is not optional.** Samples in a group
    stop at different lengths — one hits its stop token at seven tokens while
    the rest run to twelve — and a ragged nested list cannot become the
    `(B, G, T)` tensor GRPO reads. Unpadded, this raises inside the trainer
    on a GPU machine after the expensive part is already paid for. Padding is
    `0.0`, which is a *log*-probability of 1.0, so summing without applying
    `response_mask` silently adds free probability mass to every short sample
    and biases the ratio toward whichever ones stopped early. The mask ships
    in metadata alongside the lengths for exactly that reason.
    """
    if not groups:
        raise ValueError("no groups: an empty batch has nothing to learn from")

    render = decode or (lambda ids: " ".join(str(i) for i in ids))

    width = max(
        (len(c.logprobs) for g in groups for c in g.completions), default=0
    )
    padded: list[list[list[float]]] = []
    mask: list[list[list[float]]] = []
    lengths: list[list[int]] = []
    for group in groups:
        padded.append([
            [*c.logprobs, *([0.0] * (width - len(c.logprobs)))]
            for c in group.completions
        ])
        mask.append([
            [1.0] * len(c.logprobs) + [0.0] * (width - len(c.logprobs))
            for c in group.completions
        ])
        lengths.append([len(c.logprobs) for c in group.completions])

    report = BatchReport(
        groups=len(groups),
        samples=sum(len(g.completions) for g in groups),
        degenerate_groups=sum(1 for g in groups if g.degenerate),
        prefix_tokens_saved=sum(g.prefix_saved for g in groups),
        versions=tuple(g.version for g in groups),
    )
    batch = RolloutBatch(
        prompts=[render(g.spec.prompt) for g in groups],
        responses=[[render(c.tokens) for c in g.completions] for g in groups],
        rewards=[list(g.rewards) for g in groups],
        log_probs=padded,
        metadata={
            "source": "inference.rollout",
            "policy_versions": sorted(set(report.versions)),
            "advantages": [list(g.advantages) for g in groups],
            # Apply this before reducing over T. See the docstring: padding is
            # 0.0 in log space, which is not a neutral element for a sum.
            "response_mask": mask,
            "response_lengths": lengths,
            "padded_to": width,
            **report.to_dict(),
        },
    )
    return batch, report
