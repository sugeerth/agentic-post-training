"""Inference infrastructure for agentic post-training.

Post-training an agent spends most of its wall clock generating rollouts, not
computing gradients. A GRPO step needs *k* trajectories per prompt, each one a
multi-turn episode against an environment, and every one of them is a sampling
problem before it is a learning problem. This package is that half of the
loop, built to the three things that actually decide whether it works:

**Speed.** Keys and values are computed once per position instead of once per
emitted token, and the sampling path carries no autograd graph. Measured at
13.4x on this package's own transformer, with logits bit-identical to the
training path — because a sampler that disagrees with its trainer generates
rollouts for a policy that does not exist.

**Reuse.** A GRPO group is one prompt sampled *k* times, and a multi-turn
episode grows by appending to its own context. Both are prefix sharing waiting
to happen, and a radix trie over token ids finds it.

**Attribution.** Every completion is stamped with the policy version that
produced it, and a batch that mixes versions is a stated choice rather than an
accident. On-policy objectives quietly stop holding when a sampler lags its
trainer, and nothing about the loss curve reveals it.

The seam is `Engine`. `LocalEngine` implements it in pure Python against this
repo's own model, so the loop runs end to end in CI with no GPU and no service;
a vLLM or SGLang engine implements the same protocol without touching anything
above it.
"""

from inference.cache import CacheStats, RadixCache
from inference.engine import (
    Completion,
    Engine,
    EngineStats,
    LocalEngine,
    Request,
    RequestRejected,
)
from inference.rollout import (
    BatchReport,
    Group,
    GroupSpec,
    sample_group,
    to_rollout_batch,
)
from inference.runtime import KVCache, LayerWeights, PolicyWeights
from inference.version import StalenessError, StalenessReport, check_staleness

__all__ = [
    "BatchReport",
    "CacheStats",
    "Completion",
    "Engine",
    "EngineStats",
    "Group",
    "GroupSpec",
    "KVCache",
    "LayerWeights",
    "LocalEngine",
    "PolicyWeights",
    "RadixCache",
    "Request",
    "RequestRejected",
    "StalenessError",
    "StalenessReport",
    "check_staleness",
    "sample_group",
    "to_rollout_batch",
]
