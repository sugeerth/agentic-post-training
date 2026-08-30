"""The sampler a trainer calls, and the seam a real engine plugs into.

One protocol, two implementations in principle and one in fact. `Engine` is
the surface: submit prompts, get completions with per-token log-probs and a
policy version stamp. `LocalEngine` implements it against this package's own
transformer, in pure Python, so the loop is runnable end to end in CI with no
GPU and no service. A vLLM or SGLang engine implements the same protocol and
is a drop-in — the seam is drawn where it is precisely so that the scheduling,
caching, versioning and accounting above it are not re-litigated per backend.

Two design points worth arguing with.

**Log-probs come back from generation, not from a second pass.** A trainer
needs the log-probability of each sampled token under the sampling policy.
Re-deriving it afterwards means a second forward over every rollout — doubling
the most expensive part of the loop — and it silently drifts if the weights
moved in between. The sampler already computes the distribution it drew from,
so it returns it.

**Admission is checked before work starts, not during.** A request whose
prompt already exceeds the context window cannot be completed, and finding
that out 150 tokens in wastes all 150. `submit` rejects it up front with a
reason the caller can act on.
"""

from __future__ import annotations

import math
import random
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from inference.cache import RadixCache
from inference.runtime import KVCache, PolicyWeights, logits_at, step

__all__ = [
    "Completion",
    "Engine",
    "EngineStats",
    "LocalEngine",
    "Request",
    "RequestRejected",
]


class RequestRejected(ValueError):
    """A request that cannot be served, refused before any work is spent."""


@dataclass(frozen=True)
class Request:
    """One thing to sample."""

    prompt: list[int]
    max_new: int = 32
    stop: tuple[int, ...] = ()
    temperature: float = 0.0
    #: Ties a completion back to whatever asked for it — a task, a group index.
    tag: str = ""

    def __post_init__(self) -> None:
        if not self.prompt:
            raise RequestRejected("empty prompt")
        if self.max_new < 1:
            raise RequestRejected(f"max_new must be >= 1, got {self.max_new}")
        if self.temperature < 0.0:
            raise RequestRejected(f"temperature must be >= 0, got {self.temperature}")


@dataclass(frozen=True)
class Completion:
    """What came back, with everything a trainer needs to learn from it."""

    tokens: list[int]
    #: log P(token | prefix) under the sampling policy, one per emitted token.
    logprobs: list[float]
    #: Which trainer step's weights produced this. See `inference.version`.
    version: int
    #: Prompt tokens served from cache rather than recomputed.
    prefix_hit: int
    stopped: bool
    tag: str = ""

    @property
    def logprob_sum(self) -> float:
        return sum(self.logprobs)

    def __len__(self) -> int:
        return len(self.tokens)


@dataclass
class EngineStats:
    """Throughput accounting, separated into the two things that differ."""

    prompt_tokens: int = 0
    generated_tokens: int = 0
    requests: int = 0
    rejected: int = 0
    seconds: float = 0.0

    @property
    def tokens_per_second(self) -> float:
        return self.generated_tokens / self.seconds if self.seconds else 0.0

    def to_dict(self) -> dict[str, float | int]:
        return {
            "requests": self.requests,
            "rejected": self.rejected,
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": self.generated_tokens,
            "seconds": round(self.seconds, 3),
            "tokens_per_second": round(self.tokens_per_second, 1),
        }


class Engine(Protocol):
    """What a trainer needs from a sampler, and nothing more."""

    @property
    def version(self) -> int:
        """The policy version this engine is currently serving."""
        ...

    def generate(self, requests: Sequence[Request]) -> list[Completion]:
        """Sample every request against the currently loaded weights."""
        ...

    def load(self, weights: PolicyWeights) -> None:
        """Swap in new weights. Invalidates anything cached under the old ones."""
        ...


class LocalEngine:
    """`Engine` over this package's transformer, with prefix reuse.

    Single-threaded by design. Concurrency here would be a lie: the work is
    pure Python arithmetic holding the GIL, so threads would add scheduling
    overhead and no throughput. Requests are instead *ordered* to make the
    prefix cache pay — see `generate`.
    """

    def __init__(
        self,
        weights: PolicyWeights,
        *,
        cache_capacity: int = 64,
        seed: int = 0,
    ) -> None:
        self._weights = weights
        self.cache = RadixCache(weights, capacity=cache_capacity)
        self.stats = EngineStats()
        self._rng = random.Random(seed)

    @property
    def version(self) -> int:
        return self._weights.version

    @property
    def weights(self) -> PolicyWeights:
        return self._weights

    def load(self, weights: PolicyWeights) -> None:
        """Swap weights and drop the cache built under the previous ones."""
        self._weights = weights
        self.cache.rebind(weights)

    # -- sampling --------------------------------------------------------- #

    def _admit(self, request: Request) -> None:
        limit = self._weights.config.max_len
        needed = len(request.prompt) + request.max_new
        if len(request.prompt) >= limit:
            raise RequestRejected(
                f"prompt of {len(request.prompt)} tokens meets the context "
                f"limit of {limit}; there is no room to generate into"
            )
        if needed > limit:
            raise RequestRejected(
                f"prompt ({len(request.prompt)}) plus max_new ({request.max_new}) "
                f"is {needed}, over the {limit}-token context. Shorten either, "
                "rather than discovering it mid-generation"
            )

    def _pick(self, row: list[float], temperature: float) -> tuple[int, float]:
        """Choose a token and report the log-probability it was chosen with.

        The log-prob is always taken from the *sampling* distribution — the one
        temperature actually shaped — because that is the distribution the
        trainer's importance ratio is against. Reporting the greedy
        distribution's value here would be a subtle, permanent bias.
        """
        if temperature <= 0.0:
            token = max(range(len(row)), key=row.__getitem__)
            top = max(row)
            total = sum(math.exp(v - top) for v in row)
            return token, (row[token] - top) - math.log(total)

        scaled = [v / temperature for v in row]
        top = max(scaled)
        exps = [math.exp(v - top) for v in scaled]
        total = sum(exps)
        token = self._rng.choices(range(len(row)), weights=exps, k=1)[0]
        return token, math.log(exps[token] / total)

    def _one(self, request: Request) -> Completion:
        cache: KVCache
        cache, hit = self.cache.acquire(list(request.prompt))
        state = self._replay_tail(cache, request.prompt)

        tokens: list[int] = []
        logprobs: list[float] = []
        stopped = False
        stop = set(request.stop)
        limit = self._weights.config.max_len

        for _ in range(request.max_new):
            row = logits_at(self._weights, state)
            token, logprob = self._pick(row, request.temperature)
            tokens.append(token)
            logprobs.append(logprob)
            if token in stop:
                stopped = True
                break
            if cache.length >= limit:
                break
            state = step(self._weights, cache, token)

        self.stats.requests += 1
        self.stats.prompt_tokens += len(request.prompt)
        self.stats.generated_tokens += len(tokens)
        return Completion(
            tokens=tokens, logprobs=logprobs, version=self._weights.version,
            prefix_hit=hit, stopped=stopped, tag=request.tag,
        )

    def _replay_tail(self, cache: KVCache, prompt: Sequence[int]) -> list[float]:
        """The hidden state at the prompt's last position.

        `RadixCache.acquire` returns a cache holding the whole prompt, but the
        final hidden state is not part of what a KV cache stores — only keys
        and values are. Rather than keep hidden states alive for every cached
        prefix, the last position alone is recomputed, which is one position's
        work regardless of how long the prompt was.
        """
        cache.truncate(len(prompt) - 1)
        return step(self._weights, cache, prompt[-1])

    def generate(self, requests: Sequence[Request]) -> list[Completion]:
        """Sample every request, ordered so the prefix cache pays.

        Sorting by prompt puts requests sharing a prefix next to each other, so
        the shared work is computed once and hit by the rest of the run before
        anything can evict it. Results come back in submission order — a
        scheduler that reorders its output would make the caller responsible
        for the scheduler's internals.
        """
        started = time.monotonic()
        for request in requests:
            self._admit(request)

        order = sorted(range(len(requests)), key=lambda i: requests[i].prompt)
        out: list[Completion | None] = [None] * len(requests)
        for index in order:
            out[index] = self._one(requests[index])
        self.stats.seconds += time.monotonic() - started
        return [c for c in out if c is not None]
