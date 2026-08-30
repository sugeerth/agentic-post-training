"""Sharing the work behind a shared prefix.

Two things dominate an agentic post-training corpus, and both are prefix
reuse waiting to happen:

**A GRPO group is one prompt sampled *k* times.** Every sample in the group
re-reads the identical prompt before it diverges. Computing that prompt once
instead of *k* times is not an optimization at the margin — for a group of 8
over a 150-token prompt with 20-token continuations, the prompt is 88% of the
tokens, and 7 of the 8 copies are pure waste.

**A multi-turn episode grows by appending.** Step *n+1*'s context is step
*n*'s instruction, plus a new observation. Whatever the two share, the second
one need not recompute.

A radix trie over token ids finds the longest already-computed prefix of an
incoming request, and the cache forks its keys and values at that depth. What
remains is only the tokens nobody has seen.

**What this is not.** Forking copies the shared prefix rather than sharing it
by reference with a refcount, which is what a production engine does — that is
PagedAttention, and it is how vLLM makes the same idea hold at a scale where
copying would dominate. Copying is honest at this scale and wrong at that one;
the trie and the accounting here are the parts that transfer, and
`hit_rate` / `tokens_saved` are reported so the claim is a measurement rather
than an assertion.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from inference.runtime import KVCache, PolicyWeights, prefill

__all__ = ["CacheStats", "RadixCache"]


@dataclass
class _Node:
    """One trie node: the tokens along this edge, and the KV that ends it."""

    token: int
    depth: int
    children: dict[int, _Node] = field(default_factory=dict)
    cache: KVCache | None = None
    #: Monotonic counter, for least-recently-used eviction.
    touched: int = 0


@dataclass
class CacheStats:
    """What the cache actually bought, in tokens rather than in adjectives."""

    requests: int = 0
    hits: int = 0
    tokens_requested: int = 0
    tokens_computed: int = 0
    evictions: int = 0

    @property
    def tokens_saved(self) -> int:
        return self.tokens_requested - self.tokens_computed

    @property
    def hit_rate(self) -> float:
        """Share of requested prefix tokens that were already computed."""
        if not self.tokens_requested:
            return 0.0
        return self.tokens_saved / self.tokens_requested

    @property
    def request_hit_rate(self) -> float:
        """Share of requests that found *any* usable prefix."""
        return self.hits / self.requests if self.requests else 0.0

    def to_dict(self) -> dict[str, float | int]:
        return {
            "requests": self.requests,
            "hits": self.hits,
            "tokens_requested": self.tokens_requested,
            "tokens_computed": self.tokens_computed,
            "tokens_saved": self.tokens_saved,
            "hit_rate": round(self.hit_rate, 4),
            "request_hit_rate": round(self.request_hit_rate, 4),
            "evictions": self.evictions,
        }


class RadixCache:
    """Longest-prefix KV reuse across requests, with LRU eviction.

    Bound to one `PolicyWeights`. A cache outliving a weight update would hand
    the sampler keys and values computed by a policy that no longer exists,
    which is the subtlest way to generate rollouts nobody can attribute — so
    `rebind` exists and it clears, rather than migrating, what it holds.
    """

    #: Positions per stored checkpoint. Reuse is block-aligned for the same
    #: reason a paged engine's is: storing a cache at *every* depth costs
    #: O(n) caches of O(n) each for one sequence, which is quadratic memory to
    #: save linear work. Sixteen is small enough that a divergent branch keeps
    #: nearly all of its shared prefix, and large enough that the trie holds a
    #: bounded number of snapshots.
    BLOCK = 16

    def __init__(self, weights: PolicyWeights, *, capacity: int = 64) -> None:
        self.weights = weights
        self.capacity = capacity
        self.stats = CacheStats()
        self._root = _Node(token=-1, depth=0)
        self._clock = 0
        self._stored = 0

    # -- lookup ----------------------------------------------------------- #

    def _descend(self, ids: list[int]) -> tuple[_Node | None, int]:
        """The deepest node holding a cache along `ids`, and its depth."""
        node = self._root
        best: _Node | None = None
        best_depth = 0
        for index, token in enumerate(ids):
            child = node.children.get(token)
            if child is None:
                break
            node = child
            if node.cache is not None:
                best, best_depth = node, index + 1
        return best, best_depth

    def acquire(self, ids: list[int]) -> tuple[KVCache, int]:
        """A cache primed for `ids`, plus how many tokens came for free.

        The returned cache is always private to the caller. Never hand out the
        stored one: the caller is about to append a continuation to it, and a
        continuation appended to the shared node would corrupt the prefix for
        everyone who arrives after.
        """
        self.stats.requests += 1
        self.stats.tokens_requested += len(ids)

        node, depth = self._descend(ids)
        if node is None or depth == 0:
            cache = KVCache(self.weights.config.n_layers, self.weights.config.d_model)
            prefill(self.weights, cache, ids)
            self.stats.tokens_computed += len(ids)
            self._insert(ids, cache)
            return cache.fork(), 0

        self._clock += 1
        node.touched = self._clock
        self.stats.hits += 1

        assert node.cache is not None
        cache = node.cache.fork()
        remainder = ids[depth:]
        if remainder:
            prefill(self.weights, cache, remainder)
            self.stats.tokens_computed += len(remainder)
            self._insert(ids, cache)
        return cache.fork(), depth

    # -- storage ---------------------------------------------------------- #

    def _insert(self, ids: list[int], cache: KVCache) -> None:
        """Store the sequence, checkpointing at block boundaries and the end.

        The terminal node is always stored, so an identical prompt — a GRPO
        group's *k* samples — reuses all of it. Interior checkpoints are what
        make a *divergent* branch pay: two episodes sharing an instruction, or
        a group after its samples split, land on the deepest block boundary
        inside what they share rather than starting from nothing.
        """
        self._clock += 1
        node = self._root
        for index, token in enumerate(ids):
            child = node.children.get(token)
            if child is None:
                child = _Node(token=token, depth=node.depth + 1)
                node.children[token] = child
            node = child
            depth = index + 1
            if depth % self.BLOCK and depth != len(ids):
                continue
            if node.cache is None:
                self._stored += 1
            node.cache = cache.fork(depth)
            node.touched = self._clock
        self._evict()

    def _evict(self) -> None:
        """Drop the least recently used entries until back under capacity.

        Shallow nodes are kept preferentially at equal recency: a short prefix
        is shared by more requests than a long one, so evicting it costs more
        than its length suggests.
        """
        while self._stored > self.capacity:
            holders: list[_Node] = []
            stack = [self._root]
            while stack:
                current = stack.pop()
                if current.cache is not None and current is not self._root:
                    holders.append(current)
                stack.extend(current.children.values())
            if not holders:
                return
            victim = min(holders, key=lambda n: (n.touched, -n.depth))
            victim.cache = None
            self._stored -= 1
            self.stats.evictions += 1

    # -- lifecycle -------------------------------------------------------- #

    def rebind(self, weights: PolicyWeights) -> None:
        """Point at new weights and drop everything computed under the old ones."""
        self.weights = weights
        self._root = _Node(token=-1, depth=0)
        self._stored = 0

    def clear(self) -> None:
        self._root = _Node(token=-1, depth=0)
        self._stored = 0
