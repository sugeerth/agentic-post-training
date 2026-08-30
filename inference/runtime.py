"""The forward pass an agent's rollouts actually run on.

`transformer.generate` recomputes the whole sequence for every token it
emits. For a 170-token GUI decision that is 170 full forward passes, each
one quadratic in the prefix it re-derives, and each one allocating an
autograd graph that nothing will ever call `backward()` on. Training and
sampling want opposite things from the same weights, and this module is the
sampling half:

**Keys and values are computed once per position.** A transformer's attention
at position *n* reads the keys and values of every position before it, and
those do not change when a new token arrives. Caching them turns each new
token from a quadratic re-derivation into a linear pass over what is already
there. This is the single largest constant in agentic post-training, because
rollout generation — not the gradient step — is where the wall clock goes.

**No graph is built.** Inference needs numbers, not parents and backward
closures. Running on plain lists removes an allocation per operation per
token from the hottest loop in the system.

The bar for this module is not that it is faster. It is that it is faster
*and indistinguishable*: a policy sampled through here must produce the same
logits, to the bit, as the same weights run through `GPT.logits`. A sampler
that quietly disagrees with its trainer generates rollouts for a policy that
does not exist, and every downstream gradient is then computed against the
wrong thing — silently, with no test failing. `tests/test_runtime.py` asserts
exact equality rather than closeness for that reason.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from computer_use.transformer import GPT, ModelConfig

__all__ = ["KVCache", "LayerWeights", "PolicyWeights", "logits_at", "prefill", "step"]


# --------------------------------------------------------------------------- #
# Weights, lifted out of the graph
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LayerWeights:
    """One block's parameters as plain lists."""

    ln1_g: list[float]
    ln1_b: list[float]
    wq: list[float]
    wk: list[float]
    wv: list[float]
    wo: list[float]
    ln2_g: list[float]
    ln2_b: list[float]
    w1: list[float]
    b1: list[float]
    w2: list[float]
    b2: list[float]


@dataclass(frozen=True)
class PolicyWeights:
    """A snapshot of a policy, detached from autograd.

    Frozen on purpose. A sampler that shares mutable state with a trainer is
    the classic source of a mid-rollout weight change: half a trajectory from
    policy *n*, half from *n+1*, stamped as one sample of neither. Taking a
    copy makes that impossible rather than merely unlikely — see
    `inference.version` for what it costs and why it is worth it.
    """

    config: ModelConfig
    tok: list[float]
    pos: list[float]
    blocks: tuple[LayerWeights, ...]
    ln_g: list[float]
    ln_b: list[float]
    #: Which trainer step produced these weights. Stamped onto every sample.
    version: int = 0

    @classmethod
    def snapshot(cls, model: GPT, *, version: int = 0) -> PolicyWeights:
        """Copy a live model's parameters out of the autograd graph."""
        return cls(
            config=model.config,
            tok=list(model.tok.data),
            pos=list(model.pos.data),
            blocks=tuple(
                LayerWeights(
                    ln1_g=list(b.ln1_g.data), ln1_b=list(b.ln1_b.data),
                    wq=list(b.wq.data), wk=list(b.wk.data),
                    wv=list(b.wv.data), wo=list(b.wo.data),
                    ln2_g=list(b.ln2_g.data), ln2_b=list(b.ln2_b.data),
                    w1=list(b.w1.data), b1=list(b.b1.data),
                    w2=list(b.w2.data), b2=list(b.b2.data),
                )
                for b in model.blocks
            ),
            ln_g=list(model.ln_g.data),
            ln_b=list(model.ln_b.data),
            version=version,
        )


# --------------------------------------------------------------------------- #
# The cache
# --------------------------------------------------------------------------- #


@dataclass
class KVCache:
    """Per-layer keys and values, one row per position processed so far.

    Stored as a flat list per layer with `d_model` floats per position, which
    is the layout attention reads them in. Length is tracked separately from
    the lists so a cache can be truncated back to a shared prefix without
    copying — see `inference.cache`, where that is the whole point.
    """

    n_layers: int
    d_model: int
    keys: list[list[float]] = field(default_factory=list)
    values: list[list[float]] = field(default_factory=list)
    length: int = 0

    def __post_init__(self) -> None:
        if not self.keys:
            self.keys = [[] for _ in range(self.n_layers)]
            self.values = [[] for _ in range(self.n_layers)]

    def fork(self, upto: int | None = None) -> KVCache:
        """A copy holding the first `upto` positions.

        Copies rather than shares because a forked branch appends to it, and
        two branches appending to one list would interleave two sequences into
        a cache that describes neither.
        """
        n = self.length if upto is None else min(upto, self.length)
        width = n * self.d_model
        return KVCache(
            n_layers=self.n_layers,
            d_model=self.d_model,
            keys=[k[:width] for k in self.keys],
            values=[v[:width] for v in self.values],
            length=n,
        )

    def truncate(self, n: int) -> None:
        """Drop everything after position `n`, in place."""
        n = max(0, min(n, self.length))
        width = n * self.d_model
        for layer in range(self.n_layers):
            del self.keys[layer][width:]
            del self.values[layer][width:]
        self.length = n


# --------------------------------------------------------------------------- #
# Arithmetic
# --------------------------------------------------------------------------- #


def _row_mm(row: list[float], w: list[float], k: int, n: int) -> list[float]:
    """`row @ w` for a single row.

    Summed in the same order `nn._mm` sums it — index `i` ascending within an
    output column — so the result is bit-identical rather than merely close.
    Accumulating column-blocks instead would be friendlier to the cache and
    would change the last few bits, which is exactly the kind of difference
    that makes a sampler disagree with its trainer for no visible reason.
    """
    return [sum(row[i] * w[i * n + j] for i in range(k)) for j in range(n)]


def _layer_norm(row: list[float], g: list[float], b: list[float]) -> list[float]:
    n = len(row)
    mean = sum(row) / n
    centred = [v - mean for v in row]
    var = sum(v * v for v in centred) / n
    inv = 1.0 / math.sqrt(var + 1e-5)
    return [v * inv * gi + bi for v, gi, bi in zip(centred, g, b, strict=True)]


def _softmax(scores: list[float]) -> list[float]:
    top = max(scores)
    exps = [math.exp(v - top) for v in scores]
    total = sum(exps)
    return [e / total for e in exps]


def _block(
    x: list[float], w: LayerWeights, cache: KVCache, layer: int, config: ModelConfig
) -> list[float]:
    """One position through one block, appending its key and value."""
    d, heads = config.d_model, config.n_heads
    head_dim = config.head_dim
    inv = 1.0 / math.sqrt(head_dim)

    normed = _layer_norm(x, w.ln1_g, w.ln1_b)
    q = _row_mm(normed, w.wq, d, d)
    k = _row_mm(normed, w.wk, d, d)
    v = _row_mm(normed, w.wv, d, d)

    cache.keys[layer].extend(k)
    cache.values[layer].extend(v)
    seen = len(cache.keys[layer]) // d

    attended = [0.0] * d
    keys, values = cache.keys[layer], cache.values[layer]
    for h in range(heads):
        start = h * head_dim
        qh = q[start : start + head_dim]
        # Causal by construction: the cache holds only positions already
        # processed, so there is nothing after the diagonal to mask.
        scores = [
            sum(qh[c] * keys[p * d + start + c] for c in range(head_dim)) * inv
            for p in range(seen)
        ]
        weights = _softmax(scores)
        for p, weight in enumerate(weights):
            if weight == 0.0:
                continue
            base = p * d + start
            for c in range(head_dim):
                attended[start + c] += weight * values[base + c]

    projected = _row_mm(attended, w.wo, d, d)
    x = [a + b for a, b in zip(x, projected, strict=True)]

    normed = _layer_norm(x, w.ln2_g, w.ln2_b)
    hidden = _row_mm(normed, w.w1, d, config.d_ff)
    hidden = [max(0.0, h + b) for h, b in zip(hidden, w.b1, strict=True)]
    out = _row_mm(hidden, w.w2, config.d_ff, d)
    return [a + o + b for a, o, b in zip(x, out, w.b2, strict=True)]


# --------------------------------------------------------------------------- #
# The public forward
# --------------------------------------------------------------------------- #


def step(weights: PolicyWeights, cache: KVCache, token: int) -> list[float]:
    """Push one token through, returning its final hidden state.

    The cache's current length *is* the position, which is what makes a
    prefix reusable: a cache truncated to `n` and then extended describes the
    same sequence as one built from scratch, so two requests sharing a prefix
    can share the work that produced it.
    """
    config = weights.config
    d = config.d_model
    position = cache.length
    if position >= config.max_len:
        raise ValueError(
            f"position {position} exceeds max_len={config.max_len}; "
            "the request should have been rejected at admission, not here"
        )
    x = [
        weights.tok[token * d + i] + weights.pos[position * d + i] for i in range(d)
    ]
    for layer, block in enumerate(weights.blocks):
        x = _block(x, block, cache, layer, config)
    cache.length += 1
    return _layer_norm(x, weights.ln_g, weights.ln_b)


def prefill(weights: PolicyWeights, cache: KVCache, ids: list[int]) -> list[float]:
    """Push a whole prompt through. Returns the last position's hidden state."""
    if not ids:
        raise ValueError("nothing to prefill")
    state: list[float] = []
    for token in ids:
        state = step(weights, cache, token)
    return state


def logits_at(weights: PolicyWeights, state: list[float]) -> list[float]:
    """Vocabulary scores for one hidden state, through the tied embedding."""
    d = weights.config.d_model
    tok = weights.tok
    return [
        sum(state[i] * tok[row * d + i] for i in range(d))
        for row in range(weights.config.vocab_size)
    ]
