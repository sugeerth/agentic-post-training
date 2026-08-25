"""A small reverse-mode autograd, written to have no dependencies.

`tokens` ends by handing a `TokenBatch` to "a transformer" — and there was no
transformer. That is the weakest kind of claim a pipeline can make: the data is
in the right shape for a model nobody ran. This module and `transformer` exist
to remove the hand-wave, by training an actual one on the actual tokens.

There is no numpy and no torch in this environment, so the arithmetic is Python
floats. That forces one design decision and rewards it: autograd has to be at
**matrix** granularity, not scalar. A micrograd-style graph over individual
numbers would allocate a node per multiply — hundreds of millions of them for a
single sequence, which is not slow so much as impossible. Here every op is a
whole matrix operation with a hand-written backward, so the graph for a forward
pass is a few hundred nodes regardless of how big the matrices are.

Hand-written backwards are exactly the kind of thing that is subtly wrong and
still trains — a transposed index or a missing sum over the batch produces
gradients that point *roughly* downhill, so the loss falls and the bug survives.
`tests/test_nn.py` finite-difference checks every op against its analytic
gradient rather than trusting the loss curve.

Tensors are flat lists plus a shape. Two dimensions only: everything a
decoder-only transformer needs is `(rows, cols)` if you fold the batch into
rows and treat heads as column slices, and refusing the general case keeps the
indexing arithmetic small enough to verify by eye.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Iterable, Sequence

__all__ = [
    "Adam",
    "Tensor",
    "add",
    "add_bias",
    "concat_cols",
    "cross_entropy",
    "embed",
    "layer_norm",
    "matmul",
    "parameter",
    "relu",
    "scale",
    "select_rows",
    "slice_cols",
    "softmax_rows",
    "transpose",
]


class Tensor:
    """A 2-D array of floats, and how it was computed.

    `data` is flat and row-major; `shape` is `(rows, cols)`. `grad` is `None`
    until a backward pass reaches this node, which keeps the common case — a
    constant, an input, an activation nobody differentiates — free.
    """

    __slots__ = ("_backward", "_parents", "data", "grad", "requires_grad", "shape")

    def __init__(
        self,
        data: list[float],
        shape: tuple[int, int],
        *,
        requires_grad: bool = False,
        parents: tuple[Tensor, ...] = (),
        backward: Callable[[], None] | None = None,
    ) -> None:
        rows, cols = shape
        if len(data) != rows * cols:
            raise ValueError(f"{len(data)} values do not fill {rows}x{cols}")
        self.data = data
        self.shape = shape
        self.grad: list[float] | None = None
        # Annotated because the initializer reads the same attribute on other
        # tensors; without it mypy cannot break the cycle to infer a type.
        self.requires_grad: bool = requires_grad or any(p.requires_grad for p in parents)
        self._parents = parents
        self._backward = backward

    # -- shape helpers ----------------------------------------------------- #

    @property
    def rows(self) -> int:
        return self.shape[0]

    @property
    def cols(self) -> int:
        return self.shape[1]

    def row(self, index: int) -> list[float]:
        return self.data[index * self.cols : (index + 1) * self.cols]

    def zero_grad(self) -> None:
        self.grad = None

    def _accumulate(self, values: Sequence[float]) -> None:
        """Add into `.grad`, creating it on first touch.

        Accumulation rather than assignment is what makes a tensor usable more
        than once in a graph — a weight matrix applied at every position, or a
        residual stream read by both branches of a block.
        """
        if self.grad is None:
            self.grad = list(values)
        else:
            grad = self.grad
            for i, v in enumerate(values):
                grad[i] += v

    def backward(self) -> None:
        """Seed this scalar with 1.0 and propagate in reverse topological order."""
        if self.shape != (1, 1):
            raise ValueError("backward() starts from a scalar")
        order: list[Tensor] = []
        seen: set[int] = set()

        # Iterative rather than recursive: sequences here are long enough that
        # the natural recursion overruns Python's stack on a real batch.
        stack: list[tuple[Tensor, bool]] = [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                order.append(node)
                continue
            if id(node) in seen:
                continue
            seen.add(id(node))
            stack.append((node, True))
            for parent in node._parents:
                if parent.requires_grad and id(parent) not in seen:
                    stack.append((parent, False))

        self.grad = [1.0]
        for node in reversed(order):
            if node._backward is not None and node.grad is not None:
                node._backward()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Tensor(shape={self.shape})"


def parameter(rows: int, cols: int, rng: random.Random, *, scale_: float | None = None) -> Tensor:
    """A trainable matrix, initialized the way attention layers like.

    Variance 1/fan_in keeps activations from growing or vanishing as depth
    increases; at this size the model trains without it too, but the loss curve
    is visibly worse for the first few hundred steps and that is wasted compute.
    """
    std = scale_ if scale_ is not None else (1.0 / math.sqrt(cols))
    data = [rng.gauss(0.0, std) for _ in range(rows * cols)]
    return Tensor(data, (rows, cols), requires_grad=True)


# --------------------------------------------------------------------------- #
# Operations
# --------------------------------------------------------------------------- #


def _transposed(data: Sequence[float], rows: int, cols: int) -> list[list[float]]:
    """`data` as a list of its columns, for dot-product-form multiplication."""
    return [list(data[j::cols]) for j in range(cols)]


def _flat_transpose(data: Sequence[float], rows: int, cols: int) -> list[float]:
    """The `(cols, rows)` transpose of a flat row-major matrix, still flat."""
    out: list[float] = []
    for j in range(cols):
        out.extend(data[j::cols])
    return out


def _mm(ad: Sequence[float], bd: Sequence[float], m: int, k: int, n: int) -> list[float]:
    """`A @ B` as flat lists, as one dot product per output element.

    The obvious alternative — accumulate `A[i,p] * B[p,:]` into an output row —
    looks cheaper because `[x + av*y for x, y in zip(..., strict=True)]` is a comprehension
    rather than a generator, and on isolated calls it measures ~15% faster per
    element. On the shapes this model actually uses it loses anyway, by 5-25%,
    because it re-slices `B` and rebuilds the accumulator on every one of its
    `m*k` iterations, and those allocations cost more than the arithmetic saves.
    That is worth stating plainly: this line was picked by measuring all three
    candidate loop orders on the five real shapes, not by reasoning about them,
    and the reasoning had it backwards.
    """
    columns = _transposed(bd, k, n)
    out = [0.0] * (m * n)
    for i in range(m):
        row = ad[i * k : (i + 1) * k]
        base = i * n
        for j in range(n):
            out[base + j] = sum(x * y for x, y in zip(row, columns[j], strict=True))
    return out


def matmul(a: Tensor, b: Tensor) -> Tensor:
    """`a @ b`, with the inner loop written for CPython rather than for looks.

    The comprehension over `zip` runs the multiply-accumulate inside the C
    layer; the equivalent indexed loop is several times slower, which is the
    difference between a training run finishing and not.
    """
    m, k = a.shape
    k2, n = b.shape
    if k != k2:
        raise ValueError(f"cannot multiply {a.shape} by {b.shape}")
    ad, bd = a.data, b.data
    out = _mm(ad, bd, m, k, n)
    result = Tensor(out, (m, n), parents=(a, b))

    def _backward() -> None:
        g = result.grad
        assert g is not None
        if a.requires_grad:
            a._accumulate(_mm(g, _flat_transpose(bd, k, n), m, n, k))  # dC @ B^T
        if b.requires_grad:
            b._accumulate(_mm(_flat_transpose(ad, m, k), g, k, m, n))  # A^T @ dC

    result._backward = _backward
    return result


def add_bias(x: Tensor, bias: Tensor) -> Tensor:
    """Add a `(1, cols)` row to every row of `x`."""
    if bias.shape[0] != 1 or bias.shape[1] != x.cols:
        raise ValueError(f"bias {bias.shape} does not fit {x.shape}")
    n = x.cols
    bd = bias.data
    out = [0.0] * len(x.data)
    for i in range(x.rows):
        off = i * n
        out[off : off + n] = [a + b for a, b in zip(x.data[off : off + n], bd, strict=True)]
    result = Tensor(out, x.shape, parents=(x, bias))

    def _backward() -> None:
        g = result.grad
        assert g is not None
        if x.requires_grad:
            x._accumulate(g)
        if bias.requires_grad:
            acc = [0.0] * n
            for i in range(x.rows):
                off = i * n
                acc = [a + b for a, b in zip(acc, g[off : off + n], strict=True)]
            bias._accumulate(acc)

    result._backward = _backward
    return result


def add(a: Tensor, b: Tensor) -> Tensor:
    """Elementwise sum of two identically shaped tensors — the residual stream."""
    if a.shape != b.shape:
        raise ValueError(f"cannot add {a.shape} and {b.shape}")
    result = Tensor([x + y for x, y in zip(a.data, b.data, strict=True)], a.shape, parents=(a, b))

    def _backward() -> None:
        g = result.grad
        assert g is not None
        if a.requires_grad:
            a._accumulate(g)
        if b.requires_grad:
            b._accumulate(g)

    result._backward = _backward
    return result


def scale(x: Tensor, factor: float) -> Tensor:
    """Multiply by a constant."""
    result = Tensor([v * factor for v in x.data], x.shape, parents=(x,))

    def _backward() -> None:
        g = result.grad
        assert g is not None
        if x.requires_grad:
            x._accumulate([v * factor for v in g])

    result._backward = _backward
    return result


def relu(x: Tensor) -> Tensor:
    """ReLU, not GELU.

    At this width the two are indistinguishable in final loss, and ReLU's
    backward is a comparison instead of a `tanh` per element — which is a real
    fraction of the step time when the arithmetic is interpreted.
    """
    out = [v if v > 0.0 else 0.0 for v in x.data]
    result = Tensor(out, x.shape, parents=(x,))

    def _backward() -> None:
        g = result.grad
        assert g is not None
        if x.requires_grad:
            x._accumulate([gv if xv > 0.0 else 0.0 for gv, xv in zip(g, x.data, strict=True)])

    result._backward = _backward
    return result


def layer_norm(x: Tensor, gain: Tensor, bias: Tensor, *, eps: float = 1e-5) -> Tensor:
    """Normalize each row, then rescale it with learned per-column parameters."""
    n = x.cols
    out = [0.0] * len(x.data)
    cache: list[tuple[float, list[float]]] = []
    gd, bd = gain.data, bias.data
    for i in range(x.rows):
        off = i * n
        row = x.data[off : off + n]
        mean = sum(row) / n
        centred = [v - mean for v in row]
        var = sum(v * v for v in centred) / n
        inv = 1.0 / math.sqrt(var + eps)
        norm = [v * inv for v in centred]
        cache.append((inv, norm))
        out[off : off + n] = [nv * g + b for nv, g, b in zip(norm, gd, bd, strict=True)]
    result = Tensor(out, x.shape, parents=(x, gain, bias))

    def _backward() -> None:
        g = result.grad
        assert g is not None
        dx = [0.0] * len(x.data) if x.requires_grad else []
        dgain = [0.0] * n if gain.requires_grad else []
        dbias = [0.0] * n if bias.requires_grad else []
        for i in range(x.rows):
            off = i * n
            grow = g[off : off + n]
            inv, norm = cache[i]
            if dgain:
                dgain = [a + gv * nv for a, gv, nv in zip(dgain, grow, norm, strict=True)]
            if dbias:
                dbias = [a + gv for a, gv in zip(dbias, grow, strict=True)]
            if not dx:
                continue
            dnorm = [gv * gg for gv, gg in zip(grow, gd, strict=True)]
            mean_dnorm = sum(dnorm) / n
            mean_dnorm_norm = sum(dn * nv for dn, nv in zip(dnorm, norm, strict=True)) / n
            dx[off : off + n] = [
                inv * (dn - mean_dnorm - nv * mean_dnorm_norm)
                for dn, nv in zip(dnorm, norm, strict=True)
            ]
        if dx:
            x._accumulate(dx)
        if dgain:
            gain._accumulate(dgain)
        if dbias:
            bias._accumulate(dbias)

    result._backward = _backward
    return result


def softmax_rows(x: Tensor, *, causal: bool = False) -> Tensor:
    """Row-wise softmax, optionally masking every position after the diagonal.

    Causality is applied *here* rather than by adding -inf beforehand: skipping
    the masked entries avoids computing `exp` for the half of the attention
    matrix that is about to be discarded, which at these sizes is a third of
    the forward pass.
    """
    n = x.cols
    out = [0.0] * len(x.data)
    for i in range(x.rows):
        off = i * n
        limit = min(i + 1, n) if causal else n
        row = x.data[off : off + limit]
        top = max(row)
        exps = [math.exp(v - top) for v in row]
        total = sum(exps)
        out[off : off + limit] = [e / total for e in exps]
    result = Tensor(out, x.shape, parents=(x,))

    def _backward() -> None:
        g = result.grad
        assert g is not None
        if not x.requires_grad:
            return
        dx = [0.0] * len(x.data)
        for i in range(x.rows):
            off = i * n
            limit = min(i + 1, n) if causal else n
            p = out[off : off + limit]
            gr = g[off : off + limit]
            dot = sum(pv * gv for pv, gv in zip(p, gr, strict=True))
            dx[off : off + limit] = [pv * (gv - dot) for pv, gv in zip(p, gr, strict=True)]
        x._accumulate(dx)

    result._backward = _backward
    return result


def transpose(x: Tensor) -> Tensor:
    """Swap rows and columns."""
    m, n = x.shape
    d = x.data
    out = [0.0] * (m * n)
    for i in range(m):
        off = i * n
        for j in range(n):
            out[j * m + i] = d[off + j]
    result = Tensor(out, (n, m), parents=(x,))

    def _backward() -> None:
        g = result.grad
        assert g is not None
        if not x.requires_grad:
            return
        dx = [0.0] * (m * n)
        for j in range(n):
            off = j * m
            for i in range(m):
                dx[i * n + j] = g[off + i]
        x._accumulate(dx)

    result._backward = _backward
    return result


def slice_cols(x: Tensor, start: int, width: int) -> Tensor:
    """Columns `[start, start+width)` — how a head takes its share of the stream."""
    n = x.cols
    out = [0.0] * (x.rows * width)
    for i in range(x.rows):
        off = i * n + start
        out[i * width : (i + 1) * width] = x.data[off : off + width]
    result = Tensor(out, (x.rows, width), parents=(x,))

    def _backward() -> None:
        g = result.grad
        assert g is not None
        if not x.requires_grad:
            return
        dx = [0.0] * len(x.data)
        for i in range(x.rows):
            dx[i * n + start : i * n + start + width] = g[i * width : (i + 1) * width]
        x._accumulate(dx)

    result._backward = _backward
    return result


def concat_cols(parts: Sequence[Tensor]) -> Tensor:
    """Join tensors side by side — how heads rejoin before the output projection."""
    rows = parts[0].rows
    widths = [p.cols for p in parts]
    total = sum(widths)
    out = [0.0] * (rows * total)
    for i in range(rows):
        cursor = i * total
        for part, w in zip(parts, widths, strict=True):
            out[cursor : cursor + w] = part.data[i * w : (i + 1) * w]
            cursor += w
    result = Tensor(out, (rows, total), parents=tuple(parts))

    def _backward() -> None:
        g = result.grad
        assert g is not None
        cursor = 0
        for part, w in zip(parts, widths, strict=True):
            if part.requires_grad:
                dp = [0.0] * (rows * w)
                for i in range(rows):
                    off = i * total + cursor
                    dp[i * w : (i + 1) * w] = g[off : off + w]
                part._accumulate(dp)
            cursor += w

    result._backward = _backward
    return result


def select_rows(x: Tensor, indices: Sequence[int]) -> Tensor:
    """Keep only these rows, in this order.

    This is what makes the output projection affordable: the loss is charged on
    a handful of action positions, so the hidden states are cut down to those
    before being multiplied by the vocabulary matrix. The rows that are dropped
    receive no gradient because nothing downstream reads them, which is exactly
    what would have happened had they been projected and then ignored.
    """
    n = x.cols
    out: list[float] = []
    for i in indices:
        if not 0 <= i < x.rows:
            raise IndexError(f"row {i} out of range for {x.shape}")
        out.extend(x.data[i * n : (i + 1) * n])
    result = Tensor(out, (len(indices), n), parents=(x,))

    def _backward() -> None:
        g = result.grad
        assert g is not None
        if not x.requires_grad:
            return
        dx = [0.0] * len(x.data)
        for slot, i in enumerate(indices):
            off, src = i * n, slot * n
            dx[off : off + n] = [a + b for a, b in zip(dx[off : off + n], g[src : src + n], strict=True)]
        x._accumulate(dx)

    result._backward = _backward
    return result


def embed(weight: Tensor, ids: Sequence[int]) -> Tensor:
    """Look up one row of `weight` per id."""
    n = weight.cols
    out: list[float] = []
    for token in ids:
        out.extend(weight.data[token * n : (token + 1) * n])
    result = Tensor(out, (len(ids), n), parents=(weight,))

    def _backward() -> None:
        g = result.grad
        assert g is not None
        if not weight.requires_grad:
            return
        dw = [0.0] * len(weight.data)
        for i, token in enumerate(ids):
            off = token * n
            src = i * n
            dw[off : off + n] = [a + b for a, b in zip(dw[off : off + n], g[src : src + n], strict=True)]
        weight._accumulate(dw)

    result._backward = _backward
    return result


def cross_entropy(
    logits: Tensor, targets: Sequence[int], mask: Sequence[float] | None = None
) -> Tensor:
    """Mean negative log-likelihood over the rows the mask keeps.

    The mask is the whole point of training on interaction tokens: the loss is
    charged only on positions that are part of an *action*. Scoring the model
    on the instruction text as well would spend most of the gradient teaching
    it to spell task descriptions, which no one is going to ask it to do.
    """
    rows, vocab = logits.shape
    weights = list(mask) if mask is not None else [1.0] * rows
    total_weight = sum(weights)
    if total_weight <= 0.0:
        raise ValueError("cross_entropy needs at least one unmasked row")
    probs: list[list[float]] = []
    loss = 0.0
    for i in range(rows):
        off = i * vocab
        row = logits.data[off : off + vocab]
        if weights[i] == 0.0:
            probs.append([])
            continue
        top = max(row)
        exps = [math.exp(v - top) for v in row]
        total = sum(exps)
        p = [e / total for e in exps]
        probs.append(p)
        loss -= weights[i] * math.log(max(p[targets[i]], 1e-12))
    result = Tensor([loss / total_weight], (1, 1), parents=(logits,))

    def _backward() -> None:
        g = result.grad
        assert g is not None
        if not logits.requires_grad:
            return
        seed = g[0] / total_weight
        dl = [0.0] * len(logits.data)
        for i in range(rows):
            if weights[i] == 0.0:
                continue
            off = i * vocab
            p = probs[i]
            w = weights[i] * seed
            dl[off : off + vocab] = [w * v for v in p]
            dl[off + targets[i]] -= w
        logits._accumulate(dl)

    result._backward = _backward
    return result


# --------------------------------------------------------------------------- #
# Optimization
# --------------------------------------------------------------------------- #


class Adam:
    """Adam with decoupled weight decay and global gradient clipping.

    Clipping is not decoration here. The corpus is small enough that a single
    unusual batch can produce a step large enough to knock the model out of the
    basin it spent hundreds of steps finding, and on a run that takes minutes
    per epoch a restart is expensive.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        *,
        lr: float = 3e-3,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        clip: float = 1.0,
    ) -> None:
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.clip = clip
        self.step_count = 0
        self._m = [[0.0] * len(p.data) for p in self.params]
        self._v = [[0.0] * len(p.data) for p in self.params]

    def zero_grad(self) -> None:
        for p in self.params:
            p.zero_grad()

    def grad_norm(self) -> float:
        total = 0.0
        for p in self.params:
            if p.grad is not None:
                total += sum(g * g for g in p.grad)
        return math.sqrt(total)

    def step(self, *, lr: float | None = None) -> float:
        """Apply one update; returns the pre-clip gradient norm."""
        norm = self.grad_norm()
        factor = 1.0
        if self.clip > 0.0 and norm > self.clip:
            factor = self.clip / (norm + 1e-12)
        self.step_count += 1
        rate = self.lr if lr is None else lr
        bias1 = 1.0 - self.beta1**self.step_count
        bias2 = 1.0 - self.beta2**self.step_count
        b1, b2, eps = self.beta1, self.beta2, self.eps
        for index, p in enumerate(self.params):
            if p.grad is None:
                continue
            m, v, data = self._m[index], self._v[index], p.data
            for i, raw in enumerate(p.grad):
                g = raw * factor
                m[i] = b1 * m[i] + (1.0 - b1) * g
                v[i] = b2 * v[i] + (1.0 - b2) * g * g
                update = (m[i] / bias1) / (math.sqrt(v[i] / bias2) + eps)
                if self.weight_decay:
                    update += self.weight_decay * data[i]
                data[i] -= rate * update
        return norm
