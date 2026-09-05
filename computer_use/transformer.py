"""A decoder-only transformer over interaction tokens.

Small on purpose. The claim this package is trying to support is not "a large
model can operate a GUI" — everyone already believes that. It is that the
*data* this pipeline generates carries the signal, and the cleanest way to show
that is to train something small enough to be read end to end and watch it pick
the signal up. 55k parameters, two layers, no pretraining, no outside corpus:
whatever it learns, it learned here.

Three choices are worth knowing about before reading the code.

**Weight tying.** The output projection is the transpose of the token
embedding. At this scale untied output weights are a third of the parameter
budget spent on a matrix that sees one gradient per token occurrence, and the
task is largely *copying* — reading a coordinate out of the observation and
emitting it — which is exactly what tying makes easy.

**The loss is only projected where it is charged.** Computing logits at all
`T` positions costs `T x d x vocab`, which is the single largest matmul in the
model and dwarfs everything else. But the loss is charged on the four-odd
tokens of the action, so the hidden states are sliced down to the supervised
positions *before* the projection. Same gradients, roughly a tenth of the time.

**Sequences are not padded.** Every example is run alone at its own length.
Padding to a common length would cost the square of the longest sequence on
every example, and with a median of ~120 tokens against a maximum near 190,
the batch would spend most of its arithmetic on `<pad>`. Gradient accumulation
across a minibatch gets the same update without the waste.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from computer_use import nn
from computer_use.nn import Tensor
from computer_use.tokens import VOCAB

__all__ = ["GPT", "ModelConfig", "generate", "load_model", "save_model"]


@dataclass(frozen=True)
class ModelConfig:
    """Model shape. The defaults are what the reported runs use."""

    vocab_size: int = VOCAB.size
    d_model: int = 48
    n_heads: int = 3
    n_layers: int = 2
    d_ff: int = 96
    max_len: int = 208
    #: Width of an injected visual feature, or 0 for a text-only model. When
    #: set, the model carries a projection from that width into `d_model` and
    #: `hidden` will accept features to add at chosen positions.
    visual_dim: int = 0

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError(
                f"d_model={self.d_model} does not divide into {self.n_heads} heads"
            )

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


class Block:
    """Pre-norm attention and MLP, each around a residual connection.

    Pre-norm rather than post-norm because a two-layer model trained without a
    warmup schedule diverges often enough from post-norm to matter, and warmup
    is one more thing to tune on runs that take twenty minutes each.
    """

    def __init__(self, config: ModelConfig, rng: random.Random) -> None:
        d = config.d_model
        self.config = config
        self.ln1_g = Tensor([1.0] * d, (1, d), requires_grad=True)
        self.ln1_b = Tensor([0.0] * d, (1, d), requires_grad=True)
        self.wq = nn.parameter(d, d, rng)
        self.wk = nn.parameter(d, d, rng)
        self.wv = nn.parameter(d, d, rng)
        # Scaled down by depth: the standard trick that keeps the residual
        # stream from growing as layers are stacked.
        self.wo = nn.parameter(d, d, rng, scale_=1.0 / math.sqrt(d * 2 * config.n_layers))
        self.ln2_g = Tensor([1.0] * d, (1, d), requires_grad=True)
        self.ln2_b = Tensor([0.0] * d, (1, d), requires_grad=True)
        self.w1 = nn.parameter(d, config.d_ff, rng)
        self.b1 = Tensor([0.0] * config.d_ff, (1, config.d_ff), requires_grad=True)
        self.w2 = nn.parameter(
            config.d_ff, d, rng, scale_=1.0 / math.sqrt(config.d_ff * 2 * config.n_layers)
        )
        self.b2 = Tensor([0.0] * d, (1, d), requires_grad=True)

    def params(self) -> list[Tensor]:
        return [
            self.ln1_g, self.ln1_b, self.wq, self.wk, self.wv, self.wo,
            self.ln2_g, self.ln2_b, self.w1, self.b1, self.w2, self.b2,
        ]

    def __call__(self, x: Tensor, record: list[Tensor] | None = None) -> Tensor:
        cfg = self.config
        head_dim = cfg.head_dim
        inv = 1.0 / math.sqrt(head_dim)

        normed = nn.layer_norm(x, self.ln1_g, self.ln1_b)
        q, k, v = (nn.matmul(normed, w) for w in (self.wq, self.wk, self.wv))

        heads: list[Tensor] = []
        for h in range(cfg.n_heads):
            start = h * head_dim
            qh = nn.slice_cols(q, start, head_dim)
            kh = nn.slice_cols(k, start, head_dim)
            vh = nn.slice_cols(v, start, head_dim)
            scores = nn.scale(nn.matmul(qh, nn.transpose(kh)), inv)
            weights = nn.softmax_rows(scores, causal=True)
            # Kept by reference, not recomputed: the matrix is already built,
            # so watching where the model looks costs nothing but a list append
            # and cannot perturb what it does.
            if record is not None:
                record.append(weights)
            heads.append(nn.matmul(weights, vh))
        attended = nn.matmul(nn.concat_cols(heads), self.wo)
        x = nn.add(x, attended)

        normed = nn.layer_norm(x, self.ln2_g, self.ln2_b)
        hidden = nn.relu(nn.add_bias(nn.matmul(normed, self.w1), self.b1))
        return nn.add(x, nn.add_bias(nn.matmul(hidden, self.w2), self.b2))


class GPT:
    """Token and position embeddings, N blocks, a final norm, tied output."""

    def __init__(self, config: ModelConfig | None = None, *, seed: int = 0) -> None:
        self.config = config or ModelConfig()
        rng = random.Random(seed)
        d = self.config.d_model
        self.tok = nn.parameter(self.config.vocab_size, d, rng, scale_=0.02)
        self.pos = nn.parameter(self.config.max_len, d, rng, scale_=0.02)
        self.blocks = [Block(self.config, rng) for _ in range(self.config.n_layers)]
        self.ln_g = Tensor([1.0] * d, (1, d), requires_grad=True)
        self.ln_b = Tensor([0.0] * d, (1, d), requires_grad=True)
        # Scaled well down: a visual feature enters the residual stream on top
        # of an embedding that is already the right size, and a projection
        # initialised at the usual scale swamps it for the first few hundred
        # steps — the model spends that time recovering from its own eyes.
        self.w_vis = (
            nn.parameter(self.config.visual_dim, d, rng, scale_=0.02)
            if self.config.visual_dim
            else None
        )

    # -- plumbing ---------------------------------------------------------- #

    def params(self) -> list[Tensor]:
        out = [self.tok, self.pos, self.ln_g, self.ln_b]
        if self.w_vis is not None:
            out.append(self.w_vis)
        for block in self.blocks:
            out.extend(block.params())
        return out

    @property
    def n_params(self) -> int:
        return sum(len(p.data) for p in self.params())

    def zero_grad(self) -> None:
        for p in self.params():
            p.zero_grad()

    # -- forward ----------------------------------------------------------- #

    def hidden(
        self,
        ids: Sequence[int],
        record: list[Tensor] | None = None,
        visual: Sequence[tuple[int, Sequence[float]]] | None = None,
    ) -> Tensor:
        """Final-layer states for every position.

        Pass `record` to collect the attention matrices as they are computed,
        one per layer per head in order. Nothing else changes: the same tensors
        are returned either way, and `diagnose` uses this to ask what the model
        was looking at when it emitted a coordinate.
        """
        if len(ids) > self.config.max_len:
            raise ValueError(
                f"sequence of {len(ids)} exceeds max_len={self.config.max_len}; "
                "truncate when building the corpus, not here — a silently "
                "clipped example is a mislabelled one"
            )
        x = nn.add(nn.embed(self.tok, ids), nn.embed(self.pos, range(len(ids))))
        if visual:
            if self.w_vis is None:
                raise ValueError(
                    "visual features were supplied to a model built without "
                    "visual_dim; the projection that would read them does not "
                    "exist, and silently ignoring them would train a blind "
                    "model that reports as a seeing one"
                )
            positions = [p for p, _ in visual]
            flat: list[float] = []
            for _, feature in visual:
                if len(feature) != self.config.visual_dim:
                    raise ValueError(
                        f"visual feature of {len(feature)} against "
                        f"visual_dim={self.config.visual_dim}"
                    )
                flat.extend(feature)
            features = Tensor(flat, (len(visual), self.config.visual_dim))
            x = nn.add_rows_at(x, nn.matmul(features, self.w_vis), positions)
        for block in self.blocks:
            x = block(x, record)
        return nn.layer_norm(x, self.ln_g, self.ln_b)

    def logits(
        self,
        ids: Sequence[int],
        positions: Sequence[int] | None = None,
        visual: Sequence[tuple[int, Sequence[float]]] | None = None,
    ) -> Tensor:
        """Vocabulary scores, at `positions` only if given.

        Restricting the projection is not an approximation — the rows it drops
        are the ones nothing reads.
        """
        states = self.hidden(ids, visual=visual)
        if positions is not None:
            states = nn.select_rows(states, positions)
        return nn.matmul(states, nn.transpose(self.tok))

    def loss(
        self,
        ids: Sequence[int],
        supervised: Sequence[int],
        visual: Sequence[tuple[int, Sequence[float]]] | None = None,
    ) -> Tensor:
        """Next-token loss, charged only at `supervised` positions.

        `supervised` holds positions *whose next token* is scored, so each one
        must be less than `len(ids) - 1`.
        """
        if not supervised:
            raise ValueError("nothing to train on: no supervised positions")
        scores = self.logits(ids, supervised, visual=visual)
        targets = [ids[p + 1] for p in supervised]
        return nn.cross_entropy(scores, targets)


def generate(
    model: GPT,
    prefix: Sequence[int],
    *,
    max_new: int = 24,
    stop: Sequence[int] = (),
    temperature: float = 0.0,
    rng: random.Random | None = None,
    visual: Sequence[tuple[int, Sequence[float]]] | None = None,
) -> list[int]:
    """Continue `prefix`, returning only the new ids.

    Greedy by default. A policy is being *scored by execution* here, so the
    sampling temperature is a confound rather than a feature: at t=0 a run is
    reproducible and a failure is the model's, not the dice's.
    """
    stop_set = set(stop)
    ids = list(prefix)
    produced: list[int] = []
    limit = model.config.max_len
    for _ in range(max_new):
        if len(ids) >= limit:
            break
        row = model.logits(ids, [len(ids) - 1], visual=visual).data
        if temperature <= 0.0:
            nxt = max(range(len(row)), key=row.__getitem__)
        else:
            top = max(row)
            weights = [math.exp((v - top) / temperature) for v in row]
            picker = rng or random.Random(0)
            nxt = picker.choices(range(len(row)), weights=weights, k=1)[0]
        ids.append(nxt)
        produced.append(nxt)
        if nxt in stop_set:
            break
    return produced


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def save_model(model: GPT, path: str | Path) -> None:
    """Weights as JSON.

    Plain JSON rather than pickle because a checkpoint is a result: it should
    be readable by anything, including a person, and it should not execute
    code when it is opened.
    """
    payload = {
        "config": asdict(model.config),
        "params": [p.data for p in model.params()],
    }
    Path(path).write_text(json.dumps(payload), encoding="utf-8")


def load_model(path: str | Path) -> GPT:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    model = GPT(ModelConfig(**payload["config"]))
    tensors = model.params()
    if len(payload["params"]) != len(tensors):
        raise ValueError(
            f"checkpoint does not match the model shape: {len(payload['params'])} "
            f"tensors on disk against {len(tensors)} in the model"
        )
    for tensor, values in zip(tensors, payload["params"], strict=True):
        if len(values) != len(tensor.data):
            raise ValueError("checkpoint does not match the model shape")
        tensor.data = list(values)
    return model
