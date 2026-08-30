"""User interactions as tokens, so a transformer can read and write them.

Today a trajectory reaches a trainer as JSON text:

    {"action": "left_click", "coordinate": [290, 218]}

A subword tokenizer turns that into roughly twenty tokens, most of them
punctuation, and it spells the coordinate as digits — so the model has to learn
that "2", "9", "0" composes into a horizontal position, and that the same pixel
written "291" means almost the same thing. Neither is a fact about operating a
GUI. It is an encoding tax, paid on every step of every episode.

This module gives interactions their own vocabulary. One symbol per action
kind, one symbol per quantized coordinate, one symbol per character the
renderer can actually draw. A click becomes three tokens instead of twenty, and
"nearby pixels are nearby" is built into the representation rather than learned
from digit strings.

The bar for a lossy encoding is not that the numbers round-trip. It is that the
*action still works*: a click decoded from its tokens has to land on the same
control it started on, or the tokenizer has quietly rewritten the data. The
grid here is the coarsest one that passes that check on every control in every
generated world — see `tests/test_tokens.py`, which measures it rather than
assuming it.

What comes out is `TokenBatch`: padded id matrices and attention masks, plus a
`torch` adapter for anyone who has it installed.
"""

from __future__ import annotations

import string
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from computer_use._render import _GLYPHS
from computer_use.types import Action, ActionKind, Trajectory

#: Reserved ids. PAD is 0 so a zero-filled matrix is a batch of empty rows.
PAD, BOS, EOS, SEP = "<pad>", "<bos>", "<eos>", "<sep>"
OBS, ACT = "<obs>", "<act>"
SPECIALS: tuple[str, ...] = (PAD, BOS, EOS, SEP, OBS, ACT)

#: Coordinate resolution. 1280x720 into 64x36 cells is 20px square — chosen
#: because it is the coarsest grid on which every control in every generated
#: world still hit-tests to itself after a round trip, and a coarser grid is a
#: smaller vocabulary but a silently corrupted dataset.
X_BINS, Y_BINS = 64, 36

#: The characters an interaction can carry. The font supplies the alphabet the
#: environment can *draw*, and lowercase is added because that is not the same
#: set as what a user can *type*: the renderer upper-cases captions on the way
#: to the screen, while a field stores the string it was given and a verifier
#: compares against that. Folding case at encode time turned every
#: `type "ada@example.com"` into `ADA@EXAMPLE.COM` and failed the task —
#: silently, since the click coordinates it also encoded were all still right.
CHARS: tuple[str, ...] = tuple(sorted(set(_GLYPHS) | set(string.ascii_lowercase)))

#: A toggle's state, for the observation. Without these two symbols a screen
#: encodes identically before and after a checkbox is flipped, so the same
#: context carries two different correct actions — measured at 59% of a
#: generated training corpus, and unlearnable by construction.
STATES: tuple[str, ...] = ("on", "off", "focus")

SCROLL_DIRECTIONS: tuple[str, ...] = ("up", "down", "left", "right")
#: Scroll amounts are small integers in practice; larger ones clamp.
MAX_SCROLL = 8

#: How many controls a screen can be marked with. One more than the element
#: limit any corpus uses, so a mark index is never silently clamped onto a
#: different control.
MAX_MARKS = 16


def _vocabulary() -> tuple[str, ...]:
    """Every token, in a fixed order so ids are stable across processes."""
    tokens: list[str] = list(SPECIALS)
    tokens += [f"<k:{kind.value}>" for kind in ActionKind]
    tokens += [f"<x:{i}>" for i in range(X_BINS)]
    tokens += [f"<y:{i}>" for i in range(Y_BINS)]
    tokens += [f"<s:{s}>" for s in STATES]
    tokens += [f"<d:{d}>" for d in SCROLL_DIRECTIONS]
    tokens += [f"<n:{n}>" for n in range(MAX_SCROLL + 1)]
    tokens += [f"<c:{c}>" for c in CHARS]
    tokens.append("<c:?unk>")
    # Marks are appended, never inserted. Everything above has a stable id, and
    # a checkpoint's embedding rows are addressed by id — so slotting a new
    # block into the middle silently renumbers every token after it and
    # repoints a trained model's rows at symbols it never saw. Nothing raises;
    # the model simply becomes a different model. New symbols go on the end.
    tokens += [f"<m:{i}>" for i in range(MAX_MARKS)]
    return tuple(tokens)


@dataclass(frozen=True)
class Vocabulary:
    """A fixed token list with id lookup in both directions."""

    tokens: tuple[str, ...] = field(default_factory=_vocabulary)

    def __post_init__(self) -> None:
        if len(set(self.tokens)) != len(self.tokens):
            raise ValueError("duplicate token in vocabulary")

    @property
    def size(self) -> int:
        return len(self.tokens)

    @property
    def ids(self) -> dict[str, int]:
        return {token: index for index, token in enumerate(self.tokens)}

    def id(self, token: str) -> int:
        try:
            return self.ids[token]
        except KeyError:
            return self.ids["<c:?unk>"]

    def token(self, index: int) -> str:
        return self.tokens[index]

    def encode(self, tokens: Sequence[str]) -> list[int]:
        lookup = self.ids
        return [lookup.get(t, lookup["<c:?unk>"]) for t in tokens]

    def decode(self, ids: Sequence[int]) -> list[str]:
        return [self.tokens[i] for i in ids if 0 <= i < len(self.tokens)]


VOCAB = Vocabulary()


# --------------------------------------------------------------------------- #
# Coordinates
# --------------------------------------------------------------------------- #


def quantize(x: int, y: int, *, width: int = 1280, height: int = 720) -> tuple[int, int]:
    """Pixel to cell. Clamped, so an out-of-frame coordinate stays in vocabulary."""
    cx = min(X_BINS - 1, max(0, x * X_BINS // max(width, 1)))
    cy = min(Y_BINS - 1, max(0, y * Y_BINS // max(height, 1)))
    return cx, cy


def dequantize(cx: int, cy: int, *, width: int = 1280, height: int = 720) -> tuple[int, int]:
    """Cell back to a pixel — the cell's centre, which is the safest point in it.

    Taking the centre rather than a corner is what makes the round trip land
    inside the same control: a corner sits on the boundary, and a boundary
    rounds into the neighbour.
    """
    x = (cx * width) // X_BINS + (width // X_BINS) // 2
    y = (cy * height) // Y_BINS + (height // Y_BINS) // 2
    return min(x, width - 1), min(y, height - 1)


# --------------------------------------------------------------------------- #
# Actions
# --------------------------------------------------------------------------- #


def mark_of(
    action: Action, marks: Sequence[tuple[int, int]], *,
    width: int = 1280, height: int = 720,
) -> int | None:
    """Which marked control this action clicks, if any.

    `marks` is the quantized cell of each control on screen, in the order the
    observation lists them. Matching is done on the cell rather than the pixel
    because that is the resolution the model works at either way.
    """
    if action.coordinate is None or action.scroll_direction is not None:
        return None
    cell = quantize(*action.coordinate, width=width, height=height)
    for index, candidate in enumerate(marks):
        if candidate == cell and index < MAX_MARKS:
            return index
    return None


def encode_action(
    action: Action, *, width: int = 1280, height: int = 720,
    marks: Sequence[tuple[int, int]] | None = None,
) -> list[str]:
    """One interaction as tokens: kind, then only the fields that kind uses.

    With `marks`, a click on a marked control is emitted as a single `<m:i>`
    naming that control, instead of the `<x:..> <y:..>` pair naming a cell.

    The difference is not notation. Emitting a coordinate makes the model
    perform a two-hop copy — find the label the instruction names, then reach
    back for the number beside it — over tokens that carry no spatial meaning
    of their own; `<x:8>` and `<x:9>` are as unrelated in the embedding as
    `<x:8>` and `<k:type>` until something teaches otherwise. Naming a mark
    turns the same decision into a choice among the handful of controls that
    are actually on the screen. A click that lands on no marked control still
    falls back to coordinates, so the encoding never loses an action it could
    previously express.
    """
    out = [f"<k:{action.kind.value}>"]
    if marks is not None:
        index = mark_of(action, marks, width=width, height=height)
        if index is not None:
            return [*out, f"<m:{index}>"]
    if action.start_coordinate is not None:
        cx, cy = quantize(*action.start_coordinate, width=width, height=height)
        out += [f"<x:{cx}>", f"<y:{cy}>"]
    if action.coordinate is not None:
        cx, cy = quantize(*action.coordinate, width=width, height=height)
        out += [f"<x:{cx}>", f"<y:{cy}>"]
    if action.scroll_direction is not None:
        out.append(f"<d:{action.scroll_direction}>")
    if action.scroll_amount is not None:
        out.append(f"<n:{min(int(action.scroll_amount), MAX_SCROLL)}>")
    if action.text:
        out += [f"<c:{c}>" for c in action.text]
    return out


def resolve_marks(
    tokens: Sequence[str], marks: Sequence[tuple[int, int]]
) -> list[str]:
    """Rewrite `<m:i>` back into the cell it names, using the screen it named it on.

    A mark is only meaningful against the observation that produced it, so
    resolution happens where that observation is in hand rather than inside
    `decode_action`, which has no screen. An index past the end of `marks` —
    the model naming a control that is not there — is dropped rather than
    clamped onto whichever control happens to be last: a click on the wrong
    control scores as a wrong click, while a malformed action scores as
    malformed, and those are different failures worth telling apart.
    """
    out: list[str] = []
    for token in tokens:
        if not token.startswith("<m:"):
            out.append(token)
            continue
        index = int(token[3:-1])
        if index < len(marks):
            cx, cy = marks[index]
            out += [f"<x:{cx}>", f"<y:{cy}>"]
    return out


def decode_action(
    tokens: Sequence[str], *, width: int = 1280, height: int = 720
) -> Action | None:
    """Tokens back to an action, or None if they do not start with a kind."""
    if not tokens or not tokens[0].startswith("<k:"):
        return None
    kind = ActionKind(tokens[0][3:-1])

    points: list[tuple[int, int]] = []
    direction: str | None = None
    amount: int | None = None
    characters: list[str] = []

    index = 1
    while index < len(tokens):
        token = tokens[index]
        if token.startswith("<x:") and index + 1 < len(tokens) and tokens[index + 1].startswith("<y:"):
            points.append(dequantize(
                int(token[3:-1]), int(tokens[index + 1][3:-1]),
                width=width, height=height,
            ))
            index += 2
            continue
        if token.startswith("<d:"):
            direction = token[3:-1]
        elif token.startswith("<n:"):
            amount = int(token[3:-1])
        elif token.startswith("<c:"):
            body = token[3:-1]
            # A visible marker rather than a space: an unrepresentable
            # character should show up in the data, not disappear into
            # whitespace that reads as a deliberate blank.
            characters.append("\ufffd" if body == "?unk" else body)
        index += 1

    start = points[0] if len(points) > 1 else None
    coordinate = points[-1] if points else None
    return Action(
        kind=kind,
        coordinate=coordinate,
        start_coordinate=start,
        text="".join(characters) or None,
        scroll_direction=direction,
        scroll_amount=amount,
    )


# --------------------------------------------------------------------------- #
# Screens and trajectories
# --------------------------------------------------------------------------- #


def encode_screen(
    screen: Any,
    *,
    limit: int = 24,
    label_chars: int = 24,
    label_first: bool = False,
) -> list[str]:
    """A parsed screen as tokens: what is on it and where.

    The point of spending tokens on the observation is that a policy trained on
    actions alone learns a fixed sequence, while one that sees the screen can
    learn to *look*. Elements come in reading order and are truncated rather
    than sampled, so the encoding is deterministic.

    `label_first` moves each control's coordinate to *after* its label. The
    field order is not cosmetic: the model's job is to find the label the
    instruction names and emit the coordinate attached to it, and an attention
    head matches a pattern and then copies what *follows* the match. With the
    coordinate first, the answer sits behind the thing that identifies it, and
    the copy has to run backwards. With the label first, it runs forwards —
    which is the operation two layers implement without being taught.

    Both orders are kept because the difference is an empirical question, and
    `pretrain` trains under either so the answer can be measured rather than
    asserted.
    """
    out: list[str] = [OBS]
    for element in list(getattr(screen, "elements", ()))[:limit]:
        if element.box is None:
            continue
        cx, cy = quantize(*element.click,
                          width=screen.width, height=screen.height)
        point = [f"<x:{cx}>", f"<y:{cy}>"]
        state: list[str] = []
        checked = getattr(element, "checked", None)
        if checked is not None:
            state.append(f"<s:{'on' if checked else 'off'}>")
        if getattr(element, "focused", False):
            state.append("<s:focus>")
        label = [f"<c:{c}>" for c in element.label.upper()[:label_chars]]
        # State stays next to the coordinate in both orders: it qualifies the
        # control, not the name, and splitting it from the point would make the
        # two encodings differ in more than the one thing under test.
        out += [*label, *point, *state] if label_first else [*point, *state, *label]
        out.append(SEP)
    return out


def encode_trajectory(
    trajectory: Trajectory,
    *,
    vocab: Vocabulary = VOCAB,
    include_screens: bool = False,
    width: int = 1280,
    height: int = 720,
) -> list[int]:
    """A whole episode as ids: the task, then the interactions that followed."""
    from computer_use.perception import parse_screen

    tokens: list[str] = [BOS]
    tokens += [f"<c:{c}>" for c in trajectory.task.upper()]
    for step in trajectory.steps:
        if include_screens and step.observation is not None:
            tokens += encode_screen(parse_screen(step.observation.data))
        tokens.append(ACT)
        tokens += encode_action(step.action, width=width, height=height)
    tokens.append(EOS)
    return vocab.encode(tokens)


def decode_actions(
    ids: Sequence[int], *, vocab: Vocabulary = VOCAB,
    width: int = 1280, height: int = 720,
) -> list[Action]:
    """Every action in an id sequence, ignoring the task and any screens."""
    tokens = vocab.decode(ids)
    actions: list[Action] = []
    current: list[str] | None = None
    for token in tokens:
        if token == ACT:
            if current:
                decoded = decode_action(current, width=width, height=height)
                if decoded is not None:
                    actions.append(decoded)
            current = []
            continue
        if token in (EOS, OBS):
            if current:
                decoded = decode_action(current, width=width, height=height)
                if decoded is not None:
                    actions.append(decoded)
            current = None
            continue
        if current is not None:
            current.append(token)
    if current:
        decoded = decode_action(current, width=width, height=height)
        if decoded is not None:
            actions.append(decoded)
    return actions


# --------------------------------------------------------------------------- #
# Batching
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TokenBatch:
    """Padded ids and masks — what a transformer's forward pass wants."""

    input_ids: list[list[int]]
    attention_mask: list[list[int]]
    rewards: list[float] = field(default_factory=list)
    vocab_size: int = VOCAB.size

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.input_ids), len(self.input_ids[0]) if self.input_ids else 0)

    def to_torch(self) -> Any:
        """`{input_ids, attention_mask, rewards}` as tensors, if torch is here.

        Kept behind a call rather than done eagerly so the tokenizer itself
        stays dependency-free — the rest of this package runs without torch,
        and a tokenizer that imports it at module scope would break that.
        """
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "to_torch needs PyTorch. The batch is plain lists without it, "
                "which most trainers accept."
            ) from exc
        out = {
            "input_ids": torch.tensor(self.input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask, dtype=torch.long),
        }
        if self.rewards:
            out["rewards"] = torch.tensor(self.rewards, dtype=torch.float)
        return out


def to_token_batch(
    trajectories: Iterable[Trajectory],
    *,
    vocab: Vocabulary = VOCAB,
    include_screens: bool = False,
    max_length: int | None = None,
) -> TokenBatch:
    """Tokenize episodes into one padded batch.

    Truncation keeps the *head* of a sequence: it starts with the task, and a
    row that has lost the instruction is not a shorter example, it is a
    mislabelled one.
    """
    rows = [
        encode_trajectory(t, vocab=vocab, include_screens=include_screens)
        for t in trajectories
    ]
    rewards = [t.reward for t in trajectories]
    if max_length is not None:
        rows = [row[:max_length] for row in rows]
    longest = max((len(row) for row in rows), default=0)
    pad = vocab.id(PAD)
    return TokenBatch(
        input_ids=[row + [pad] * (longest - len(row)) for row in rows],
        attention_mask=[[1] * len(row) + [0] * (longest - len(row)) for row in rows],
        rewards=rewards,
        vocab_size=vocab.size,
    )


def cost(trajectory: Trajectory, *, vocab: Vocabulary = VOCAB) -> dict[str, int]:
    """Tokens this encoding spends against what the JSON rendering would.

    The JSON figure is a character count over four, the usual rule of thumb for
    a subword tokenizer on punctuation-heavy text. It is an estimate and is
    labelled as one; the interaction figure is exact.
    """
    from computer_use.dataset import render_action

    interaction = sum(len(encode_action(s.action)) for s in trajectory.steps)
    json_chars = sum(len(render_action(s.action)) for s in trajectory.steps)
    return {
        "interaction_tokens": interaction,
        "json_chars": json_chars,
        "json_tokens_estimate": json_chars // 4,
        "steps": len(trajectory.steps),
    }


__all__ = [
    "ACT",
    "BOS",
    "CHARS",
    "EOS",
    "MAX_MARKS",
    "OBS",
    "PAD",
    "SEP",
    "STATES",
    "VOCAB",
    "X_BINS",
    "Y_BINS",
    "TokenBatch",
    "Vocabulary",
    "cost",
    "decode_action",
    "decode_actions",
    "dequantize",
    "encode_action",
    "encode_screen",
    "encode_trajectory",
    "mark_of",
    "quantize",
    "resolve_marks",
    "to_token_batch",
]
