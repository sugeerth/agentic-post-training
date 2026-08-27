"""Why the model gets a decision wrong, rather than how often.

`pretrain.step_accuracy` answers one question — what fraction of held-out
decisions come back exactly right — and answers it with a single bit per
example. That is the right headline and the wrong instrument for improving
anything, because every distinct way of being wrong collapses into the same
zero:

* a click one cell from the target scores the same as a click on the far side
  of the screen;
* a fifteen-character `type` that reproduces fourteen characters scores the
  same as one that emits nothing;
* an example whose answer is *not present in its own context* scores the same
  as one the model simply misread.

This module takes those apart. It is a measuring device, not a training
change: nothing here touches the model, and every function is a pure reading
of predictions the model has already made.

Three readings, in the order they are worth taking.

**The ceiling.** The premise of the whole encoding is that the answer is
already in the context — the coordinate to click sits next to the label it
belongs to, the string to type sits inside the instruction — so the task is a
pointer operation rather than a feat of memory. `ceiling` checks that premise
per example instead of assuming it. Where it fails, the model is being graded
on something it cannot derive, and that share of the score is unavailable to
any model of any size.

**The floor.** `baselines` scores non-learned policies on the same examples:
uniform choice among the perceived controls, always the first control, and the
control whose label overlaps the instruction most. The last of those is a rule
anyone could write in ten lines, and a learned model that does not clear it
has not earned the word *learned*. The uniform rate matters just as much in
the other direction: on a screen with ten controls, chance is ten percent, so
a raw accuracy has to be read against that and not against zero.

**The decomposition.** `field_scores` splits a prediction into the parts that
can independently be wrong — the verb, the column, the row, the characters —
and records how far off the wrong clicks were. A model that has learned to
find the right control but quantizes it into the neighbouring cell is one bug
away from working. A model that clicks uniformly at random is not. Exact-match
accuracy cannot tell those apart; this can.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from computer_use.nn import Tensor
from computer_use.pretrain import Example, decode_generated
from computer_use.tokens import ACT, EOS, OBS, SEP, VOCAB, Vocabulary, quantize
from computer_use.transformer import GPT, generate
from computer_use.types import Action

__all__ = [
    "Attention",
    "Baseline",
    "Ceiling",
    "Element",
    "FieldScores",
    "attention_to_target",
    "baselines",
    "ceiling",
    "context_elements",
    "context_instruction",
    "field_scores",
    "format_diagnosis",
    "predictions",
]


# --------------------------------------------------------------------------- #
# Reading the context back
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Element:
    """One control, as the model sees it: a cell and a label."""

    cx: int
    cy: int
    label: str
    state: str | None = None
    focused: bool = False


def context_elements(
    example: Example, *, vocab: Vocabulary = VOCAB
) -> list[Element]:
    """The controls in an example's observation span.

    Decoded from the ids rather than carried alongside them on purpose: this
    is the screen *as tokenized*, so anything perception dropped or the
    element limit truncated is already gone. A ceiling computed against the
    original `Screenshot` would be a ceiling for a model that cannot see it.
    """
    tokens = vocab.decode(example.ids[: example.prompt_length])
    try:
        start = tokens.index(OBS) + 1
    except ValueError:
        return []
    end = tokens.index(ACT) if ACT in tokens else len(tokens)

    out: list[Element] = []
    cx = cy = None
    state: str | None = None
    focused = False
    label: list[str] = []
    for token in tokens[start:end]:
        if token == SEP:
            if cx is not None and cy is not None:
                out.append(Element(cx, cy, "".join(label), state, focused))
            cx = cy = None
            state, focused, label = None, False, []
        elif token.startswith("<x:"):
            cx = int(token[3:-1])
        elif token.startswith("<y:"):
            cy = int(token[3:-1])
        elif token == "<s:focus>":
            focused = True
        elif token.startswith("<s:"):
            state = token[3:-1]
        elif token.startswith("<c:"):
            label.append(token[3:-1])
    return out


def context_instruction(example: Example, *, vocab: Vocabulary = VOCAB) -> str:
    """The instruction span of an example, as tokenized.

    Decoded from the ids rather than re-derived from `example.task`, for the
    same reason `context_elements` is: the corpus config that truncated it is
    not carried on the example, so re-deriving it here would silently apply a
    different budget than the one the model was trained under — and the
    resulting ceiling would be a ceiling for nobody.
    """
    tokens = vocab.decode(example.ids[: example.prompt_length])
    end = tokens.index(OBS) if OBS in tokens else len(tokens)
    return "".join(t[3:-1] for t in tokens[:end] if t.startswith("<c:"))


# --------------------------------------------------------------------------- #
# The ceiling
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Ceiling:
    """What share of the answers are present in their own context."""

    per_kind: dict[str, tuple[int, int]] = field(default_factory=dict)
    #: Why an answer was not reachable, counted. Empty when all of them were.
    reasons: dict[str, int] = field(default_factory=dict)

    @property
    def reachable(self) -> int:
        return sum(hit for hit, _ in self.per_kind.values())

    @property
    def total(self) -> int:
        return sum(seen for _, seen in self.per_kind.values())

    @property
    def rate(self) -> float:
        return self.reachable / self.total if self.total else 0.0


def _reaches(example: Example) -> tuple[bool, str]:
    """Whether this example's answer can be copied, and if not, what is missing."""
    action = example.action
    if action is None:
        return False, "no gold action"

    if action.text:
        # A `type` is reachable when its string appears verbatim in the
        # instruction the model was given — not the original instruction, the
        # truncated uppercased one, since that is what is in the context.
        if action.text.upper() in context_instruction(example):
            return True, ""
        return False, "typed text not in instruction"

    if action.scroll_direction is not None:
        # A scroll is not a pointer operation. It carries a direction and an
        # amount from small closed sets, and an anchor that is a fixed point
        # rather than a control — so its target cell is legitimately absent
        # from the observation and checking for it there measures nothing.
        # Reachable by memorization, which is a different claim, so it is
        # counted apart rather than folded into the pointer ceiling.
        return True, ""

    if action.coordinate is None:
        return False, "no coordinate and no text"

    want = quantize(*action.coordinate)
    cells = {(e.cx, e.cy) for e in context_elements(example)}
    if want in cells:
        return True, ""
    if not cells:
        return False, "no controls perceived"
    return False, "target cell absent from observation"


def ceiling(examples: Sequence[Example]) -> Ceiling:
    """How many of these decisions a perfect pointer could get right.

    This is an upper bound on any model that works the way this encoding
    intends, and it is not 100%: perception can miss a control, the element
    limit can truncate one away, and a click the search placed between two
    cells can land on a cell no element reports.
    """
    per_kind: dict[str, list[int]] = {}
    reasons: dict[str, int] = {}
    for example in examples:
        if example.action is None:
            continue
        ok, why = _reaches(example)
        if not ok:
            reasons[why] = reasons.get(why, 0) + 1
        for key in (example.action.kind.value, "all"):
            slot = per_kind.setdefault(key, [0, 0])
            slot[0] += int(ok)
            slot[1] += 1
    return Ceiling(
        per_kind={k: (v[0], v[1]) for k, v in per_kind.items()},
        reasons=reasons,
    )


# --------------------------------------------------------------------------- #
# The floor
# --------------------------------------------------------------------------- #


def _words(text: str) -> set[str]:
    return {w.strip('".,') for w in text.split() if w.strip('".,')}


def _overlap(label: str, instruction: str) -> int:
    """How much of a control's label the instruction names."""
    shared = _words(label) & _words(instruction)
    return sum(len(w) for w in shared)


@dataclass(frozen=True)
class Baseline:
    """A policy that does not learn, scored on the same decisions."""

    name: str
    correct: int
    total: int
    note: str = ""

    @property
    def rate(self) -> float:
        return self.correct / self.total if self.total else 0.0


def _click_hits(example: Example, chosen: Element | None) -> bool:
    action = example.action
    if action is None or action.coordinate is None or chosen is None:
        return False
    return quantize(*action.coordinate) == (chosen.cx, chosen.cy)


def baselines(examples: Sequence[Example]) -> list[Baseline]:
    """Score the rules that require no training on the click decisions.

    Restricted to clicks. A `type` baseline would be a regex over the quoted
    span, which measures the generator's phrasing rather than the difficulty
    of the decision, and a `scroll` baseline is a coin flip over two
    directions — neither tells you anything about a model.

    The uniform rate is the one to read first. It is not a policy anybody would
    ship; it is the number a learned accuracy has to be compared against, and
    it is much higher than zero.
    """
    clicks = [
        e for e in examples
        if e.action is not None
        and e.action.coordinate is not None
        and e.action.scroll_direction is None
        and not e.action.text
    ]

    first = overlap = 0
    expected = 0.0
    for example in clicks:
        elements = context_elements(example)
        if not elements:
            continue
        first += _click_hits(example, elements[0])
        instruction = context_instruction(example)
        best = max(elements, key=lambda e: _overlap(e.label, instruction))
        overlap += _click_hits(example, best)
        # Uniform choice is scored in expectation rather than sampled: the
        # answer is exact, and a sampled rate would need a seed and a rerun
        # policy to be reproducible.
        matches = sum(1 for e in elements if _click_hits(example, e))
        expected += matches / len(elements)

    total = len(clicks)
    return [
        Baseline("uniform over controls", round(expected), total,
                 "in expectation, not sampled"),
        Baseline("always the first control", first, total),
        Baseline("label overlapping the instruction", overlap, total,
                 "the bar a learned policy has to clear"),
    ]


# --------------------------------------------------------------------------- #
# The decomposition
# --------------------------------------------------------------------------- #


def predictions(
    model: GPT, examples: Sequence[Example], *, vocab: Vocabulary = VOCAB
) -> list[tuple[Example, Action | None]]:
    """Run the model once over the examples, teacher-forced on the screen.

    Every reading below is a function of this list, so the expensive part
    happens once. `None` means the generation did not decode to an action at
    all, which is itself a failure mode worth counting separately from a wrong
    one.
    """
    stop = vocab.id(EOS)
    out: list[tuple[Example, Action | None]] = []
    for example in examples:
        if example.action is None:
            continue
        produced = generate(
            model, example.ids[: example.prompt_length], max_new=24, stop=(stop,)
        )
        out.append((example, decode_generated(produced, vocab=vocab)))
    return out


def _lcs(left: str, right: str) -> int:
    """Length of the longest common subsequence. Strings here are under 20."""
    previous = [0] * (len(right) + 1)
    for a in left:
        current = [0]
        for j, b in enumerate(right):
            current.append(previous[j] + 1 if a == b else max(previous[j + 1], current[j]))
        previous = current
    return previous[-1]


@dataclass(frozen=True)
class FieldScores:
    """A prediction split into the parts that can independently be wrong."""

    #: (right, seen) for each part.
    kind: tuple[int, int] = (0, 0)
    column: tuple[int, int] = (0, 0)
    row: tuple[int, int] = (0, 0)
    point: tuple[int, int] = (0, 0)
    #: Characters of a `type` recovered, and strings recovered whole.
    characters: tuple[int, int] = (0, 0)
    text_exact: tuple[int, int] = (0, 0)
    #: Undecodable generations — not a wrong action, no action at all.
    malformed: int = 0
    #: Cell distance of every click that missed, so near misses are visible.
    miss_distance: tuple[int, ...] = ()

    @property
    def median_miss(self) -> float:
        if not self.miss_distance:
            return 0.0
        ordered = sorted(self.miss_distance)
        middle = len(ordered) // 2
        if len(ordered) % 2:
            return float(ordered[middle])
        return (ordered[middle - 1] + ordered[middle]) / 2


def field_scores(
    pairs: Sequence[tuple[Example, Action | None]],
) -> FieldScores:
    """Break the predictions apart along the axes that can fail on their own.

    The columns are scored *conditionally on the kind being right*, because a
    coordinate emitted for the wrong verb is not a near miss, it is a different
    action that happens to contain numbers.
    """
    kind_right = kind_seen = 0
    col_right = row_right = point_right = point_seen = 0
    chars_right = chars_seen = 0
    exact_right = exact_seen = 0
    malformed = 0
    misses: list[int] = []

    for example, predicted in pairs:
        gold = example.action
        assert gold is not None  # predictions() skips examples without one
        kind_seen += 1
        if predicted is None:
            malformed += 1
            continue
        if predicted.kind is not gold.kind:
            continue
        kind_right += 1

        if gold.coordinate is not None:
            point_seen += 1
            want = quantize(*gold.coordinate)
            if predicted.coordinate is None:
                misses.append(-1)  # right verb, no point at all
            else:
                got = quantize(*predicted.coordinate)
                col_right += got[0] == want[0]
                row_right += got[1] == want[1]
                if got == want:
                    point_right += 1
                else:
                    misses.append(abs(got[0] - want[0]) + abs(got[1] - want[1]))

        if gold.text:
            exact_seen += 1
            chars_seen += len(gold.text)
            got_text = predicted.text or ""
            chars_right += _lcs(gold.text.upper(), got_text.upper())
            exact_right += got_text == gold.text

    return FieldScores(
        kind=(kind_right, kind_seen),
        column=(col_right, point_seen),
        row=(row_right, point_seen),
        point=(point_right, point_seen),
        characters=(chars_right, chars_seen),
        text_exact=(exact_right, exact_seen),
        malformed=malformed,
        miss_distance=tuple(misses),
    )


# --------------------------------------------------------------------------- #
# Where the model looked
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Attention:
    """How much of the model's attention landed on the control it needed.

    The encoding's whole premise is that emitting a coordinate is a copy: find
    the label the instruction names, take the cell beside it. If that is what
    the model does, then at the moment it emits the coordinate its attention
    should concentrate on the target control's tokens. If it is doing something
    else — reproducing a position it memorized, or following the instruction
    without consulting the screen — the mass goes elsewhere, and that is
    visible here whether or not the accuracy moved.
    """

    #: Mass on the target control's span, summed over layers and heads then
    #: averaged over examples.
    on_target: float = 0.0
    #: What the same measurement would give if attention were spread evenly
    #: over the observation. The number `on_target` has to beat to mean
    #: anything — the target's share of the span is not small.
    if_uniform: float = 0.0
    examples: int = 0

    @property
    def ratio(self) -> float:
        """Above one, the model is looking at the right control."""
        return self.on_target / self.if_uniform if self.if_uniform else 0.0


def _element_spans(
    tokens: Sequence[str], start: int, end: int
) -> list[tuple[int, int]]:
    """Where each control's tokens begin and end, in `tokens[start:end]`."""
    spans: list[tuple[int, int]] = []
    left = start
    for index in range(start, end):
        if tokens[index] == SEP:
            spans.append((left, index))
            left = index + 1
    return spans


def attention_to_target(
    model: GPT, examples: Sequence[Example], *, vocab: Vocabulary = VOCAB
) -> Attention:
    """At the position that emits the coordinate, what was the model reading?

    Teacher-forced on the gold action, so this asks where attention goes when
    the model is about to produce the right answer — not where it went while
    producing a wrong one, which would confound looking in the wrong place with
    being in the wrong state.
    """
    on_target = uniform = 0.0
    counted = 0

    for example in examples:
        gold = example.action
        if gold is None or gold.coordinate is None or gold.scroll_direction:
            continue
        tokens = vocab.decode(example.ids)
        if OBS not in tokens or ACT not in tokens:
            continue
        obs, act = tokens.index(OBS) + 1, tokens.index(ACT)
        spans = _element_spans(tokens, obs, act)
        if len(spans) < 2:
            continue  # nothing to discriminate between

        want = quantize(*gold.coordinate)
        target = [
            (a, b) for a, b in spans
            if f"<x:{want[0]}>" in tokens[a:b] and f"<y:{want[1]}>" in tokens[a:b]
        ]
        if len(target) != 1:
            continue  # ambiguous or absent; the ceiling already counts these

        # The coordinate is emitted one step after `<act>` names the kind.
        query = min(act + 1, len(example.ids) - 1)
        record: list[Tensor] = []
        model.hidden(example.ids[: query + 1], record)

        width = len(example.ids[: query + 1])
        row = [w.data[query * width : (query + 1) * width] for w in record]
        first, last = target[0]
        on_target += sum(sum(r[first:last]) for r in row) / len(row)
        uniform += (last - first) / max(act - obs, 1)
        counted += 1

    if not counted:
        return Attention()
    return Attention(on_target / counted, uniform / counted, counted)


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def _pct(hit: int, seen: int) -> str:
    return f"{hit:>4}/{seen:<4} {hit / seen * 100:5.1f}%" if seen else "     —"


def format_diagnosis(
    limit: Ceiling,
    floor: Sequence[Baseline],
    scores: FieldScores,
    gaze: Attention | None = None,
) -> str:
    """The three readings as one block, ceiling first and score last."""
    lines = ["", "  ceiling — answers present in their own context"]
    for kind, (hit, seen) in sorted(limit.per_kind.items()):
        lines.append(f"    {kind:<24}{_pct(hit, seen)}")
    for why, count in sorted(limit.reasons.items(), key=lambda kv: -kv[1]):
        lines.append(f"      unreachable: {why} x{count}")

    lines += ["", "  floor — policies that do not learn (clicks only)"]
    for base in floor:
        note = f"   {base.note}" if base.note else ""
        lines.append(f"    {base.name:<24}{_pct(base.correct, base.total)}{note}")

    lines += ["", "  the model, by field"]
    lines.append(f"    {'action kind':<24}{_pct(*scores.kind)}")
    lines.append(f"    {'column | kind':<24}{_pct(*scores.column)}")
    lines.append(f"    {'row | kind':<24}{_pct(*scores.row)}")
    lines.append(f"    {'both | kind':<24}{_pct(*scores.point)}")
    lines.append(f"    {'typed characters':<24}{_pct(*scores.characters)}")
    lines.append(f"    {'typed strings whole':<24}{_pct(*scores.text_exact)}")
    if scores.malformed:
        lines.append(f"    {'undecodable':<24}{scores.malformed:>4}")
    if scores.miss_distance:
        near = sum(1 for d in scores.miss_distance if 0 < d <= 2)
        lines.append(
            f"    missed clicks: median {scores.median_miss:.0f} cells away, "
            f"{near} within two"
        )

    if gaze is not None and gaze.examples:
        lines += ["", "  where it looked when emitting a coordinate"]
        lines.append(f"    {'on the target control':<24}{gaze.on_target:6.3f}")
        lines.append(f"    {'if spread evenly':<24}{gaze.if_uniform:6.3f}")
        lines.append(
            f"    {'ratio':<24}{gaze.ratio:6.2f}   "
            f"{'above 1 is looking at the right control' if gaze.ratio > 1 else 'at or below 1 is not'}"
            f"  ({gaze.examples} decisions)"
        )
    return "\n".join(lines)
