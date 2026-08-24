"""Reading a GUI screenshot back out of its pixels.

Everything else in this package produces frames. Nothing consumed them: the
benchmark scored policies that were handed the environment's ground-truth
state, or a hosted vision model that we cannot run offline. So the claim "these
frames are legible" was an assertion, and any policy trained here would have
been training on privileged information rather than on what an agent actually
sees.

This module closes that. It decodes the PNG, recovers the text by matching the
renderer's 3x5 font, and finds the controls by locating their borders — from
the image alone, with no access to the environment. What comes out is a
`Screen`: labelled boxes with a point to click.

Two consequences worth stating plainly:

  It is a mechanical legibility test. If a label cannot be recovered from the
  frame, the frame does not carry it, and no vision model was going to read it
  either. `tests/test_perception.py` runs that check across generated worlds.

  It makes an offline policy honest. `learn.LearnedPolicy` sees this and only
  this, so a score it earns is a score for operating a GUI rather than for
  being handed the answer.

The parser targets frames from this package's renderer: 8-bit RGB, axis-aligned
controls, the bundled font. It is not a general-purpose OCR, and it does not
try to be — a browser screenshot needs `PlaywrightComputer.state_script`, which
is why that exists.
"""

from __future__ import annotations

import struct
import zlib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

from computer_use._render import _GLYPHS, GLYPH_H, GLYPH_W

#: Pattern -> character, for reading the font backwards.
_BY_PATTERN: dict[tuple[str, ...], str] = {
    pattern: char for char, pattern in _GLYPHS.items() if char != " "
}

#: The renderer draws widget text at 3 and the scroll hint at 2.
_SCALES = (3, 2)

_BLANK = ("...",) * GLYPH_H


class DecodeError(ValueError):
    """The bytes were not a PNG this module can read."""


# --------------------------------------------------------------------------- #
# Raster
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Raster:
    """A decoded RGB image, addressable by pixel."""

    width: int
    height: int
    pixels: bytes  # width * height * 3

    def at(self, x: int, y: int) -> tuple[int, int, int]:
        if not (0 <= x < self.width and 0 <= y < self.height):
            return (0, 0, 0)
        off = (y * self.width + x) * 3
        return (self.pixels[off], self.pixels[off + 1], self.pixels[off + 2])

    def positions(self, color: tuple[int, int, int]) -> Iterator[tuple[int, int]]:
        """Every pixel of exactly this color.

        Uses `bytes.find` rather than a Python-level scan: the frames are
        nearly a megapixel and the interesting colors cover a small fraction of
        them, so jumping between matches is the difference between a parser you
        can call inside a rollout loop and one you cannot.
        """
        needle = bytes(color)
        stride = self.width * 3
        start = self.pixels.find(needle)
        while start != -1:
            if start % 3 == 0:  # a match straddling two pixels is not a pixel
                yield ((start % stride) // 3, start // stride)
            start = self.pixels.find(needle, start + 1)

    def count(self, color: tuple[int, int, int]) -> int:
        return sum(1 for _ in self.positions(color))


def decode_png(data: bytes) -> Raster:
    """Decode an 8-bit RGB or RGBA PNG.

    Deliberately narrow: enough for this package's renderer, with every filter
    type implemented because a re-encoded frame may use them even though ours
    does not.
    """
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise DecodeError("not a PNG")

    pos = 8
    header: tuple[int, ...] | None = None
    idat = bytearray()
    while pos + 8 <= len(data):
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        tag = data[pos + 4:pos + 8]
        payload = data[pos + 8:pos + 8 + length]
        pos += 12 + length
        if tag == b"IHDR":
            header = struct.unpack(">IIBBBBB", payload)
        elif tag == b"IDAT":
            idat += payload
        elif tag == b"IEND":
            break

    if header is None:
        raise DecodeError("no IHDR chunk")
    width, height, depth, color_type, compression, filtering, interlace = header
    if depth != 8 or color_type not in (2, 6) or interlace != 0:
        raise DecodeError(
            f"unsupported PNG: depth={depth} color_type={color_type} "
            f"interlace={interlace} (want 8-bit RGB/RGBA, non-interlaced)"
        )
    if compression != 0 or filtering != 0:
        raise DecodeError("unsupported PNG compression or filter method")

    channels = 3 if color_type == 2 else 4
    raw = zlib.decompress(bytes(idat))
    stride = width * channels
    out = bytearray(width * height * 3)
    previous = bytearray(stride)
    offset = 0
    for y in range(height):
        filter_type = raw[offset]
        line = bytearray(raw[offset + 1:offset + 1 + stride])
        offset += 1 + stride
        _unfilter(line, previous, filter_type, channels)
        if channels == 3:
            out[y * width * 3:(y + 1) * width * 3] = line
        else:
            row = out
            base = y * width * 3
            for x in range(width):
                src = x * 4
                row[base + x * 3:base + x * 3 + 3] = line[src:src + 3]
        previous = line

    return Raster(width=width, height=height, pixels=bytes(out))


def _unfilter(line: bytearray, previous: bytearray, filter_type: int, bpp: int) -> None:
    """Reverse one PNG scanline filter, in place."""
    if filter_type == 0:
        return
    if filter_type == 1:  # Sub
        for i in range(bpp, len(line)):
            line[i] = (line[i] + line[i - bpp]) & 0xFF
    elif filter_type == 2:  # Up
        for i in range(len(line)):
            line[i] = (line[i] + previous[i]) & 0xFF
    elif filter_type == 3:  # Average
        for i in range(len(line)):
            left = line[i - bpp] if i >= bpp else 0
            line[i] = (line[i] + ((left + previous[i]) >> 1)) & 0xFF
    elif filter_type == 4:  # Paeth
        for i in range(len(line)):
            left = line[i - bpp] if i >= bpp else 0
            up = previous[i]
            up_left = previous[i - bpp] if i >= bpp else 0
            estimate = left + up - up_left
            da, db, dc = abs(estimate - left), abs(estimate - up), abs(estimate - up_left)
            nearest = left if (da <= db and da <= dc) else (up if db <= dc else up_left)
            line[i] = (line[i] + nearest) & 0xFF
    else:
        raise DecodeError(f"unknown scanline filter {filter_type}")


# --------------------------------------------------------------------------- #
# What the parser finds
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class TextRun:
    """A word or phrase recovered from the pixels, with where it was drawn."""

    text: str
    x: int
    y: int
    width: int
    height: int
    scale: int
    color: tuple[int, int, int]

    @property
    def center(self) -> tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)


@dataclass(frozen=True, slots=True)
class Box:
    """An axis-aligned control outline found in the pixels."""

    x: int
    y: int
    width: int
    height: int
    color: tuple[int, int, int]

    @property
    def center(self) -> tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)

    def contains(self, px: int, py: int) -> bool:
        return self.x <= px < self.x + self.width and self.y <= py < self.y + self.height


@dataclass(frozen=True, slots=True)
class Element:
    """A control the agent could act on: what it says, and where to click it.

    `kind` is a guess from geometry — small square boxes are toggles, wide
    short ones are fields or buttons. The agent gets no more than that, which
    is the same guess a person makes from a screenshot.
    """

    label: str
    kind: str  # "toggle" | "field" | "button" | "text"
    box: Box | None
    text_run: TextRun | None
    click: tuple[int, int]
    #: Painted in something other than a surface color — how a primary action
    #: announces itself, and visible in the pixels rather than looked up.
    filled: bool = False

    @property
    def tokens(self) -> tuple[str, ...]:
        return tuple(t for t in self.label.replace("-", " ").split() if t)


@dataclass(frozen=True, slots=True)
class Screen:
    """Everything the parser could recover from one frame."""

    width: int
    height: int
    title: str
    elements: tuple[Element, ...] = ()
    runs: tuple[TextRun, ...] = ()
    scroll: tuple[int, int] | None = None  # (position, maximum), when shown

    def labelled(self) -> tuple[Element, ...]:
        return tuple(e for e in self.elements if e.label)

    def find(self, phrase: str) -> Element | None:
        """The element whose label matches `phrase` exactly, if any."""
        wanted = phrase.strip().upper()
        for element in self.elements:
            if element.label.upper() == wanted:
                return element
        return None


# --------------------------------------------------------------------------- #
# Text recovery
# --------------------------------------------------------------------------- #


def _ink_colors(raster: Raster, *, stride: int = 5, floor: int = 12) -> list[tuple[int, int, int]]:
    """Colors that plausibly carry marks rather than fill.

    Sampled on a coarse grid — enough to name the palette without a
    million-pixel pass. Backgrounds win by area, so the tail of the histogram
    is the text, the borders, and the indicators.
    """
    counts: dict[tuple[int, int, int], int] = {}
    for y in range(0, raster.height, stride):
        base = y * raster.width * 3
        for x in range(0, raster.width, stride):
            off = base + x * 3
            key = (raster.pixels[off], raster.pixels[off + 1], raster.pixels[off + 2])
            counts[key] = counts.get(key, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    # The dominant few are page, panel, and chrome. Everything else that shows
    # up more than a handful of times is a candidate mark color.
    return [color for color, seen in ranked[3:] if seen >= floor // stride]


def _surfaces(raster: Raster, *, stride: int = 5, keep: int = 3) -> set[tuple[int, int, int]]:
    """The colors the page is made of, as opposed to painted onto.

    Used to tell a primary action from an ordinary one: a filled button is the
    one control that is not sitting on a surface color, which is precisely the
    signal the design is using to draw the eye.
    """
    counts: dict[tuple[int, int, int], int] = {}
    for y in range(0, raster.height, stride):
        base = y * raster.width * 3
        for x in range(0, raster.width, stride):
            off = base + x * 3
            key = (raster.pixels[off], raster.pixels[off + 1], raster.pixels[off + 2])
            counts[key] = counts.get(key, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    return {color for color, _ in ranked[:keep]}


def _pattern_at(ink: frozenset[tuple[int, int]], x: int, y: int, scale: int) -> tuple[str, ...]:
    rows = []
    for row in range(GLYPH_H):
        cells = "".join(
            "#" if (x + col * scale, y + row * scale) in ink else "."
            for col in range(GLYPH_W)
        )
        rows.append(cells)
    return tuple(rows)


def _bands(ys: Sequence[int]) -> list[tuple[int, int]]:
    """Maximal runs of consecutive rows that contain ink."""
    out: list[tuple[int, int]] = []
    start = previous = ys[0]
    for y in ys[1:]:
        if y == previous + 1:
            previous = y
            continue
        out.append((start, previous))
        start = previous = y
    out.append((start, previous))
    return out


def text_runs(raster: Raster, colors: Sequence[tuple[int, int, int]] | None = None) -> list[TextRun]:
    """Recover drawn text by matching the renderer's font against the pixels.

    Works per color: a run of glyphs shares one color, one baseline, and one
    scale, so grouping the ink by color and then by row band leaves a strip
    that can be stepped through a cell at a time.
    """
    found: list[TextRun] = []
    for color in (colors if colors is not None else _ink_colors(raster)):
        by_row: dict[int, set[int]] = {}
        ink: set[tuple[int, int]] = set()
        for x, y in raster.positions(color):
            ink.add((x, y))
            by_row.setdefault(y, set()).add(x)
        if not ink:
            continue
        frozen = frozenset(ink)
        for top, bottom in _bands(sorted(by_row)):
            height = bottom - top + 1
            for scale in _SCALES:
                if height != GLYPH_H * scale:
                    continue
                found.extend(_decode_band(frozen, by_row, top, scale, color))
                break
    found.sort(key=lambda r: (r.y, r.x))
    return found


def _decode_band(
    ink: frozenset[tuple[int, int]],
    by_row: dict[int, set[int]],
    top: int,
    scale: int,
    color: tuple[int, int, int],
) -> list[TextRun]:
    """Read every run of glyphs sitting on one baseline."""
    advance = (GLYPH_W + 1) * scale
    columns = sorted({x for row in range(top, top + GLYPH_H * scale) for x in by_row.get(row, ())})
    if not columns:
        return []

    runs: list[TextRun] = []
    cursor = columns[0]
    limit = columns[-1] + 1
    chars: list[str] = []
    start = cursor
    blanks = 0
    while cursor < limit + advance:
        pattern = _pattern_at(ink, cursor, top, scale)
        if pattern == _BLANK:
            blanks += 1
            # One blank cell is a space inside a phrase; two is the gap between
            # two separate controls that happen to share a baseline.
            if blanks >= 2 or not chars:
                if chars:
                    runs.append(_finish(chars, start, top, scale, color))
                    chars = []
                nxt = [x for x in columns if x >= cursor + advance]
                if not nxt:
                    break
                cursor = nxt[0]
                start = cursor
                blanks = 0
                continue
            chars.append(" ")
            cursor += advance
            continue

        char = _BY_PATTERN.get(pattern)
        if char is None:
            # Not the font — an indicator, a cursor, or a partially covered
            # glyph. End the run here rather than emitting a wrong character.
            if chars:
                runs.append(_finish(chars, start, top, scale, color))
                chars = []
            nxt = [x for x in columns if x >= cursor + scale]
            if not nxt:
                break
            cursor = nxt[0]
            start = cursor
            blanks = 0
            continue

        blanks = 0
        chars.append(char)
        cursor += advance

    if chars:
        runs.append(_finish(chars, start, top, scale, color))
    return runs


def read_region(raster: Raster, box: Box) -> list[TextRun]:
    """Read text inside one control, against that control's own background.

    A primary button is filled with the accent color and captioned in white —
    and white is also the page. Judged against the whole frame, white is
    background and the caption disappears, which is how the one control the
    task is usually about ends up unlabelled. Inside the button's own bounds
    the roles swap: the fill is the background and the white is the mark.
    """
    x0, y0 = box.x + 1, box.y + 1
    x1, y1 = box.x + box.width - 1, box.y + box.height - 1
    counts: dict[tuple[int, int, int], int] = {}
    for y in range(y0, y1):
        for x in range(x0, x1):
            key = raster.at(x, y)
            counts[key] = counts.get(key, 0) + 1
    if not counts:
        return []

    fill = max(counts.items(), key=lambda kv: kv[1])[0]
    runs: list[TextRun] = []
    for color, seen in counts.items():
        if color == fill or seen < GLYPH_W:
            continue
        by_row: dict[int, set[int]] = {}
        ink: set[tuple[int, int]] = set()
        for y in range(y0, y1):
            for x in range(x0, x1):
                if raster.at(x, y) == color:
                    ink.add((x, y))
                    by_row.setdefault(y, set()).add(x)
        if not ink:
            continue
        frozen = frozenset(ink)
        for top, bottom in _bands(sorted(by_row)):
            for scale in _SCALES:
                if bottom - top + 1 != GLYPH_H * scale:
                    continue
                runs.extend(_decode_band(frozen, by_row, top, scale, color))
                break
    runs.sort(key=lambda r: (r.y, r.x))
    return runs


def _finish(
    chars: list[str], x: int, y: int, scale: int, color: tuple[int, int, int]
) -> TextRun:
    # Measure the trimmed run: a trailing blank cell is not part of what was
    # drawn, and counting it shifts the apparent right margin by a whole cell —
    # enough to make centred text look flush left.
    while chars and chars[-1] == " ":
        chars.pop()
    text = "".join(chars).strip()
    width = max(len(chars) * (GLYPH_W + 1) * scale - scale, 0)
    return TextRun(
        text=text, x=x, y=y, width=max(width, 0),
        height=GLYPH_H * scale, scale=scale, color=color,
    )


# --------------------------------------------------------------------------- #
# Control outlines
# --------------------------------------------------------------------------- #


def boxes(raster: Raster, colors: Sequence[tuple[int, int, int]] | None = None,
          *, min_side: int = 16) -> list[Box]:
    """Find axis-aligned rectangles by pairing their horizontal edges.

    A control in this renderer is a stroked rectangle, so its top and bottom
    edges are long runs of one color at the same x-extent. Matching those pairs
    is enough, and it does not care whether the rectangle is filled.
    """
    out: list[Box] = []
    for color in (colors if colors is not None else _ink_colors(raster)):
        rows: dict[int, list[tuple[int, int]]] = {}
        by_row: dict[int, set[int]] = {}
        for x, y in raster.positions(color):
            by_row.setdefault(y, set()).add(x)
        for y, xs in by_row.items():
            rows[y] = _runs(sorted(xs), min_length=min_side)

        # An edge is only an edge if the two verticals under it exist.
        seen: set[tuple[int, int, int, int]] = set()
        for y, spans in sorted(rows.items()):
            for x0, x1 in spans:
                for y2, lower in sorted(rows.items()):
                    if y2 <= y + min_side - 1:
                        continue
                    if (x0, x1) not in lower:
                        continue
                    if not _has_sides(by_row, x0, x1, y, y2):
                        continue
                    key = (x0, y, x1 - x0 + 1, y2 - y + 1)
                    if key not in seen:
                        seen.add(key)
                        out.append(Box(x=x0, y=y, width=key[2], height=key[3], color=color))
                    break
    out.sort(key=lambda b: (b.y, b.x))
    return _dedupe(out)


def _dedupe(found: Sequence[Box], *, slack: int = 4) -> list[Box]:
    """Collapse the outlines a thick stroke produces into one control.

    A 2px border has an outer and an inner edge, and both pair up into a valid
    rectangle. Left alone that is two controls at the same place, which double
    counts every checkbox on the screen.
    """
    kept: list[Box] = []
    for box in sorted(found, key=lambda b: b.width * b.height, reverse=True):
        if any(
            abs(box.x - k.x) <= slack and abs(box.y - k.y) <= slack
            and abs(box.width - k.width) <= 2 * slack
            and abs(box.height - k.height) <= 2 * slack
            for k in kept
        ):
            continue
        kept.append(box)
    kept.sort(key=lambda b: (b.y, b.x))
    return kept


def _drop_nested(found: Sequence[Box], *, slack: int = 4) -> list[Box]:
    """Remove rectangles sitting wholly inside another control.

    Controls in this renderer do not nest, so anything inside one is part of
    it. The case that matters is a filled button: its caption is painted in a
    single colour across several rows, and those rows pair into perfectly good
    rectangles *inside* the button — phantom controls made out of the letters
    of a real one, each one wide enough to be mistaken for a field.

    Runs after the content panel has been dropped. Every control is inside the
    panel, so applying this with the panel still in the list deletes the entire
    screen.
    """
    kept: list[Box] = []
    for box in sorted(found, key=lambda b: b.width * b.height, reverse=True):
        if not any(_within(box, k, slack) for k in kept):
            kept.append(box)
    kept.sort(key=lambda b: (b.y, b.x))
    return kept


def _within(inner: Box, outer: Box, slack: int) -> bool:
    return (
        inner.x >= outer.x - slack
        and inner.y >= outer.y - slack
        and inner.x + inner.width <= outer.x + outer.width + slack
        and inner.y + inner.height <= outer.y + outer.height + slack
    )


def _runs(xs: Sequence[int], *, min_length: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = previous = xs[0] if xs else 0
    for x in xs[1:]:
        if x == previous + 1:
            previous = x
            continue
        if previous - start + 1 >= min_length:
            spans.append((start, previous))
        start = previous = x
    if xs and previous - start + 1 >= min_length:
        spans.append((start, previous))
    return spans


def _has_sides(by_row: dict[int, set[int]], x0: int, x1: int, top: int, bottom: int,
               *, tolerance: float = 0.1) -> bool:
    """Both verticals present down essentially the whole span.

    Sampling a few rows is not enough. Stack three radio buttons and the
    bottom edge of one has the same x-extent as the top edge of the next, so
    they pair into a rectangle that spans the gap between them — a control
    that does not exist, sitting over the label of one that does. Requiring the
    sides to actually run the full height rules that out, with a little slack
    for the corners.
    """
    span = bottom - top + 1
    missing = 0
    budget = int(span * tolerance) + 2
    for y in range(top, bottom + 1):
        row = by_row.get(y, ())
        if x0 not in row or x1 not in row:
            missing += 1
            if missing > budget:
                return False
    return True


# --------------------------------------------------------------------------- #
# Putting a screen together
# --------------------------------------------------------------------------- #

#: A toggle is a small square; a field is wide and short; a button sits between.
_TOGGLE_MAX = 44
_FIELD_MIN_WIDTH = 200


def parse_screen(data: bytes | Raster) -> Screen:
    """Read a frame into labelled, clickable elements.

    Pairing is by position, the way a person reads a form: the caption to the
    right of a small square belongs to that square, and the words inside a
    rectangle name the rectangle.
    """
    raster = data if isinstance(data, Raster) else decode_png(data)
    colors = _ink_colors(raster)
    runs = text_runs(raster, colors)
    outlines = _drop_nested(
        [b for b in boxes(raster, colors) if b.width < raster.width - 60]
    )
    surfaces = _surfaces(raster)

    title = ""
    scroll: tuple[int, int] | None = None
    body: list[TextRun] = []
    for run in runs:
        if run.y < 56:  # window chrome
            if run.scale == 2 and run.text.startswith("SCROLL"):
                scroll = _scroll(run.text)
            elif not title:
                title = run.text
            continue
        body.append(run)

    used: set[int] = set()
    elements: list[Element] = []

    for box in outlines:
        inside = [
            (i, r) for i, r in enumerate(body)
            if i not in used and box.contains(*r.center)
        ]
        right = [
            (i, r) for i, r in enumerate(body)
            if i not in used
            and 0 < r.x - (box.x + box.width) <= 24
            and abs(r.center[1] - box.center[1]) <= box.height
        ]
        if box.width <= _TOGGLE_MAX and box.height <= _TOGGLE_MAX:
            kind, label_runs = "toggle", right
        elif not inside:
            # Nothing readable against the page background does not mean an
            # empty control: a caption painted in a surface colour on a filled
            # button is invisible globally and obvious locally. Read it before
            # falling back to guessing from the width.
            local = read_region(raster, box)
            if local and _centred(box, local):
                kind, label_runs = "button", []
            else:
                kind, label_runs = (
                    "field" if box.width >= _FIELD_MIN_WIDTH else "button"), []
        else:
            # Width alone gets this wrong — a 200px CONTINUE button is exactly
            # as wide as a short input, and calling it a field leaves the agent
            # with no way off the screen. What actually separates them is where
            # the text sits: a caption is centred, a value is flush left at a
            # fixed inset, and that is visible in the pixels.
            kind = "button" if _centred(box, [r for _, r in inside]) else "field"
            label_runs = inside

        for i, _ in label_runs:
            used.add(i)
        label = " ".join(r.text for _, r in sorted(label_runs, key=lambda p: p[1].x))
        caption = label_runs[0][1] if label_runs else None
        if not label and kind == "button":
            local = read_region(raster, box)
            if local:
                label = " ".join(r.text for r in local)
                caption = local[0]
        elements.append(Element(
            label=label, kind=kind, box=box, text_run=caption, click=box.center,
            filled=raster.at(box.x + 3, box.y + 3) not in surfaces,
        ))

    # Text with no box around it is still worth reporting: it is the caption
    # that names the field under it, which is how a form is read.
    for i, run in enumerate(body):
        if i in used:
            continue
        elements.append(Element(
            label=run.text, kind="text", box=None, text_run=run, click=run.center,
        ))

    elements.sort(key=lambda e: (e.click[1], e.click[0]))
    return Screen(
        width=raster.width, height=raster.height, title=title,
        elements=tuple(elements), runs=tuple(runs), scroll=scroll,
    )


def _centred(box: Box, runs: Sequence[TextRun], *, slack: int = 8) -> bool:
    """Whether the text inside a control is centred rather than flush left."""
    if not runs:
        return False
    left = min(r.x for r in runs) - box.x
    right = (box.x + box.width) - max(r.x + r.width for r in runs)
    return abs(left - right) <= slack


def _scroll(text: str) -> tuple[int, int] | None:
    try:
        position, maximum = text.split()[1].split("/")
        return (int(position), int(maximum))
    except (IndexError, ValueError):
        return None


@dataclass(frozen=True, slots=True)
class Legibility:
    """How much of what was drawn could be read back."""

    expected: tuple[str, ...] = ()
    recovered: tuple[str, ...] = ()
    missing: tuple[str, ...] = field(default=())

    @property
    def rate(self) -> float:
        return 1.0 - len(self.missing) / len(self.expected) if self.expected else 1.0


def legibility(frame: bytes, expected: Sequence[str]) -> Legibility:
    """Check a frame against the strings it was supposed to show.

    A frame that fails this is one no vision model could have read either, so
    it belongs in a test rather than in a benchmark result.
    """
    screen = parse_screen(frame)
    # Element labels, not just raw runs: a primary button's caption is white on
    # the accent color, which is only recoverable against the button's own
    # background. Checking runs alone would report the one control the task is
    # usually about as illegible.
    seen = {r.text.upper() for r in screen.runs}
    seen |= {e.label.upper() for e in screen.elements if e.label}
    joined = " ".join(sorted(seen))
    missing = tuple(
        text for text in expected
        if text and text.upper() not in seen and text.upper() not in joined
    )
    return Legibility(
        expected=tuple(expected), recovered=tuple(sorted(seen)), missing=missing,
    )


__all__ = [
    "Box",
    "DecodeError",
    "Element",
    "Legibility",
    "Raster",
    "Screen",
    "TextRun",
    "boxes",
    "decode_png",
    "legibility",
    "parse_screen",
    "text_runs",
]
