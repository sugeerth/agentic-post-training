"""Looking at a control's caption instead of being told what it says.

`perception` recovers a label by matching the renderer's 3x5 font glyph by
glyph, and hands the policy a string. That is a large hand-written prior
sitting between the screen and the model, and it makes the package's central
claim — that the policy reads pixels — quietly weaker than it sounds: the
policy reads *the parser's opinion of* the pixels, and the hardest part of
seeing has already been done for it in Python.

This module removes that step for the part where it matters most. A control's
caption region is cropped from the raw raster, resampled to a fixed small
grid, and handed to the model as numbers. Nothing decodes it into characters.
For the model to ground an instruction now it has to associate the *shape* of
rendered text with the instruction's own character tokens, which is the thing
a vision-language model is supposed to do and the thing this package had so
far assumed.

**The resolution is the experiment.** A caption is drawn at 3x5 per glyph and
a label runs to a dozen glyphs, so the whole caption is roughly 50x9 pixels of
actual signal in a box several times that size. Resampling to 24x6 keeps
stroke-level structure while costing one embedding instead of the twelve
character tokens a decoded label spends — so the visual encoding is *cheaper*
in context than the textual one it replaces, and any loss is a statement
about legibility rather than about budget.

**Grayscale, not colour.** The renderer draws captions in one ink on one
surface; the colour carries state (`filled`, `focused`) which the observation
already encodes as its own symbols. Spending three channels to re-encode what
two flags already say would triple the projection for nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

from computer_use.perception import Box, Element, Raster

__all__ = ["PATCH_HEIGHT", "PATCH_WIDTH", "VISUAL_DIM", "caption_patch", "region_of"]

#: The grid a caption is resampled onto. Wide and short because that is the
#: shape of a line of text; a square patch would spend most of its cells on
#: the empty space above and below the glyphs.
PATCH_WIDTH, PATCH_HEIGHT = 24, 6
VISUAL_DIM = PATCH_WIDTH * PATCH_HEIGHT


@dataclass(frozen=True)
class _Region:
    x: int
    y: int
    width: int
    height: int


def region_of(element: Element) -> _Region | None:
    """Where this control's caption is drawn.

    The text run when the parser found one, and the control's own box when it
    did not — a button whose caption could not be read still has pixels worth
    looking at, and skipping it would hand the visual encoding a hole exactly
    where the textual one has its worst failures.
    """
    run = element.text_run
    if run is not None and run.width > 0 and run.height > 0:
        return _Region(run.x, run.y, run.width, run.height)
    box: Box | None = element.box
    if box is not None and box.width > 0 and box.height > 0:
        return _Region(box.x, box.y, box.width, box.height)
    return None


def caption_patch(raster: Raster, element: Element) -> list[float]:
    """A control's caption as `VISUAL_DIM` numbers in [-1, 1].

    Nearest-neighbour resampling, deliberately. Area-averaging a 3x5 glyph
    down to a few cells greys the strokes into the background and destroys
    exactly the structure the model would have to read; sampling keeps a hard
    edge where there was one. The cost is aliasing, which is the right trade
    when the signal *is* the edges.

    Ink is positive. The renderer draws dark text on light surfaces, so the
    raw values run the other way, and a representation where "there is a
    stroke here" is the large number rather than the small one is the one that
    survives a zero-initialised projection.
    """
    region = region_of(element)
    if region is None:
        return [0.0] * VISUAL_DIM

    out: list[float] = []
    for row in range(PATCH_HEIGHT):
        sy = region.y + (row * region.height) // PATCH_HEIGHT
        sy = min(max(sy, 0), raster.height - 1)
        for col in range(PATCH_WIDTH):
            sx = region.x + (col * region.width) // PATCH_WIDTH
            sx = min(max(sx, 0), raster.width - 1)
            r, g, b = raster.at(sx, sy)
            # Rec. 601 luma, which weights green the way the eye does. The
            # renderer's palette is near-neutral so a flat mean would do, but
            # a browser screenshot's would not, and this costs nothing.
            luma = (299 * r + 587 * g + 114 * b) / 255000.0
            out.append(1.0 - 2.0 * luma)
    return out
