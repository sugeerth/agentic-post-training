"""A dependency-free PNG renderer for the mock GUI.

`MockComputer` needs to hand a VLM a real image, and the point of the mock is
that it runs anywhere — CI, a laptop with no display, a notebook without
Pillow. So this module encodes PNG bytes directly (zlib is stdlib) and draws
with a 3x5 bitmap font.

Small enough to be readable, real enough that the frames it produces are
legible to an actual vision model — which means the mock doubles as a smoke
test against the live API, not just a test fixture.
"""

from __future__ import annotations

import struct
import zlib

RGB = tuple[int, int, int]

# 3x5 bitmap font. Only the glyphs the mock UI needs; anything else renders as
# a filled block so missing characters are visible rather than silently blank.
_GLYPHS: dict[str, tuple[str, str, str, str, str]] = {
    "A": (".#.", "#.#", "###", "#.#", "#.#"),
    "B": ("##.", "#.#", "##.", "#.#", "##."),
    "C": (".##", "#..", "#..", "#..", ".##"),
    "D": ("##.", "#.#", "#.#", "#.#", "##."),
    "E": ("###", "#..", "##.", "#..", "###"),
    "F": ("###", "#..", "##.", "#..", "#.."),
    "G": (".##", "#..", "#.#", "#.#", ".##"),
    "H": ("#.#", "#.#", "###", "#.#", "#.#"),
    "I": ("###", ".#.", ".#.", ".#.", "###"),
    "J": ("..#", "..#", "..#", "#.#", ".#."),
    "K": ("#.#", "#.#", "##.", "#.#", "#.#"),
    "L": ("#..", "#..", "#..", "#..", "###"),
    "M": ("#.#", "###", "###", "#.#", "#.#"),
    "N": ("#.#", "###", "###", "###", "#.#"),
    "O": (".#.", "#.#", "#.#", "#.#", ".#."),
    "P": ("##.", "#.#", "##.", "#..", "#.."),
    "Q": (".#.", "#.#", "#.#", "###", ".##"),
    "R": ("##.", "#.#", "##.", "#.#", "#.#"),
    "S": (".##", "#..", ".#.", "..#", "##."),
    "T": ("###", ".#.", ".#.", ".#.", ".#."),
    "U": ("#.#", "#.#", "#.#", "#.#", ".#."),
    "V": ("#.#", "#.#", "#.#", ".#.", ".#."),
    "W": ("#.#", "#.#", "###", "###", "#.#"),
    "X": ("#.#", "#.#", ".#.", "#.#", "#.#"),
    "Y": ("#.#", "#.#", ".#.", ".#.", ".#."),
    "Z": ("###", "..#", ".#.", "#..", "###"),
    "0": ("###", "#.#", "#.#", "#.#", "###"),
    "1": (".#.", "##.", ".#.", ".#.", "###"),
    "2": ("##.", "..#", ".#.", "#..", "###"),
    "3": ("##.", "..#", ".#.", "..#", "##."),
    "4": ("#.#", "#.#", "###", "..#", "..#"),
    "5": ("###", "#..", "##.", "..#", "##."),
    "6": (".##", "#..", "###", "#.#", "###"),
    "7": ("###", "..#", ".#.", ".#.", ".#."),
    "8": ("###", "#.#", "###", "#.#", "###"),
    "9": ("###", "#.#", "###", "..#", "##."),
    " ": ("...", "...", "...", "...", "..."),
    ".": ("...", "...", "...", "...", ".#."),
    ",": ("...", "...", "...", ".#.", "#.."),
    "-": ("...", "...", "###", "...", "..."),
    "_": ("...", "...", "...", "...", "###"),
    ":": ("...", ".#.", "...", ".#.", "..."),
    "/": ("..#", "..#", ".#.", "#..", "#.."),
    "@": (".#.", "#.#", "###", "#..", ".##"),
    "?": ("##.", "..#", ".#.", "...", ".#."),
    "!": (".#.", ".#.", ".#.", "...", ".#."),
    "*": ("#.#", ".#.", "#.#", "...", "..."),
    "+": ("...", ".#.", "###", ".#.", "..."),
    "'": (".#.", ".#.", "...", "...", "..."),
}
_MISSING = ("###", "###", "###", "###", "###")

GLYPH_W, GLYPH_H = 3, 5


class Canvas:
    """A mutable RGB pixel buffer that can encode itself as PNG."""

    __slots__ = ("_px", "height", "width")

    def __init__(self, width: int, height: int, background: RGB = (255, 255, 255)) -> None:
        self.width = width
        self.height = height
        self._px = bytearray(bytes(background) * (width * height))

    # ---- primitives ------------------------------------------------------- #

    def set_pixel(self, x: int, y: int, color: RGB) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            off = (y * self.width + x) * 3
            self._px[off:off + 3] = bytes(color)

    def fill_rect(self, x: int, y: int, w: int, h: int, color: RGB) -> None:
        row = bytes(color) * max(0, min(w, self.width - x))
        if not row:
            return
        for yy in range(max(0, y), min(y + h, self.height)):
            off = (yy * self.width + max(0, x)) * 3
            self._px[off:off + len(row)] = row

    def stroke_rect(self, x: int, y: int, w: int, h: int, color: RGB, weight: int = 1) -> None:
        for i in range(weight):
            self.fill_rect(x + i, y + i, w - 2 * i, 1, color)
            self.fill_rect(x + i, y + h - 1 - i, w - 2 * i, 1, color)
            self.fill_rect(x + i, y + i, 1, h - 2 * i, color)
            self.fill_rect(x + w - 1 - i, y + i, 1, h - 2 * i, color)

    def text(self, x: int, y: int, message: str, color: RGB, scale: int = 2) -> int:
        """Draw `message` with its top-left at (x, y). Returns the width drawn."""
        cursor = x
        advance = (GLYPH_W + 1) * scale
        for char in message.upper():
            glyph = _GLYPHS.get(char, _MISSING)
            for row, bits in enumerate(glyph):
                for col, bit in enumerate(bits):
                    if bit != "#":
                        continue
                    self.fill_rect(
                        cursor + col * scale, y + row * scale, scale, scale, color
                    )
            cursor += advance
        return cursor - x

    # ---- encoding --------------------------------------------------------- #

    def to_png(self) -> bytes:
        """Encode as a PNG. Filter type 0 on every row — simple and adequate."""
        stride = self.width * 3
        raw = bytearray()
        for y in range(self.height):
            raw.append(0)
            raw += self._px[y * stride:(y + 1) * stride]

        def chunk(tag: bytes, payload: bytes) -> bytes:
            body = tag + payload
            return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body))

        header = struct.pack(">IIBBBBB", self.width, self.height, 8, 2, 0, 0, 0)
        return (
            b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", header)
            + chunk(b"IDAT", zlib.compress(bytes(raw), 6))
            + chunk(b"IEND", b"")
        )


def text_width(message: str, scale: int = 2) -> int:
    return len(message) * (GLYPH_W + 1) * scale
