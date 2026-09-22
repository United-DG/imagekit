"""Selections: the mask behind every marquee, lasso and wand.

A selection is an 8-bit mask over the whole frame — 0 outside, 255 inside — rather than a
list of shapes. Everything that consumes one (Delete, the brush clipped to it, the on-screen
overlay, the wand's own add/subtract) wants to ask "is *this* pixel in?", and nothing wants
to ask "which rectangle is that?", so a shape is rendered down to pixels once, where the
tool is, instead of being carried around and re-answered at every use.

One invariant makes the rest cheap: **a mask is never modified in place.** Every operation
assigns a fresh image, so a mask handed to anything else is a snapshot of the selection at
that moment. That is what lets an undo action hold a selection by reference — a pointer
rather than a 12 megabyte copy — and still replay exactly.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageChops, ImageDraw

try:
    from scipy import ndimage as _ndimage
    HAVE_SCIPY = True
    SCIPY_WHY = ""
except Exception as _exc:                          # pragma: no cover - scipy is installed here
    _ndimage = None
    HAVE_SCIPY = False
    SCIPY_WHY = f"scipy is not installed ({_exc})"

# How a new shape joins what is already selected. Shift, Alt and Shift+Alt in Photoshop.
REPLACE = "replace"
ADD = "add"
SUBTRACT = "subtract"
INTERSECT = "intersect"


def _combine(base: Image.Image, other: Image.Image, mode: str) -> Image.Image:
    """Merge two binary masks. Always a new image — see the module docstring."""
    if mode == ADD:
        return ImageChops.lighter(base, other)
    if mode == SUBTRACT:
        return ImageChops.subtract(base, other)
    if mode == INTERSECT:
        return ImageChops.darker(base, other)
    return other


class Selection:
    """What is selected right now, at full resolution.

    Deliberately not undoable, unlike the pixels: a selection is a way of pointing at part
    of the image rather than a change to it, and Ctrl+Z is far more valuable for taking back
    an edit than for taking back a marquee. What it *did* to the pixels is on the undo stack
    like any other edit.
    """

    def __init__(self, size: tuple[int, int]) -> None:
        self.size = (max(1, int(size[0])), max(1, int(size[1])))
        self.mask = Image.new("L", self.size, 0)

    # ------------------------------------------------------------------ questions
    def is_empty(self) -> bool:
        """True when nothing is selected.

        Every consumer has to ask this, because "nothing selected" and "everything selected"
        mean opposite things and both are cheap to get wrong: an empty selection has to mean
        *no restriction*, not *no pixels*.
        """
        return self.mask.getbbox() is None

    def area(self) -> int:
        """Selected pixels. Counted, not estimated — `getbbox` says where, not how many."""
        return int(np.count_nonzero(np.asarray(self.mask, dtype=np.uint8)))

    def bbox(self) -> tuple[int, int, int, int] | None:
        return self.mask.getbbox()

    def contains(self, x: float, y: float) -> bool:
        xi, yi = int(x), int(y)
        if not (0 <= xi < self.size[0] and 0 <= yi < self.size[1]):
            return False
        return self.mask.getpixel((xi, yi)) > 0

    def describe(self) -> str:
        if self.is_empty():
            return "Nothing selected"
        return f"{self.area():,} px selected"

    # ------------------------------------------------------------------ whole-mask
    def clear(self) -> None:
        self.mask = Image.new("L", self.size, 0)

    def select_all(self) -> None:
        self.mask = Image.new("L", self.size, 255)

    def invert(self) -> None:
        self.mask = ImageChops.invert(self.mask)

    def replace(self, mask: Image.Image) -> None:
        """Adopt a mask outright, resized if it is not the right shape."""
        if mask.size != self.size:
            mask = mask.resize(self.size, Image.NEAREST)
        self.mask = mask.convert("L")

    def resized(self, size: tuple[int, int]) -> Image.Image:
        """The mask at another resolution, for the preview proxy or an overlay.

        NEAREST on purpose: this answers "is this pixel in", and a smooth resample would
        invent a fringe of half-selected pixels that nothing downstream means to honour.
        """
        return self.mask.resize((max(1, size[0]), max(1, size[1])), Image.NEAREST)

    # ------------------------------------------------------------------ shapes
    def rectangle(self, box: tuple[float, float, float, float], mode: str = REPLACE) -> None:
        """A drag's bounding box. Corners may come in any order, as a drag's do."""
        x0, y0, x1, y1 = box
        shape = Image.new("L", self.size, 0)
        ImageDraw.Draw(shape).rectangle(
            (int(round(min(x0, x1))), int(round(min(y0, y1))),
             int(round(max(x0, x1))), int(round(max(y0, y1)))), fill=255)
        self.mask = _combine(self.mask, shape, mode)

    def polygon(self, points, mode: str = REPLACE) -> None:
        """A freehand lasso. Fewer than three points enclose nothing, so nothing happens."""
        pts = [(float(x), float(y)) for x, y in points]
        if len(pts) < 3:
            return
        shape = Image.new("L", self.size, 0)
        ImageDraw.Draw(shape).polygon(pts, fill=255)
        self.mask = _combine(self.mask, shape, mode)

    def wand(self, rgb: Image.Image, xy: tuple[float, float], tolerance: int = 24,
             mode: str = REPLACE) -> bool:
        """Select the contiguous run of colours like the one under (x, y).

        Tolerance is per channel, and it is the largest channel difference that counts, so a
        value of 24 means "no channel of any pixel differs by more than 24". That is stricter
        than a distance and much easier to predict, which matters for a control the user
        tunes by eye.

        Contiguity is decided by labelling connected components across the whole frame and
        then taking the one the seed landed in — the same machinery the mask's own island
        work uses, so this cannot drift away from it, and a click on an outline selects the
        outline rather than the world.

        Returns False when the click landed outside the image.
        """
        if not HAVE_SCIPY:
            return False
        x, y = int(xy[0]), int(xy[1])
        if not (0 <= x < self.size[0] and 0 <= y < self.size[1]):
            return False
        arr = np.asarray(rgb.convert("RGB"), dtype=np.int16)
        seed = arr[y, x]
        tolerance = max(0, int(tolerance))
        near = np.abs(arr - seed).max(axis=2) <= tolerance
        labels, _n = _ndimage.label(near)
        target = labels[y, x]
        # The seed is always within tolerance of itself, so label 0 cannot be what it hit.
        shape = Image.fromarray(np.where(labels == target, np.uint8(255), np.uint8(0)), "L")
        self.mask = _combine(self.mask, shape, mode)
        return True


def combine_mode(shift: bool, alt: bool) -> str:
    """The modifier state Photoshop uses for add / subtract / intersect.

    One place decides it, so the marquee, the lasso and the wand cannot end up disagreeing
    about what holding shift means.
    """
    if shift and alt:
        return INTERSECT
    if shift:
        return ADD
    if alt:
        return SUBTRACT
    return REPLACE
