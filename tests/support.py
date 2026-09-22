"""Fixtures shared by the suite.

Nothing here touches the network or a model: `StubEngine` answers for rembg, so the same
pipeline code that runs in production can be tested in milliseconds.
"""
from __future__ import annotations

from PIL import Image, ImageDraw


class StubEngine:
    """Stands in for rembg. The "subject" is a rectangle inset from the frame edge."""

    def __init__(self, inset: int = 2) -> None:
        self.inset = inset
        self.calls = 0

    def is_loaded(self, model: str) -> bool:
        return True

    def loaded(self) -> list[str]:
        return []

    def session(self, model: str):
        return None

    def cutout(self, img: Image.Image, p) -> Image.Image:
        self.calls += 1
        rgba = img.convert("RGBA")
        alpha = Image.new("L", rgba.size, 0)
        ImageDraw.Draw(alpha).rectangle(
            (self.inset, self.inset,
             rgba.width - 1 - self.inset, rgba.height - 1 - self.inset), fill=255)
        rgba.putalpha(alpha)
        return rgba


def scene(w: int = 80, h: int = 60, box: tuple | None = None, bg=(30, 60, 200),
          fg=(220, 60, 40)) -> tuple[Image.Image, Image.Image]:
    """`(original RGB, cutout RGBA)`.

    The cutout's colour under the transparent area is black, deliberately: a test can then
    tell a restored pixel (bright) from one that was never touched.
    """
    box = box or (20, 15, 60, 45)
    orig = Image.new("RGB", (w, h), bg)
    ImageDraw.Draw(orig).rectangle(box, fill=fg)
    cut = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    cut.paste(orig.crop(box).convert("RGBA"), (box[0], box[1]))
    return orig, cut


def mask_with(size: tuple[int, int], *boxes: tuple) -> Image.Image:
    """A binary mask with one filled rectangle per box."""
    m = Image.new("L", size, 0)
    d = ImageDraw.Draw(m)
    for b in boxes:
        d.rectangle(b, fill=255)
    return m


def flat(size: tuple[int, int], colour=(120, 120, 120)) -> Image.Image:
    return Image.new("RGB", size, colour)


def opaque_pixels(alpha: Image.Image) -> int:
    """How many pixels count as subject. Cheap stand-in for an area measurement."""
    return sum(alpha.point(lambda v: 255 if v > 128 else 0).histogram()[255:])
