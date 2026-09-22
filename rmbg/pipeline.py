"""The render pipeline, shared by the live preview and the export.

One invariant holds this file together: **the preview and the export run the same code
path**. The preview just runs it on a downscaled proxy. Anything added here has to work
identically at both scales, which is what `scale` is for.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageColor, ImageEnhance, ImageFilter, ImageOps

from . import mask as maskops
from .imaging import encode, load_image, trim_transparent, upscale
from .params import Params


@dataclass
class Render:
    """A rendered frame, plus what is needed to interpret clicks on it.

    `crop` is the trim box in the coordinates of the image that was passed in, or None
    when nothing was trimmed. `alpha` is the finished alpha channel at `img`'s size —
    which is what the Mask view displays and what `--mask` writes.
    """

    img: Image.Image
    crop: tuple[int, int, int, int] | None = None
    alpha: Image.Image | None = None


class BackdropCache:
    """Keeps the chosen backdrop image loaded, plus its last cover-fit result."""

    def __init__(self) -> None:
        self._path: str | None = None
        self._src: Image.Image | None = None
        self._key: tuple | None = None
        self._out: Image.Image | None = None

    def reset(self) -> None:
        self._path = self._key = self._out = self._src = None

    def cover(self, path: str, size: tuple[int, int]) -> Image.Image:
        if path != self._path:
            self._src = load_image(path)
            self._path = path
            self._key = None
        if self._key != (path, size):
            self._out = ImageOps.fit(self._src, size, Image.LANCZOS)
            self._key = (path, size)
        return self._out


def _adjust_colour(rgb: Image.Image, p: Params) -> Image.Image:
    if abs(p.brightness - 1) > 1e-3:
        rgb = ImageEnhance.Brightness(rgb).enhance(p.brightness)
    if abs(p.contrast - 1) > 1e-3:
        rgb = ImageEnhance.Contrast(rgb).enhance(p.contrast)
    if abs(p.saturation - 1) > 1e-3:
        rgb = ImageEnhance.Color(rgb).enhance(p.saturation)
    return rgb


def refine(rgb: Image.Image, alpha: Image.Image, p: Params, scale: float = 1.0
           ) -> tuple[Image.Image, Image.Image]:
    """Colour adjustments, then edge shrink/dilate and feather on the alpha.

    `scale` maps pixel values onto a preview proxy: lengths scale linearly, so a 4 px
    shrink looks the same in the preview as it will in the export.
    """
    rgb = _adjust_colour(rgb, p)
    steps = int(round(p.shrink * scale))
    if steps:
        # Negative shrink grows the subject, which is how you cover a fringe of original
        # background that survived on the outside of the edge.
        filt = ImageFilter.MaxFilter(3) if steps < 0 else ImageFilter.MinFilter(3)
        for _ in range(min(abs(steps), 40)):
            alpha = alpha.filter(filt)
    feather = p.feather * scale
    if feather > 0.05:
        alpha = alpha.filter(ImageFilter.GaussianBlur(feather))
    return rgb, alpha


def defringe(rgb: Image.Image, alpha: Image.Image, radius: int = 1) -> Image.Image:
    """Recolour partly transparent pixels from nearby opaque ones.

    A model that segments slightly *inside* the subject leaves a rim of original background
    colour sitting under a soft alpha edge; that rim is what reads as a halo when the
    subject is placed on a new backdrop. Rebuilding the colour of those pixels out of the
    opaque neighbourhood removes it without eating into the edge.
    """
    radius = max(1, int(radius))
    a = np.asarray(alpha, dtype=np.float32) / 255.0
    if a.size == 0 or a.min() > 0.95 or a.max() < 0.95:
        return rgb
    opaque = a > 0.95
    arr = np.asarray(rgb, dtype=np.float32)

    weight = Image.fromarray((opaque * 255).astype(np.uint8), "L").filter(
        ImageFilter.GaussianBlur(radius))
    wf = np.asarray(weight, dtype=np.float32) / 255.0
    masked = Image.fromarray(np.rint(arr * opaque[..., None]).astype(np.uint8), "RGB")
    blurred = np.asarray(masked.filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32)
    # Divide out the weight, or the estimate darkens toward the transparent side.
    estimate = np.clip(blurred / np.maximum(wf, 1e-4)[..., None], 0, 255)

    mix = np.clip(1.0 - a, 0.0, 1.0)[..., None]
    out = arr * (1.0 - mix) + estimate * mix
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")


def compose(img: Image.Image, p: Params, cache: BackdropCache) -> Image.Image:
    """Put a backdrop behind an image.

    The source is converted rather than required to be RGBA: an image with no cutout yet
    is plain RGB, and a backdrop colour must still work on it.
    """
    if p.bg_mode == "color":
        base = Image.new("RGBA", img.size, ImageColor.getrgb(p.bg_color) + (255,))
    elif p.bg_mode == "image" and p.bg_image:
        base = cache.cover(p.bg_image, img.size).copy()
    else:
        return img
    base.alpha_composite(img.convert("RGBA"))
    return base


def render(rgb: Image.Image, alpha: Image.Image | None, p: Params, *, scale: float = 1.0,
           cache: BackdropCache | None = None) -> Render:
    """Run the whole pipeline once. `alpha=None` means there is no cutout yet.

    Order matters and is the same everywhere:

        curve -> despeckle -> colour, shrink, feather -> defringe -> trim -> backdrop

    The curve comes first so despeckle judges a mask the user has already cleaned up;
    trimming comes last so the crop reflects the finished alpha.

    Every length scales by `scale` and every *area* by `scale ** 2`. An area that scales
    linearly would make the preview remove islands the export keeps — the preview would
    quietly lie about the one thing this app is for.
    """
    cache = cache if cache is not None else BackdropCache()
    if alpha is None:
        return Render(compose(_adjust_colour(rgb, p), p, cache))

    a = maskops.apply_curve(alpha, p.mask_threshold, p.mask_hardness)
    min_area = int(round(p.despeckle * scale * scale))
    fill_area = int(round(p.fill_holes * scale * scale))
    if min_area > 0 or fill_area > 0:
        a = maskops.despeckle(a, min_area, fill_area)

    rgb, a = refine(rgb, a, p, scale)

    if p.defringe > 0:
        rgb = defringe(rgb, a, max(1, int(round(p.defringe * scale))))

    out = rgb.convert("RGBA")
    out.putalpha(a)
    crop = None
    if p.trim:
        out, crop = trim_transparent(out)
    return Render(compose(out, p, cache), crop, out.getchannel("A"))


@dataclass
class Produced:
    """The serialised output, plus the alpha when it was asked for."""

    data: bytes
    size: tuple[int, int]
    alpha: Image.Image | None = None


def produce(rgb: Image.Image, alpha: Image.Image | None, p: Params, *,
            cache: BackdropCache | None = None, with_alpha: bool = False) -> Produced:
    """Full-resolution pipeline: render -> upscale -> encode.

    Rendering once and reusing the result is why `alpha` can come back for free — the
    caller asking for the mask does not pay for a second pass.
    """
    r = render(rgb, alpha, p, cache=cache)
    out = upscale(r.img, p.upscale)
    mask_out = None
    if with_alpha and r.alpha is not None:
        mask_out = r.alpha
        if p.upscale > 1:
            mask_out = mask_out.resize(out.size, Image.LANCZOS)
    return Produced(encode(out, p.fmt, p.quality), out.size, mask_out)


def make_output(rgb: Image.Image, alpha: Image.Image | None, p: Params, *,
                cache: BackdropCache | None = None) -> tuple[bytes, tuple[int, int]]:
    """Serialise the finished image. Returns `(bytes, pixel size)`."""
    r = produce(rgb, alpha, p, cache=cache)
    return r.data, r.size


def export_image(rgb: Image.Image, alpha: Image.Image | None, p: Params,
                 dest: str | Path, *, cache: BackdropCache | None = None) -> tuple[int, int]:
    """Write the finished image to `dest`. Returns the output pixel size."""
    r = produce(rgb, alpha, p, cache=cache)
    Path(dest).write_bytes(r.data)
    return r.size
