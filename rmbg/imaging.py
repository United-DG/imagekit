"""Loading, cropping, resizing and encoding images."""
from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from .config import MAX_OUTPUT_PIXELS, TRIM_THRESHOLD

# Allow big photos, but still block decompression bombs.
Image.MAX_IMAGE_PIXELS = 200_000_000


def load_image(src) -> Image.Image:
    """Open a path or file-like as RGBA, honouring EXIF orientation."""
    with Image.open(src) as im:
        im.load()
        im = ImageOps.exif_transpose(im)
        return im.convert("RGBA")


def flatten_to_rgb(img: Image.Image, bg: tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    """Composite an image onto a solid colour and return RGB."""
    if img.mode == "RGB":
        return img
    rgba = img if img.mode == "RGBA" else img.convert("RGBA")
    if rgba.getchannel("A").getextrema() == (255, 255):
        return rgba.convert("RGB")           # nothing is transparent; skip the paste
    base = Image.new("RGB", rgba.size, bg)
    base.paste(rgba, mask=rgba.getchannel("A"))
    return base


def is_fully_transparent(img: Image.Image) -> bool:
    """True when there is no subject at all, so the model has nothing to find."""
    if img.mode not in ("RGBA", "LA"):
        return False
    return img.getchannel("A").getextrema() == (0, 0)


def trim_transparent(img: Image.Image, threshold: int = TRIM_THRESHOLD
                     ) -> tuple[Image.Image, tuple[int, int, int, int] | None]:
    """Crop away a transparent border.

    Returns `(image, box)`. The box is None when nothing was cropped, and is otherwise
    the region the image was cropped to — the GUI needs it to map a canvas click back to
    a source pixel, since a trimmed preview no longer lines up with the full image.
    """
    mask = img.getchannel("A").point(lambda v: 255 if v > threshold else 0)
    box = mask.getbbox()
    if not box or box == (0, 0, img.width, img.height):
        return img, None
    return img.crop(box), box


def upscale(img: Image.Image, factor: int) -> Image.Image:
    """Lanczos resize plus a light unsharp mask. Adds pixels, not new detail."""
    if factor <= 1:
        return img
    w, h = img.size
    if w * h * factor * factor > MAX_OUTPUT_PIXELS:
        raise ValueError(
            f"{w * factor} x {h * factor} px is too large. Try a smaller upscale factor.")
    # PIL's RGBA resize is premultiplied, so this does not drag dark fringes outward.
    big = img.resize((w * factor, h * factor), Image.LANCZOS)
    r, g, b, a = big.split()
    rgb = Image.merge("RGB", (r, g, b)).filter(
        ImageFilter.UnsharpMask(radius=1.4, percent=70, threshold=2))
    out = rgb.convert("RGBA")
    out.putalpha(a)
    return out


def encode(img: Image.Image, fmt: str, quality: int = 95) -> bytes:
    """Serialise to bytes. Unknown formats raise rather than silently guessing."""
    buf = io.BytesIO()
    fmt = fmt.upper()
    if fmt == "PNG":
        img.save(buf, "PNG")
    elif fmt == "WEBP":
        img.save(buf, "WEBP", quality=int(quality), method=4)
    elif fmt in ("JPG", "JPEG"):
        flatten_to_rgb(img).save(buf, "JPEG", quality=int(quality), optimize=True)
    else:
        raise ValueError(f"Unsupported format: {fmt}")
    return buf.getvalue()


def checkerboard(w: int, h: int, c1, c2, size: int = 12) -> Image.Image:
    """The grey/white grid shown behind transparent pixels."""
    w, h = max(1, w), max(1, h)
    yy, xx = np.indices((h, w))
    m = ((xx // size + yy // size) % 2) == 0
    arr = np.where(m[..., None], np.array(c1, np.uint8), np.array(c2, np.uint8)).astype(np.uint8)
    return Image.fromarray(arr)


def mask_preview(alpha: Image.Image) -> Image.Image:
    """Show an alpha channel as a greyscale RGBA image, for the Mask view."""
    grey = alpha.convert("L")
    out = Image.merge("RGBA", (grey, grey, grey, Image.new("L", grey.size, 255)))
    return out
