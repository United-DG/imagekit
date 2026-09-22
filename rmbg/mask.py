"""Alpha post-processing — the levers that decide output quality.

Everything here takes a PIL "L" alpha image and returns a new one, so the operations
compose in any order and are trivial to unit-test.

Two failure modes matter, and they need different fixes:

* **Haze.** The model leaves faint, real alpha around the subject. Nothing downstream can
  drop it, because it is not an artefact — it is a value. `apply_curve` is the fix:
  raise the threshold to snap weak alpha to zero.
* **Solid leftovers.** A patch of background the model decided was foreground, sitting
  detached from the subject. `despeckle` removes small ones automatically;
  `remove_island_at` removes a specific one on a click.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

try:
    from scipy import ndimage as _ndimage
    HAVE_SCIPY = True
    SCIPY_WHY = ""
except Exception as _exc:                          # pragma: no cover - scipy is installed here
    _ndimage = None
    HAVE_SCIPY = False
    SCIPY_WHY = f"scipy is not installed ({_exc})"

OPAQUE = 128          # alpha above this counts as "subject" for component work


def _require_scipy() -> None:
    if not HAVE_SCIPY:
        raise RuntimeError(
            "Removing islands and filling holes need scipy. Install it with: pip install scipy")


def apply_curve(alpha: Image.Image, threshold: int = 0, hardness: float = 1.0) -> Image.Image:
    """Remap alpha through a threshold and a contrast curve.

    `threshold` is the alpha value at or below which a pixel becomes fully transparent.
    `hardness` steepens the ramp around the midpoint: above 1 pushes half-transparent
    pixels toward fully on or fully off, which is what kills a smeared, ghostly edge.

    Identity at the defaults, so this costs nothing when unused.
    """
    threshold = int(round(min(max(threshold, 0), 254)))
    hardness = float(hardness)
    if threshold == 0 and abs(hardness - 1.0) < 1e-3:
        return alpha
    x = np.arange(256, dtype=np.float64)
    y = np.clip(x - threshold, 0.0, 255.0) / (255.0 - threshold)
    if abs(hardness - 1.0) > 1e-3:
        y = np.clip((y - 0.5) * hardness + 0.5, 0.0, 1.0)
    lut = np.clip(np.rint(y * 255.0), 0, 255).astype(np.uint8)
    return alpha.point(lut.tolist())


def _border_labels(labels: np.ndarray) -> np.ndarray:
    """Label ids that appear on the image edge — i.e. the real outside, not a hole."""
    edges = np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])
    return np.unique(edges)


def despeckle(alpha: Image.Image, min_area: int = 0, fill_area: int = 0) -> Image.Image:
    """Drop opaque islands smaller than `min_area`; fill transparent holes smaller than
    `fill_area`.

    Both are areas in pixels.

    `fill_area` leaves anything touching the image border alone, because the transparent
    surround *does* touch it and it is background, not a hole — filling it would make the
    whole frame opaque.

    `min_area` has no such exception, deliberately: leftover background very often sits
    against the frame edge, so sparing border-touching blobs would make the control useless
    on exactly the images that need it. The area threshold is the only protection, which is
    why the default is off and the slider says what it does.
    """
    min_area, fill_area = int(min_area), int(fill_area)
    if min_area <= 0 and fill_area <= 0:
        return alpha
    a = np.asarray(alpha, dtype=np.uint8)
    if a.size == 0:
        return alpha
    _require_scipy()

    out = a.copy()
    fg = a > OPAQUE
    changed = False

    if fill_area > 0:
        labels, n = _ndimage.label(~fg)
        if n:
            counts = np.bincount(labels.ravel(), minlength=n + 1)
            fillable = counts <= fill_area
            fillable[0] = False
            fillable[_border_labels(labels)] = False
            if fillable.any():
                hole = fillable[labels]
                out[hole] = 255
                fg = fg | hole
                changed = True

    if min_area > 0:
        labels, n = _ndimage.label(fg)
        if n:
            counts = np.bincount(labels.ravel(), minlength=n + 1)
            small = counts <= min_area
            small[0] = False
            if small.any():
                out[small[labels]] = 0
                changed = True

    return Image.fromarray(out, "L") if changed else alpha


@dataclass
class IslandInfo:
    label: int
    area: int
    is_largest: bool


def _labelled(alpha: Image.Image) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(labels, counts, fg) for the opaque regions of an alpha image."""
    a = np.asarray(alpha, dtype=np.uint8)
    fg = a > OPAQUE
    labels, n = _ndimage.label(fg)
    counts = np.bincount(labels.ravel(), minlength=n + 1) if n else np.zeros(1, np.int64)
    return labels, counts, fg


def island_at(alpha: Image.Image, x: float, y: float) -> IslandInfo | None:
    """The opaque island containing (x, y), or None when that pixel is transparent."""
    _require_scipy()
    labels, counts, _ = _labelled(alpha)
    if counts.size <= 1:
        return None
    h, w = labels.shape[:2]
    xi, yi = int(round(x)), int(round(y))
    if not (0 <= xi < w and 0 <= yi < h):
        return None
    label = int(labels[yi, xi])
    if label == 0:
        return None
    largest = int(np.argmax(counts[1:])) + 1
    return IslandInfo(label, int(counts[label]), label == largest)


def drop_island(alpha: Image.Image, label: int) -> Image.Image:
    """Set every pixel belonging to `label` to fully transparent."""
    _require_scipy()
    labels, _, _ = _labelled(alpha)
    out = np.asarray(alpha, dtype=np.uint8).copy()
    out[labels == label] = 0
    return Image.fromarray(out, "L")


def remove_island_at(alpha: Image.Image, x: float, y: float, protect_largest: bool = True
                     ) -> tuple[Image.Image | None, str]:
    """Delete the opaque island under (x, y).

    Returns `(new_alpha, message)`, where new_alpha is None when nothing was removed and
    the message explains why. The largest opaque region is refused by default: that is the
    subject, and silently deleting it on a stray click would be the worst thing this app
    could do.
    """
    if not HAVE_SCIPY:
        return None, f"Removing islands needs scipy. {SCIPY_WHY}"
    info = island_at(alpha, x, y)
    if info is None:
        return None, "That spot is already transparent — nothing to remove."
    if protect_largest and info.is_largest:
        return None, "That is the main subject. Use the erase brush if you really want it gone."
    return drop_island(alpha, info.label), f"Removed an island of {info.area:,} px."
