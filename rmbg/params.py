"""The settings object that drives both the live preview and the export."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

from .config import BG_MODES, DEFAULT_MODEL, FORMATS, MODEL_BY_ID, UPSCALES

# Bounds shared by the GUI sliders and the CLI, so the two can never drift apart.
RANGES = {
    "matting_fg": (150, 255),
    "matting_bg": (0, 100),
    "matting_erode": (0, 30),
    "mask_threshold": (0, 254),
    "mask_hardness": (0.25, 4.0),
    "despeckle": (0, 5000),
    "fill_holes": (0, 5000),
    "shrink": (-10, 10),
    "feather": (0.0, 10.0),
    "defringe": (0, 8),
    "brightness": (0.5, 1.5),
    "contrast": (0.5, 1.5),
    "saturation": (0.0, 2.0),
    "quality": (60, 100),
    # Not Params fields — the brush is GUI-only state — but they belong with the rest so
    # the two never disagree about what "the full range" means.
    "brush_size": (2, 400),
    "softness": (0.0, 1.0),
}


@dataclass
class Params:
    """Everything that shapes the output.

    Field order roughly follows the pipeline, so reading the class top to bottom walks
    the same path an image takes through `pipeline.render`.
    """

    # --- removal -----------------------------------------------------------
    model: str = DEFAULT_MODEL
    matting: bool = False
    matting_fg: int = 240
    matting_bg: int = 10
    matting_erode: int = 10
    post_process: bool = True       # rembg's built-in mask opening

    # --- mask quality ------------------------------------------------------
    mask_threshold: int = 0         # alpha at or below this becomes 0 (kills haze)
    mask_hardness: float = 1.0      # contrast around the midpoint (kills ghosting)
    despeckle: int = 0              # px area: drop alpha islands smaller than this
    fill_holes: int = 0             # px area: fill transparent holes inside the subject

    # --- edges and colour --------------------------------------------------
    shrink: int = 0                 # px of erosion; negative dilates (grows the subject)
    feather: float = 0.0            # px of edge softening
    defringe: int = 0               # px: recolour soft edges from opaque neighbours
    brightness: float = 1.0
    contrast: float = 1.0
    saturation: float = 1.0
    trim: bool = False              # crop away a fully transparent border

    # --- backdrop ----------------------------------------------------------
    bg_mode: str = "transparent"    # transparent | color | image
    bg_color: str = "#ffffff"
    bg_image: str = ""

    # --- export ------------------------------------------------------------
    fmt: str = "PNG"
    quality: int = 95
    upscale: int = 1

    def copy(self) -> "Params":
        """Snapshot, so background work can't be affected by later slider moves."""
        return replace(self)

    def to_dict(self) -> dict:
        return asdict(self)


# Fields the GUI persists between runs. Model/format/quality are the fiddly ones to reset.
PERSISTED = (
    "model", "fmt", "quality", "upscale", "matting", "matting_fg", "matting_bg",
    "matting_erode", "post_process", "mask_threshold", "mask_hardness", "despeckle",
    "fill_holes", "shrink", "feather", "defringe", "brightness", "contrast",
    "saturation", "trim", "bg_mode", "bg_color", "bg_image",
)


def from_dict(data: dict) -> Params:
    """Build Params from a persisted dict, ignoring unknown or badly typed keys.

    A settings file is untrusted input — it may have been written by an older version, or
    hand-edited. Fields that name a choice rather than a number are clamped here, once, so
    that no widget and no save dialog ever has to defend itself against a stale value.
    """
    defaults = Params()
    kwargs = {}
    for key, value in (data or {}).items():
        if key not in PERSISTED:
            continue
        default = getattr(defaults, key)
        try:
            kwargs[key] = bool(value) if isinstance(default, bool) else type(default)(value)
        except (TypeError, ValueError):
            continue
    p = Params(**kwargs)
    if p.model not in MODEL_BY_ID:
        p.model = DEFAULT_MODEL
    if p.fmt not in FORMATS:
        p.fmt = "PNG"
    if p.upscale not in UPSCALES:
        p.upscale = 1
    if p.bg_mode not in BG_MODES:
        p.bg_mode = "transparent"
    return p
