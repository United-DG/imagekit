"""Constants, the model registry and output formats."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
MAX_OUTPUT_PIXELS = 120_000_000        # refuse upscales beyond this
MAX_ZIP_ENTRY_BYTES = 200 * 1024 ** 2  # skip absurdly large ZIP members
PREVIEW_MAX = 1400                     # long edge of the live-preview proxy
TRIM_THRESHOLD = 8                     # alpha at or below this counts as empty when trimming

FORMATS = ["PNG", "WebP", "JPG"]
FORMAT_EXT = {"PNG": ".png", "WebP": ".webp", "JPG": ".jpg"}
UPSCALES = (1, 2, 3, 4)
BG_MODES = ("transparent", "color", "image")


@dataclass(frozen=True)
class ModelSpec:
    """One entry in the model picker.

    `size_mb` is the measured download size, or None when it has not been measured yet —
    the UI falls back to a vaguer sentence rather than inventing a number.
    """

    id: str
    label: str
    blurb: str
    size_mb: Optional[int] = None


# Ordered as shown in the picker. Sizes were read off the downloaded .onnx files.
MODELS: tuple[ModelSpec, ...] = (
    ModelSpec("u2net", "General purpose",
              "Solid all-rounder for objects and people. Fast on CPU.", 168),
    ModelSpec("isnet-general-use", "High detail",
              "Cleaner edges on complex subjects.", 170),
    ModelSpec("birefnet-general-lite", "Best edges (BiRefNet lite)",
              "Noticeably cleaner edges and fine detail. Slower, and a bigger download.", None),
    ModelSpec("birefnet-general", "Best quality (BiRefNet)",
              "The strongest general model here. Much slower on CPU.", None),
    ModelSpec("birefnet-portrait", "Portraits (BiRefNet)",
              "Tuned for people, hair and shoulders.", None),
    ModelSpec("u2net_human_seg", "People",
              "Tuned for portraits and full-body shots. Fast.", 168),
    ModelSpec("isnet-anime", "Anime and illustration",
              "Line art, characters and flat colour.", None),
    ModelSpec("silueta", "Lightweight",
              "Compact model with decent quality.", 43),
    ModelSpec("u2netp", "Fast",
              "Small and quick, less accurate.", 4),
)

MODEL_BY_ID = {m.id: m for m in MODELS}
MODEL_LABELS = [m.label for m in MODELS]
_LABEL_BY_ID = {m.id: m.label for m in MODELS}

# The default is deliberately the fast, already-cached model. BiRefNet is a real quality
# jump but is many times slower on CPU, so it is opt-in rather than a surprise on first run.
DEFAULT_MODEL = "u2net"


def label_for(model_id: str) -> str:
    """Picker label for a model id, falling back to the raw id."""
    return _LABEL_BY_ID.get(model_id, model_id)


def spec_for_label(label: str) -> ModelSpec:
    for m in MODELS:
        if m.label == label:
            return m
    raise KeyError(label)


# ---- where rembg keeps downloaded models ----------------------------------

def model_cache_dirs() -> list[Path]:
    """Directories rembg may have downloaded into, newest layout first."""
    dirs = []
    home = os.environ.get("REMBG_HOME")
    if home:
        dirs.append(Path(home).expanduser() / "models")
    dirs.append(Path.home() / ".rembg" / "models")
    dirs.append(Path.home() / ".u2net")           # legacy flat layout
    return dirs


def cached_model_bytes(model_id: str) -> int:
    """Bytes on disk for a model, or 0 when it has not been downloaded."""
    total = 0
    for d in model_cache_dirs():
        try:
            if d.name == "models":
                hits = list((d / model_id).glob("*.onnx"))
            else:
                hits = list(d.glob(f"{model_id}*.onnx"))
        except OSError:
            continue
        total += sum(f.stat().st_size for f in hits if f.is_file())
    return total


def is_model_cached(model_id: str) -> bool:
    return cached_model_bytes(model_id) > 0


# ---- settings --------------------------------------------------------------

def settings_path() -> Path:
    """Per-user settings file, next to the OS's config location."""
    try:
        from platformdirs import user_config_dir
        base = Path(user_config_dir("rmbg", appauthor=False))
    except Exception:                              # pragma: no cover - platformdirs absent
        base = Path.home() / ".config" / "rmbg"
    return base / "settings.json"
