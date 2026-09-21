#!/usr/bin/env python3
"""
Background remover
==================
Desktop app for removing image backgrounds, built on rembg (ONNX models) and
customtkinter.

    pip install -r requirements.txt
    python bgremover.py [optional_image_path]

Features
    - Single image: remove, refine edges, adjust colour, swap the backdrop, upscale, export
    - Batch: a folder or ZIP in, a folder or ZIP out, with cancel and per-file error reporting
    - Six models (downloaded on first use to ~/.u2net or ~/.rembg), alpha matting option
    - Light and dark themes

Layout of this file
    1. Core (no GUI): load, cut out, refine, compose, upscale, encode, batch
    2. GUI: App class and small widgets
"""
from __future__ import annotations

import io
import os
import platform
import queue
import sys
import threading
import traceback
import zipfile
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path, PurePosixPath
from typing import Callable, Optional

import numpy as np
from PIL import Image, ImageColor, ImageEnhance, ImageFilter, ImageOps

# --------------------------------------------------------------------------- #
# 1. CORE
# --------------------------------------------------------------------------- #

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
MAX_OUTPUT_PIXELS = 120_000_000        # refuse upscales beyond this
MAX_ZIP_ENTRY_BYTES = 200 * 1024 ** 2  # skip absurdly large ZIP members
Image.MAX_IMAGE_PIXELS = 200_000_000   # allow big photos, still block decompression bombs

# label -> (rembg model id, description shown in the UI)
MODELS: dict[str, tuple[str, str]] = {
    "General purpose": ("u2net", "Solid all-rounder for objects and people. Downloads about 170 MB on first use."),
    "High detail": ("isnet-general-use", "Cleaner edges on complex subjects. Downloads about 170 MB on first use."),
    "People": ("u2net_human_seg", "Tuned for portraits and full-body shots. Downloads about 170 MB on first use."),
    "Fast": ("u2netp", "Small and quick, less accurate. Downloads about 5 MB on first use."),
    "Lightweight": ("silueta", "Compact model with decent quality. Downloads about 40 MB on first use."),
}
FORMATS = ["PNG", "WebP", "JPG"]
FORMAT_EXT = {"PNG": ".png", "WebP": ".webp", "JPG": ".jpg"}


@dataclass
class Params:
    """Everything that shapes the output. One instance drives both preview and export."""
    model: str = "u2net"
    matting: bool = False
    matting_fg: int = 240
    matting_bg: int = 10
    shrink: int = 0            # px of edge erosion, removes leftover halo
    feather: float = 0.0       # px of edge softening
    brightness: float = 1.0
    contrast: float = 1.0
    saturation: float = 1.0
    trim: bool = False         # crop away fully transparent border
    bg_mode: str = "transparent"   # transparent | color | image
    bg_color: str = "#ffffff"
    bg_image: str = ""
    fmt: str = "PNG"
    quality: int = 95
    upscale: int = 1


def load_image(src) -> Image.Image:
    """Open a path or file-like as RGBA, honouring EXIF orientation."""
    with Image.open(src) as im:
        im.load()
        im = ImageOps.exif_transpose(im)
        return im.convert("RGBA")


class Engine:
    """Lazy wrapper around rembg that caches one session per model."""

    def __init__(self) -> None:
        self._sessions: dict[str, object] = {}
        self._lock = threading.Lock()

    def is_loaded(self, model: str) -> bool:
        return model in self._sessions

    def session(self, model: str):
        with self._lock:
            if model not in self._sessions:
                from rembg import new_session  # heavy import, keep lazy
                self._sessions[model] = new_session(model)
            return self._sessions[model]

    def cutout(self, img: Image.Image, p: Params) -> Image.Image:
        from rembg import remove
        sess = self.session(p.model)
        kwargs = {}
        if p.matting:
            kwargs = dict(
                alpha_matting=True,
                alpha_matting_foreground_threshold=int(p.matting_fg),
                alpha_matting_background_threshold=int(p.matting_bg),
                alpha_matting_erode_size=10,
            )
        out = remove(img.convert("RGB"), session=sess, **kwargs)
        return out.convert("RGBA")


def refine(img: Image.Image, p: Params, scale: float = 1.0, has_cutout: bool = True) -> Image.Image:
    """Colour adjustments plus edge shrink/feather. `scale` maps px values to a preview proxy."""
    r, g, b, a = img.split()
    rgb = Image.merge("RGB", (r, g, b))
    if abs(p.brightness - 1) > 1e-3:
        rgb = ImageEnhance.Brightness(rgb).enhance(p.brightness)
    if abs(p.contrast - 1) > 1e-3:
        rgb = ImageEnhance.Contrast(rgb).enhance(p.contrast)
    if abs(p.saturation - 1) > 1e-3:
        rgb = ImageEnhance.Color(rgb).enhance(p.saturation)
    if has_cutout:
        for _ in range(int(round(p.shrink * scale))):
            a = a.filter(ImageFilter.MinFilter(3))
        feather = p.feather * scale
        if feather > 0.05:
            a = a.filter(ImageFilter.GaussianBlur(feather))
    out = rgb.convert("RGBA")
    out.putalpha(a)
    return out


def trim_transparent(img: Image.Image, threshold: int = 8) -> Image.Image:
    mask = img.getchannel("A").point(lambda v: 255 if v > threshold else 0)
    box = mask.getbbox()
    return img.crop(box) if box else img


class BackdropCache:
    """Keeps the chosen backdrop image loaded and its last cover-fit result."""

    def __init__(self) -> None:
        self._path = None
        self._src: Optional[Image.Image] = None
        self._key = None
        self._out: Optional[Image.Image] = None

    def cover(self, path: str, size: tuple[int, int]) -> Image.Image:
        if path != self._path:
            self._src = load_image(path)
            self._path = path
            self._key = None
        if self._key != (path, size):
            self._out = ImageOps.fit(self._src, size, Image.LANCZOS)
            self._key = (path, size)
        return self._out


def compose(img: Image.Image, p: Params, cache: BackdropCache) -> Image.Image:
    if p.bg_mode == "color":
        base = Image.new("RGBA", img.size, ImageColor.getrgb(p.bg_color) + (255,))
    elif p.bg_mode == "image" and p.bg_image:
        base = cache.cover(p.bg_image, img.size).copy()
    else:
        return img
    base.alpha_composite(img)
    return base


def render(img: Image.Image, p: Params, *, has_cutout: bool, scale: float = 1.0,
           cache: Optional[BackdropCache] = None) -> Image.Image:
    """refine -> trim -> backdrop. Used by both the live preview and the export."""
    out = refine(img, p, scale, has_cutout)
    if p.trim and has_cutout:
        out = trim_transparent(out)
    return compose(out, p, cache or BackdropCache())


def upscale(img: Image.Image, factor: int) -> Image.Image:
    """Lanczos resize plus light unsharp mask. Adds pixels, not new detail."""
    if factor <= 1:
        return img
    w, h = img.size
    if w * h * factor * factor > MAX_OUTPUT_PIXELS:
        raise ValueError(f"{w * factor} x {h * factor} px is too large. Try a smaller upscale factor.")
    big = img.resize((w * factor, h * factor), Image.LANCZOS)  # premultiplied for RGBA, no dark fringes
    r, g, b, a = big.split()
    rgb = Image.merge("RGB", (r, g, b)).filter(ImageFilter.UnsharpMask(radius=1.4, percent=70, threshold=2))
    out = rgb.convert("RGBA")
    out.putalpha(a)
    return out


def encode(img: Image.Image, fmt: str, quality: int = 95) -> bytes:
    buf = io.BytesIO()
    fmt = fmt.upper()
    if fmt == "PNG":
        img.save(buf, "PNG")
    elif fmt == "WEBP":
        img.save(buf, "WEBP", quality=int(quality), method=4)
    elif fmt in ("JPG", "JPEG"):
        flat = Image.new("RGB", img.size, (255, 255, 255))
        flat.paste(img, mask=img.getchannel("A"))
        flat.save(buf, "JPEG", quality=int(quality), optimize=True)
    else:
        raise ValueError(f"Unsupported format: {fmt}")
    return buf.getvalue()


def make_output(img: Image.Image, p: Params, *, has_cutout: bool,
                cache: Optional[BackdropCache] = None) -> tuple[bytes, tuple[int, int]]:
    """Full-resolution pipeline: render -> upscale -> encode."""
    out = upscale(render(img, p, has_cutout=has_cutout, cache=cache), p.upscale)
    return encode(out, p.fmt, p.quality), out.size


def export_image(img: Image.Image, p: Params, dest: str | os.PathLike, *, has_cutout: bool) -> tuple[int, int]:
    data, size = make_output(img, p, has_cutout=has_cutout, cache=BackdropCache())
    Path(dest).write_bytes(data)
    return size


# ---- batch ----------------------------------------------------------------

@dataclass
class Entry:
    name: str
    read: Callable[[], Image.Image]


def _read_zip_member(zip_path: Path, member: str) -> Image.Image:
    with zipfile.ZipFile(zip_path) as zf:
        if zf.getinfo(member).file_size > MAX_ZIP_ENTRY_BYTES:
            raise ValueError("file is too large")
        return load_image(io.BytesIO(zf.read(member)))


def list_inputs(src: str | os.PathLike) -> list[Entry]:
    """Images in a folder (non-recursive) or a ZIP. Names are flattened, so no path traversal on output."""
    path = Path(src)
    entries: list[Entry] = []
    if path.is_dir():
        for f in sorted(path.iterdir(), key=lambda x: x.name.casefold()):
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS and not f.name.startswith("."):
                entries.append(Entry(f.name, partial(load_image, f)))
    elif path.is_file() and zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            for info in sorted(zf.infolist(), key=lambda i: i.filename.casefold()):
                if info.is_dir() or "__MACOSX" in info.filename:
                    continue
                base = PurePosixPath(info.filename.replace("\\", "/")).name
                if not base or base.startswith((".", "~")) or Path(base).suffix.lower() not in IMAGE_EXTS:
                    continue
                entries.append(Entry(base, partial(_read_zip_member, path, info.filename)))
    else:
        raise ValueError("Source must be a folder or a .zip file.")
    return entries


def _unique(name: str, used: set[str], folder: Optional[Path]) -> str:
    stem, ext = os.path.splitext(name)
    cand, n = name, 1
    while cand.lower() in used or (folder is not None and (folder / cand).exists()):
        n += 1
        cand = f"{stem}_{n}{ext}"
    used.add(cand.lower())
    return cand


@dataclass
class BatchResult:
    total: int = 0
    ok: int = 0
    failed: list[tuple[str, str]] = field(default_factory=list)
    cancelled: bool = False
    dest: str = ""


def process_batch(src: str, dest: str, to_zip: bool, p: Params, engine: Engine,
                  on_progress: Callable[[int, int, str], None] = lambda *a: None,
                  cancel: Optional[threading.Event] = None) -> BatchResult:
    entries = list_inputs(src)
    res = BatchResult(total=len(entries), dest=dest)
    if not entries:
        return res
    cancel = cancel or threading.Event()
    ext = FORMAT_EXT[p.fmt]
    used: set[str] = set()
    cache = BackdropCache()

    zf = None
    folder: Optional[Path] = None
    if to_zip:
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        zf = zipfile.ZipFile(dest, "w", zipfile.ZIP_STORED)
    else:
        folder = Path(dest)
        folder.mkdir(parents=True, exist_ok=True)

    try:
        on_progress(0, res.total, "Loading model")
        engine.session(p.model)
        for i, entry in enumerate(entries, 1):
            if cancel.is_set():
                res.cancelled = True
                break
            try:
                cut = engine.cutout(entry.read(), p)
                data, _ = make_output(cut, p, has_cutout=True, cache=cache)
                out_name = _unique(f"{Path(entry.name).stem}_nobg{ext}", used, folder)
                if zf is not None:
                    zf.writestr(out_name, data)
                else:
                    (folder / out_name).write_bytes(data)
                res.ok += 1
                on_progress(i, res.total, f"Saved {out_name}")
            except Exception as e:  # keep going, report at the end
                why = "not a readable image" if isinstance(e, Image.UnidentifiedImageError) else (str(e) or type(e).__name__)
                res.failed.append((entry.name, why))
                on_progress(i, res.total, f"Failed {entry.name}: {why}")
    finally:
        if zf is not None:
            zf.close()
    return res


# --------------------------------------------------------------------------- #
# 2. GUI
# --------------------------------------------------------------------------- #

import tkinter as tk
from tkinter import colorchooser, filedialog, messagebox

import customtkinter as ctk
from PIL import ImageTk

FAMILY = {"Windows": "Segoe UI", "Darwin": ".AppleSystemUIFont"}.get(platform.system(), "Noto Sans")
PREVIEW_MAX = 1400  # long edge of the live-preview proxy


class C:
    """(light, dark) colour pairs. Amber is reserved for primary actions and sliders."""
    bg = ("#edf0ee", "#0f191d")
    panel = ("#ffffff", "#15242a")
    field = ("#e2e8e5", "#1d3037")
    line = ("#cfd8d4", "#28444e")
    sel = ("#ffffff", "#30505b")
    text = ("#13201d", "#e7f0ed")
    muted = ("#566762", "#8fa6a0")
    accent = ("#d99a00", "#f2b134")
    accent_hi = ("#bf8600", "#ffc85a")
    on_accent = "#1b1300"
    danger = ("#b3261e", "#ff8f85")
    checker_light = ((255, 255, 255), (222, 228, 225))
    checker_dark = ((52, 74, 82), (38, 56, 63))


def checkerboard(w: int, h: int, c1, c2, size: int = 12) -> Image.Image:
    yy, xx = np.indices((h, w))
    m = ((xx // size + yy // size) % 2) == 0
    arr = np.where(m[..., None], np.array(c1, np.uint8), np.array(c2, np.uint8)).astype(np.uint8)
    return Image.fromarray(arr)


class SliderRow(ctk.CTkFrame):
    """Label + live value + slider in one tidy row."""

    def __init__(self, master, app: "App", label: str, lo: float, hi: float, value: float,
                 fmt: str = "{:.0f}", steps: Optional[int] = None, on_change=None):
        super().__init__(master, fg_color="transparent")
        self.fmt, self.on_change = fmt, on_change
        self.grid_columnconfigure(0, weight=1)
        self.lbl = ctk.CTkLabel(self, text=label, font=app.f_body, text_color=C.text, anchor="w")
        self.lbl.grid(row=0, column=0, sticky="w")
        self.val = ctk.CTkLabel(self, text=fmt.format(value), font=app.f_small, text_color=C.muted, anchor="e")
        self.val.grid(row=0, column=1, sticky="e")
        self.slider = ctk.CTkSlider(
            self, from_=lo, to=hi, number_of_steps=steps, command=self._changed, height=16,
            fg_color=C.field, progress_color=C.accent, button_color=C.accent, button_hover_color=C.accent_hi)
        self.slider.set(value)
        self.slider.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(4, 0))

    def _changed(self, v):
        self.val.configure(text=self.fmt.format(v))
        if self.on_change:
            self.on_change(v)

    def get(self) -> float:
        return self.slider.get()

    def set(self, v: float) -> None:
        self.slider.set(v)
        self.val.configure(text=self.fmt.format(v))

    def set_enabled(self, on: bool) -> None:
        self.slider.configure(
            state="normal" if on else "disabled",
            progress_color=C.accent if on else C.line, button_color=C.accent if on else C.line,
            button_hover_color=C.accent_hi if on else C.line)
        self.lbl.configure(text_color=C.text if on else C.muted)


class App(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Frionode Background remover")
        self.geometry("1240x800")
        self.minsize(1040, 720)
        self.configure(fg_color=C.bg)

        self.f_title = ctk.CTkFont(family=FAMILY, size=18, weight="bold")
        self.f_head = ctk.CTkFont(family=FAMILY, size=14, weight="bold")
        self.f_body = ctk.CTkFont(family=FAMILY, size=13)
        self.f_small = ctk.CTkFont(family=FAMILY, size=12)

        # state
        self.p = Params()
        self.engine = Engine()
        self.bgcache = BackdropCache()
        self.original: Optional[Image.Image] = None
        self.cutout: Optional[Image.Image] = None
        self.src_path = ""
        self.prev_orig: Optional[Image.Image] = None
        self.prev_cut: Optional[Image.Image] = None
        self.pscale = 1.0
        self.busy = False
        self.cancel_evt = threading.Event()
        self.q: "queue.Queue[Callable[[], None]]" = queue.Queue()
        self.lockable: list = []
        self._photo = None
        self._chk_key = None
        self._chk_img = None
        self._refresh_job = None

        self._build_shell()
        self._build_single()
        self._build_batch()
        self._show_page("Single image")
        self._bind_keys()
        self.refresh()
        self.after(50, self._poll)

    # ------------------------------------------------------------------ helpers
    def col(self, pair):
        return pair[1] if ctk.get_appearance_mode() == "Dark" else pair[0]

    def btn(self, parent, text, command, primary=False, **kw):
        if primary:
            b = ctk.CTkButton(parent, text=text, command=command, font=self.f_head, corner_radius=9, height=38,
                              fg_color=C.accent, hover_color=C.accent_hi, text_color=C.on_accent,
                              text_color_disabled=("#8a7440", "#7d6a3a"), **kw)
        else:
            b = ctk.CTkButton(parent, text=text, command=command, font=self.f_body, corner_radius=9, height=34,
                              fg_color=C.field, hover_color=C.line, text_color=C.text, **kw)
        return b

    def seg(self, parent, values, command=None, **kw):
        return ctk.CTkSegmentedButton(
            parent, values=values, command=command, font=self.f_body, corner_radius=8,
            fg_color=C.field, selected_color=C.sel, selected_hover_color=C.sel,
            unselected_color=C.field, unselected_hover_color=C.line, text_color=C.text, **kw)

    def label(self, parent, text, muted=False, wrap=0, **kw):
        return ctk.CTkLabel(parent, text=text, font=self.f_small if muted else self.f_body,
                            text_color=C.muted if muted else C.text, anchor="w", justify="left",
                            wraplength=wrap, **kw)

    def entry(self, parent, var, placeholder=""):
        return ctk.CTkEntry(parent, textvariable=var, placeholder_text=placeholder, font=self.f_body,
                            fg_color=C.field, border_width=0, text_color=C.text, height=34, corner_radius=8)

    def set_status(self, text: str, error: bool = False) -> None:
        self.status.configure(text=text, text_color=C.danger if error else C.muted)

    def _bind_keys(self) -> None:
        for mod in ("Control", "Command"):
            try:
                self.bind(f"<{mod}-o>", lambda e: self.open_image())
                self.bind(f"<{mod}-s>", lambda e: self.save_image())
                self.bind(f"<{mod}-r>", lambda e: self.remove_bg())
            except tk.TclError:
                pass

    # ------------------------------------------------------------------ shell
    def _build_shell(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        top = ctk.CTkFrame(self, fg_color="transparent")
        top.grid(row=0, column=0, sticky="ew", padx=20, pady=(16, 10))
        top.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(top, text="Background remover", font=self.f_title, text_color=C.text).grid(row=0, column=0, sticky="w")
        self.nav = self.seg(top, ["Single image", "Batch"], command=self._show_page)
        self.nav.set("Single image")
        self.nav.grid(row=0, column=1)
        self.theme_sw = ctk.CTkSwitch(top, text="Dark mode", font=self.f_body, text_color=C.text,
                                      command=self._toggle_theme, fg_color=C.line, progress_color=C.accent,
                                      button_color=("#ffffff", "#e7f0ed"), button_hover_color=("#ffffff", "#ffffff"))
        self.theme_sw.select()
        self.theme_sw.grid(row=0, column=2, sticky="e")

        self.body = ctk.CTkFrame(self, fg_color="transparent")
        self.body.grid(row=1, column=0, sticky="nsew", padx=20)
        self.body.grid_columnconfigure(0, weight=1)
        self.body.grid_rowconfigure(0, weight=1)

        bar = ctk.CTkFrame(self, fg_color="transparent")
        bar.grid(row=2, column=0, sticky="ew", padx=22, pady=(10, 14))
        bar.grid_columnconfigure(0, weight=1)
        self.status = ctk.CTkLabel(bar, text="Open an image to begin.", font=self.f_small, text_color=C.muted, anchor="w")
        self.status.grid(row=0, column=0, sticky="w")
        self.progress = ctk.CTkProgressBar(bar, width=240, height=8, fg_color=C.field, progress_color=C.field)
        self.progress.set(0)
        self.progress.grid(row=0, column=1, sticky="e")

    def _show_page(self, name: str) -> None:
        self.page_single.grid_remove()
        self.page_batch.grid_remove()
        (self.page_single if name == "Single image" else self.page_batch).grid(row=0, column=0, sticky="nsew")
        if name == "Batch":
            self._update_batch_summary()

    def _toggle_theme(self) -> None:
        ctk.set_appearance_mode("dark" if self.theme_sw.get() else "light")
        self.canvas.configure(bg=self.col(C.panel))
        self._chk_key = None
        self.refresh()

    # ------------------------------------------------------------------ single page
    def _build_single(self) -> None:
        page = self.page_single = ctk.CTkFrame(self.body, fg_color="transparent")
        page.grid_columnconfigure(0, weight=1)
        page.grid_rowconfigure(0, weight=1)

        # stage (preview)
        stage = ctk.CTkFrame(page, fg_color=C.panel, corner_radius=14)
        stage.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        stage.grid_columnconfigure(0, weight=1)
        stage.grid_rowconfigure(1, weight=1)

        tb = ctk.CTkFrame(stage, fg_color="transparent")
        tb.grid(row=0, column=0, sticky="ew", padx=14, pady=(14, 8))
        tb.grid_columnconfigure(2, weight=1)
        self.open_btn = self.btn(tb, "Open image", self.open_image, width=110)
        self.open_btn.grid(row=0, column=0)
        self.save_btn = self.btn(tb, "Save", self.save_image, width=70)
        self.save_btn.grid(row=0, column=1, padx=(8, 0))
        self.file_lbl = ctk.CTkLabel(tb, text="", font=self.f_small, text_color=C.muted, anchor="w")
        self.file_lbl.grid(row=0, column=2, sticky="ew", padx=14)
        self.view_seg = self.seg(tb, ["Original", "Result"], command=lambda v: self.refresh())
        self.view_seg.set("Result")
        self.view_seg.grid(row=0, column=3)
        self.lockable += [self.open_btn, self.save_btn]

        self.canvas = tk.Canvas(stage, bg=self.col(C.panel), highlightthickness=0, bd=0)
        self.canvas.grid(row=1, column=0, sticky="nsew", padx=14, pady=(0, 14))
        self.canvas.bind("<Configure>", lambda e: self.schedule_refresh())

        self.empty = ctk.CTkFrame(stage, fg_color="transparent")
        self.empty.place(relx=0.5, rely=0.55, anchor="center")
        ctk.CTkLabel(self.empty, text="Open an image to remove its background", font=self.f_head,
                     text_color=C.text).pack()
        ctk.CTkLabel(self.empty, text="PNG, JPG, WebP, BMP and TIFF work. Ctrl+O opens the file picker.",
                     font=self.f_small, text_color=C.muted).pack(pady=(4, 14))
        self.btn(self.empty, "Open image", self.open_image, primary=True, width=160).pack()

        # side panel
        side = ctk.CTkFrame(page, fg_color=C.panel, corner_radius=14, width=372)
        side.grid(row=0, column=1, sticky="ns")
        side.grid_propagate(False)
        side.grid_columnconfigure(0, weight=1)
        side.grid_rowconfigure(0, weight=1)
        self.tabs = ctk.CTkTabview(
            side, fg_color="transparent", border_width=0, corner_radius=10,
            segmented_button_fg_color=C.field, segmented_button_selected_color=C.sel,
            segmented_button_selected_hover_color=C.sel, segmented_button_unselected_color=C.field,
            segmented_button_unselected_hover_color=C.line, text_color=C.text)
        self.tabs.grid(row=0, column=0, sticky="nsew", padx=10, pady=(10, 10))
        for name in ("Remove", "Refine", "Backdrop", "Export"):
            self.tabs.add(name).grid_columnconfigure(0, weight=1)
        self._tab_remove(self.tabs.tab("Remove"))
        self._tab_refine(self.tabs.tab("Refine"))
        self._tab_backdrop(self.tabs.tab("Backdrop"))
        self._tab_export(self.tabs.tab("Export"))

    def _tab_remove(self, t) -> None:
        self.label(t, "Model").grid(row=0, column=0, sticky="w", pady=(6, 4))
        labels = list(MODELS)
        self.model_var = ctk.StringVar(value=labels[0])
        self.model_menu = ctk.CTkOptionMenu(
            t, values=labels, variable=self.model_var, command=self._model_changed, font=self.f_body,
            dropdown_font=self.f_body, fg_color=C.field, button_color=C.line, button_hover_color=C.line,
            text_color=C.text, dropdown_fg_color=C.panel, dropdown_text_color=C.text,
            dropdown_hover_color=C.field, height=34, corner_radius=8, anchor="w")
        self.model_menu.grid(row=1, column=0, sticky="ew")
        self.model_note = self.label(t, MODELS[labels[0]][1], muted=True, wrap=310)
        self.model_note.grid(row=2, column=0, sticky="w", pady=(6, 16))

        self.matting_sw = ctk.CTkSwitch(
            t, text="Alpha matting (hair, fur, soft edges)", font=self.f_body, text_color=C.text,
            command=self._matting_changed, fg_color=C.line, progress_color=C.accent,
            button_color=("#ffffff", "#e7f0ed"), button_hover_color=("#ffffff", "#ffffff"))
        self.matting_sw.grid(row=3, column=0, sticky="w")
        self.label(t, "Keeps fine strands but is slower, especially on large photos. Raise the foreground value if the subject looks eaten away.",
                   muted=True, wrap=310).grid(row=4, column=0, sticky="w", pady=(4, 10))
        self.fg_row = SliderRow(t, self, "Foreground threshold", 150, 255, 240, on_change=lambda v: setattr(self.p, "matting_fg", int(v)))
        self.fg_row.grid(row=5, column=0, sticky="ew", pady=(0, 8))
        self.bg_row = SliderRow(t, self, "Background threshold", 0, 100, 10, on_change=lambda v: setattr(self.p, "matting_bg", int(v)))
        self.bg_row.grid(row=6, column=0, sticky="ew")
        self.fg_row.set_enabled(False)
        self.bg_row.set_enabled(False)

        self.remove_btn = self.btn(t, "Remove background", self.remove_bg, primary=True)
        self.remove_btn.grid(row=7, column=0, sticky="ew", pady=(24, 0))
        self.lockable.append(self.remove_btn)

    def _tab_refine(self, t) -> None:
        self.label(t, "Edges", muted=True).grid(row=0, column=0, sticky="w", pady=(6, 6))
        self.shrink_row = SliderRow(t, self, "Shrink edge", 0, 10, 0, fmt="{:.0f} px", steps=10,
                                    on_change=lambda v: self._set("shrink", int(round(v))))
        self.shrink_row.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        self.feather_row = SliderRow(t, self, "Feather", 0, 10, 0, fmt="{:.1f} px", steps=40,
                                     on_change=lambda v: self._set("feather", round(v, 2)))
        self.feather_row.grid(row=2, column=0, sticky="ew", pady=(0, 16))

        self.label(t, "Colour", muted=True).grid(row=3, column=0, sticky="w", pady=(0, 6))
        self.bri_row = SliderRow(t, self, "Brightness", 0.5, 1.5, 1.0, fmt="{:.2f}", steps=100,
                                 on_change=lambda v: self._set("brightness", round(v, 3)))
        self.bri_row.grid(row=4, column=0, sticky="ew", pady=(0, 10))
        self.con_row = SliderRow(t, self, "Contrast", 0.5, 1.5, 1.0, fmt="{:.2f}", steps=100,
                                 on_change=lambda v: self._set("contrast", round(v, 3)))
        self.con_row.grid(row=5, column=0, sticky="ew", pady=(0, 10))
        self.sat_row = SliderRow(t, self, "Saturation", 0.0, 2.0, 1.0, fmt="{:.2f}", steps=100,
                                 on_change=lambda v: self._set("saturation", round(v, 3)))
        self.sat_row.grid(row=6, column=0, sticky="ew", pady=(0, 14))

        self.trim_sw = ctk.CTkSwitch(t, text="Trim transparent border", font=self.f_body, text_color=C.text,
                                     command=lambda: self._set("trim", bool(self.trim_sw.get())),
                                     fg_color=C.line, progress_color=C.accent,
                                     button_color=("#ffffff", "#e7f0ed"), button_hover_color=("#ffffff", "#ffffff"))
        self.trim_sw.grid(row=7, column=0, sticky="w")
        self.btn(t, "Reset refinements", self._reset_refine).grid(row=8, column=0, sticky="ew", pady=(16, 0))

    def _tab_backdrop(self, t) -> None:
        self.label(t, "Behind the subject", muted=True).grid(row=0, column=0, sticky="w", pady=(6, 6))
        self.bd_seg = self.seg(t, ["Transparent", "Colour", "Image"], command=self._backdrop_mode)
        self.bd_seg.set("Transparent")
        self.bd_seg.grid(row=1, column=0, sticky="ew")

        self.label(t, "Colour", muted=True).grid(row=2, column=0, sticky="w", pady=(20, 6))
        row = ctk.CTkFrame(t, fg_color="transparent")
        row.grid(row=3, column=0, sticky="ew")
        self.swatch = ctk.CTkButton(row, text="", width=38, height=38, corner_radius=8, border_width=1,
                                    border_color=C.line, fg_color=self.p.bg_color, hover_color=self.p.bg_color,
                                    command=self._pick_color)
        self.swatch.grid(row=0, column=0, padx=(0, 10))
        self.btn(row, "Custom colour", self._pick_color, width=120).grid(row=0, column=1)
        presets = ctk.CTkFrame(t, fg_color="transparent")
        presets.grid(row=4, column=0, sticky="w", pady=(12, 0))
        for i, hexv in enumerate(["#ffffff", "#f1efe9", "#111111", "#2f6fed", "#00b140"]):
            ctk.CTkButton(presets, text="", width=28, height=28, corner_radius=14, border_width=1,
                          border_color=C.line, fg_color=hexv, hover_color=hexv,
                          command=partial(self._set_color, hexv)).grid(row=0, column=i, padx=(0, 8))

        self.label(t, "Image", muted=True).grid(row=5, column=0, sticky="w", pady=(20, 6))
        self.btn(t, "Choose backdrop image", self._pick_backdrop_image).grid(row=6, column=0, sticky="ew")
        self.bd_file = self.label(t, "No image chosen", muted=True, wrap=310)
        self.bd_file.grid(row=7, column=0, sticky="w", pady=(6, 0))

    def _tab_export(self, t) -> None:
        self.label(t, "Format", muted=True).grid(row=0, column=0, sticky="w", pady=(6, 6))
        self.fmt_seg = self.seg(t, FORMATS, command=self._format_changed)
        self.fmt_seg.set("PNG")
        self.fmt_seg.grid(row=1, column=0, sticky="ew")
        self.label(t, "PNG and WebP keep transparency. JPG fills it with white.", muted=True, wrap=310)\
            .grid(row=2, column=0, sticky="w", pady=(6, 14))
        self.q_row = SliderRow(t, self, "Quality (WebP, JPG)", 60, 100, 95, steps=40,
                               on_change=lambda v: self._set("quality", int(v), redraw=False))
        self.q_row.grid(row=3, column=0, sticky="ew", pady=(0, 18))
        self.q_row.set_enabled(False)

        self.label(t, "Upscale", muted=True).grid(row=4, column=0, sticky="w", pady=(0, 6))
        self.up_seg = self.seg(t, ["1×", "2×", "3×", "4×"], command=self._upscale_changed)
        self.up_seg.set("1×")
        self.up_seg.grid(row=5, column=0, sticky="ew")
        self.label(t, "Sharp resize (Lanczos). It adds pixels, not new detail.", muted=True, wrap=310)\
            .grid(row=6, column=0, sticky="w", pady=(6, 14))
        self.size_lbl = self.label(t, "Output size: no image yet")
        self.size_lbl.grid(row=7, column=0, sticky="w")
        self.export_btn = self.btn(t, "Save image", self.save_image, primary=True)
        self.export_btn.grid(row=8, column=0, sticky="ew", pady=(22, 0))
        self.lockable.append(self.export_btn)

    # ---- single-page callbacks
    def _set(self, key: str, value, redraw: bool = True) -> None:
        setattr(self.p, key, value)
        if redraw:
            self.schedule_refresh()

    def _model_changed(self, label: str) -> None:
        self.p.model, note = MODELS[label]
        self.model_note.configure(text=note)

    def _matting_changed(self) -> None:
        on = bool(self.matting_sw.get())
        self.p.matting = on
        self.fg_row.set_enabled(on)
        self.bg_row.set_enabled(on)

    def _reset_refine(self) -> None:
        for row, v in ((self.shrink_row, 0), (self.feather_row, 0), (self.bri_row, 1), (self.con_row, 1), (self.sat_row, 1)):
            row.set(v)
        self.trim_sw.deselect()
        self.p.shrink, self.p.feather = 0, 0.0
        self.p.brightness = self.p.contrast = self.p.saturation = 1.0
        self.p.trim = False
        self.refresh()

    def _backdrop_mode(self, value: str) -> None:
        self.p.bg_mode = {"Transparent": "transparent", "Colour": "color", "Image": "image"}[value]
        if self.p.bg_mode == "image" and not self.p.bg_image:
            self._pick_backdrop_image()
            return
        self.refresh()

    def _set_color(self, hexv: str) -> None:
        self.p.bg_color = hexv
        self.swatch.configure(fg_color=hexv, hover_color=hexv)
        self.p.bg_mode = "color"
        self.bd_seg.set("Colour")
        self.view_seg.set("Result")
        self.refresh()

    def _pick_color(self) -> None:
        _, hexv = colorchooser.askcolor(color=self.p.bg_color, parent=self, title="Backdrop colour")
        if hexv:
            self._set_color(hexv)

    def _pick_backdrop_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose backdrop image", filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff")])
        if not path:
            if not self.p.bg_image:
                self.p.bg_mode = "transparent"
                self.bd_seg.set("Transparent")
            return
        self._set_backdrop_image(path)

    def _set_backdrop_image(self, path: str) -> None:
        try:
            load_image(path)
        except Exception as e:
            messagebox.showerror("Can't open image", f"{Path(path).name} couldn't be opened.\n\n{e}")
            return
        self.p.bg_image, self.p.bg_mode = path, "image"
        self.bd_file.configure(text=Path(path).name)
        self.bd_seg.set("Image")
        self.view_seg.set("Result")
        self.refresh()

    def _format_changed(self, value: str) -> None:
        self.p.fmt = value
        self.q_row.set_enabled(value != "PNG")

    def _upscale_changed(self, value: str) -> None:
        self.p.upscale = int(value[0])
        self.refresh()

    # ------------------------------------------------------------------ preview
    def _make_proxies(self) -> None:
        w, h = self.original.size
        self.pscale = min(1.0, PREVIEW_MAX / max(w, h))
        size = (max(1, round(w * self.pscale)), max(1, round(h * self.pscale)))
        shrink = (lambda im: im.resize(size, Image.LANCZOS)) if self.pscale < 1 else (lambda im: im)
        self.prev_orig = shrink(self.original)
        self.prev_cut = shrink(self.cutout) if self.cutout is not None else None

    def schedule_refresh(self) -> None:
        if self._refresh_job:
            self.after_cancel(self._refresh_job)
        self._refresh_job = self.after(40, self.refresh)

    def refresh(self) -> None:
        self._refresh_job = None
        if self.original is None:
            self.canvas.delete("all")
            self.empty.lift()
            self.empty.place(relx=0.5, rely=0.55, anchor="center")
            return
        self.empty.place_forget()
        try:
            if self.view_seg.get() == "Original":
                img = self.prev_orig
            else:
                base = self.prev_cut if self.prev_cut is not None else self.prev_orig
                img = render(base, self.p, has_cutout=self.cutout is not None, scale=self.pscale, cache=self.bgcache)
        except Exception as e:
            self.p.bg_mode = "transparent"
            self.bd_seg.set("Transparent")
            self.set_status(f"Backdrop failed: {e}", error=True)
            img = self.prev_cut if self.prev_cut is not None else self.prev_orig
        self._draw(img)
        # size readout
        w = round(img.width / self.pscale) * self.p.upscale
        h = round(img.height / self.pscale) * self.p.upscale
        approx = "about " if self.p.trim and self.cutout is not None else ""
        self.size_lbl.configure(text=f"Output size: {approx}{w} × {h} px")

    def _draw(self, img: Image.Image) -> None:
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if cw < 60 or ch < 60:
            return
        m = 12
        s = min((cw - 2 * m) / img.width, (ch - 2 * m) / img.height, 4.0)
        dw, dh = max(1, int(img.width * s)), max(1, int(img.height * s))
        disp = img.resize((dw, dh), Image.LANCZOS if s < 1 else Image.BICUBIC).convert("RGBA")
        dark = ctk.get_appearance_mode() == "Dark"
        key = (dw, dh, dark)
        if key != self._chk_key:
            c1, c2 = C.checker_dark if dark else C.checker_light
            self._chk_img = checkerboard(dw, dh, c1, c2).convert("RGBA")
            self._chk_key = key
        comp = self._chk_img.copy()
        comp.alpha_composite(disp)
        self._photo = ImageTk.PhotoImage(comp)
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, image=self._photo)

    # ------------------------------------------------------------------ busy / threads
    def set_busy(self, busy: bool, msg: str = "", determinate: bool = False) -> None:
        self.busy = busy
        for w in self.lockable:
            w.configure(state="disabled" if busy else "normal")
        self.cancel_btn.configure(state="normal" if (busy and determinate) else "disabled")
        self.progress.stop()
        self.progress.configure(progress_color=C.accent if busy else C.field)
        if busy and not determinate:
            self.progress.configure(mode="indeterminate")
            self.progress.start()
        else:
            self.progress.configure(mode="determinate")
            self.progress.set(0)
        self.set_status(msg)

    def run_task(self, work: Callable, done: Callable, msg: str, determinate: bool = False) -> None:
        if self.busy:
            return
        self.set_busy(True, msg, determinate)

        def runner():
            try:
                res, err = work(), None
            except Exception as e:
                traceback.print_exc()
                res, err = None, e
            self.q.put(lambda: self._finish(done, res, err))

        threading.Thread(target=runner, daemon=True).start()

    def _finish(self, done, res, err) -> None:
        self.set_busy(False, "")
        done(res, err)

    def _poll(self) -> None:
        try:
            while True:
                self.q.get_nowait()()
        except queue.Empty:
            pass
        except Exception:
            traceback.print_exc()
        self.after(50, self._poll)

    @staticmethod
    def friendly(err: Exception) -> str:
        if isinstance(err, Image.UnidentifiedImageError):
            return "This file isn't a readable image."
        text = str(err) or type(err).__name__
        low = text.lower()
        if any(k in low for k in ("connection", "urlopen", "timed out", "name resolution", "http")):
            return f"The model couldn't be downloaded. Check your internet connection and try again.\n\n{text}"
        return text

    # ------------------------------------------------------------------ actions: single image
    def open_image(self, path: Optional[str] = None) -> None:
        if self.busy:
            return
        path = path or filedialog.askopenfilename(
            title="Open image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff"), ("All files", "*.*")])
        if not path:
            return
        try:
            img = load_image(path)
        except Exception as e:
            messagebox.showerror("Can't open image", f"{Path(path).name} couldn't be opened.\n\n{self.friendly(e)}")
            return
        self.original, self.cutout, self.src_path = img, None, path
        self._make_proxies()
        self.view_seg.set("Original")
        self.file_lbl.configure(text=f"{Path(path).name}, {img.width} × {img.height} px")
        self.set_status("Loaded. Choose a model on the Remove tab, then remove the background.")
        self.tabs.set("Remove")
        self.refresh()

    def remove_bg(self) -> None:
        if self.busy:
            return
        if self.original is None:
            self.set_status("Open an image first.", error=True)
            return
        p, img = replace(self.p), self.original
        first_use = not self.engine.is_loaded(p.model)
        msg = "Preparing the model (first use downloads it)…" if first_use else "Removing background…"

        def work():
            return self.engine.cutout(img, p)

        def done(res, err):
            if err:
                self.set_status("Background removal failed.", error=True)
                messagebox.showerror("Background removal failed", self.friendly(err))
                return
            self.cutout = res
            self._make_proxies()
            self.view_seg.set("Result")
            self.set_status("Background removed. Fine-tune it on the Refine and Backdrop tabs, then save.")
            self.refresh()

        self.run_task(work, done, msg)

    def save_image(self) -> None:
        if self.busy:
            return
        if self.original is None:
            self.set_status("Open an image first.", error=True)
            return
        ext = FORMAT_EXT[self.p.fmt]
        stem = Path(self.src_path).stem + ("_nobg" if self.cutout is not None else "")
        path = filedialog.asksaveasfilename(
            title="Save image", defaultextension=ext, initialfile=stem + ext,
            filetypes=[(self.p.fmt, f"*{ext}")])
        if path:
            self._save_to(path)

    def _save_to(self, path: str) -> None:
        p = replace(self.p)
        src = self.cutout if self.cutout is not None else self.original
        has_cut = self.cutout is not None

        def work():
            return export_image(src, p, path, has_cutout=has_cut)

        def done(size, err):
            if err:
                self.set_status("Save failed.", error=True)
                messagebox.showerror("Save failed", str(err))
            else:
                self.set_status(f"Saved {Path(path).name} ({size[0]} × {size[1]} px).")

        self.run_task(work, done, "Saving…")

    # ------------------------------------------------------------------ batch page
    def _build_batch(self) -> None:
        page = self.page_batch = ctk.CTkFrame(self.body, fg_color="transparent")
        page.grid_columnconfigure(0, weight=1)
        page.grid_rowconfigure(0, weight=1)
        card = ctk.CTkFrame(page, fg_color=C.panel, corner_radius=14)
        card.grid(row=0, column=0, sticky="nsew")
        card.grid_columnconfigure(1, weight=1)
        card.grid_rowconfigure(8, weight=1)

        self.b_src, self.b_dst = ctk.StringVar(), ctk.StringVar()
        pad = dict(padx=(24, 24))

        ctk.CTkLabel(card, text="Process many images at once", font=self.f_head, text_color=C.text)\
            .grid(row=0, column=0, columnspan=3, sticky="w", pady=(22, 2), **pad)
        self.label(card, "Pick a folder or ZIP of images. Each one is saved with the suffix _nobg. "
                         "Existing files are never overwritten.", muted=True)\
            .grid(row=1, column=0, columnspan=3, sticky="w", pady=(0, 18), **pad)

        self.label(card, "Source").grid(row=2, column=0, sticky="w", padx=(24, 12), pady=6)
        self.b_src_entry = self.entry(card, self.b_src, "Folder or ZIP file")
        self.b_src_entry.grid(row=2, column=1, sticky="ew", pady=6)
        srcbtns = ctk.CTkFrame(card, fg_color="transparent")
        srcbtns.grid(row=2, column=2, sticky="e", padx=(10, 24))
        self.b_folder_btn = self.btn(srcbtns, "Folder…", self._pick_src_folder, width=84)
        self.b_folder_btn.grid(row=0, column=0)
        self.b_zip_btn = self.btn(srcbtns, "ZIP…", self._pick_src_zip, width=64)
        self.b_zip_btn.grid(row=0, column=1, padx=(8, 0))
        self.b_count = self.label(card, "", muted=True)
        self.b_count.grid(row=3, column=1, sticky="w")

        self.label(card, "Save as").grid(row=4, column=0, sticky="w", padx=(24, 12), pady=(14, 6))
        self.b_kind = self.seg(card, ["Folder", "ZIP file"], command=lambda v: self.b_dst.set(""))
        self.b_kind.set("Folder")
        self.b_kind.grid(row=4, column=1, sticky="w", pady=(14, 6))
        self.label(card, "Save to").grid(row=5, column=0, sticky="w", padx=(24, 12), pady=6)
        self.b_dst_entry = self.entry(card, self.b_dst, "Where the results go")
        self.b_dst_entry.grid(row=5, column=1, sticky="ew", pady=6)
        self.b_dst_btn = self.btn(card, "Browse…", self._pick_dst, width=84)
        self.b_dst_btn.grid(row=5, column=2, sticky="e", padx=(10, 24))

        self.b_summary = self.label(card, "", muted=True, wrap=760)
        self.b_summary.grid(row=6, column=0, columnspan=3, sticky="w", pady=(14, 0), **pad)

        actions = ctk.CTkFrame(card, fg_color="transparent")
        actions.grid(row=7, column=0, columnspan=3, sticky="w", pady=(18, 12), **pad)
        self.b_start = self.btn(actions, "Start batch", self.start_batch, primary=True, width=150)
        self.b_start.grid(row=0, column=0)
        self.cancel_btn = self.btn(actions, "Cancel", self._cancel_batch, width=90)
        self.cancel_btn.grid(row=0, column=1, padx=(10, 0))
        self.cancel_btn.configure(state="disabled")
        self.lockable += [self.b_start, self.b_folder_btn, self.b_zip_btn, self.b_dst_btn]

        self.log = ctk.CTkTextbox(card, font=self.f_small, fg_color=C.field, text_color=C.text,
                                  corner_radius=10, border_width=0, wrap="word")
        self.log.grid(row=8, column=0, columnspan=3, sticky="nsew", padx=24, pady=(0, 24))
        self.log.configure(state="disabled")

    def _update_batch_summary(self) -> None:
        p = self.p
        name = next((k for k, v in MODELS.items() if v[0] == p.model), p.model)
        bd = {"transparent": "transparent backdrop", "color": f"{p.bg_color} backdrop", "image": "image backdrop"}[p.bg_mode]
        self.b_summary.configure(
            text=f"Uses your settings from the Single image page: {name} model, matting {'on' if p.matting else 'off'}, "
                 f"{bd}, {p.fmt} output at {p.upscale}×.")

    def _pick_src_folder(self) -> None:
        d = filedialog.askdirectory(title="Choose a folder of images")
        if d:
            self.b_src.set(d)
            self._count_inputs()

    def _pick_src_zip(self) -> None:
        f = filedialog.askopenfilename(title="Choose a ZIP of images", filetypes=[("ZIP files", "*.zip")])
        if f:
            self.b_src.set(f)
            self._count_inputs()

    def _count_inputs(self) -> None:
        try:
            n = len(list_inputs(self.b_src.get()))
            self.b_count.configure(text=f"{n} image{'s' if n != 1 else ''} found", text_color=C.muted)
        except Exception as e:
            self.b_count.configure(text=str(e), text_color=C.danger)

    def _pick_dst(self) -> None:
        if self.b_kind.get() == "ZIP file":
            f = filedialog.asksaveasfilename(title="Save results as", defaultextension=".zip",
                                             initialfile="background_removed.zip", filetypes=[("ZIP files", "*.zip")])
            if f:
                self.b_dst.set(f)
        else:
            d = filedialog.askdirectory(title="Choose an output folder")
            if d:
                self.b_dst.set(d)

    def _log(self, line: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", line + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _cancel_batch(self) -> None:
        self.cancel_evt.set()
        self.set_status("Cancelling after the current image…")

    def start_batch(self, *, quiet: bool = False) -> None:
        if self.busy:
            return
        src, dst = self.b_src.get().strip(), self.b_dst.get().strip()
        to_zip = self.b_kind.get() == "ZIP file"
        problem = None
        if not src:
            problem = "Choose a folder or ZIP file with images."
        elif not dst:
            problem = "Choose where to save the results."
        else:
            try:
                if not list_inputs(src):
                    problem = "No supported images found in that source."
            except Exception as e:
                problem = str(e)
        if problem:
            self.set_status(problem, error=True)
            return
        if to_zip and not dst.lower().endswith(".zip"):
            dst += ".zip"
            self.b_dst.set(dst)

        p = replace(self.p)
        self.cancel_evt = evt = threading.Event()
        self._update_batch_summary()
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

        def prog(i, n, msg):
            self.q.put(lambda: self._batch_progress(i, n, msg))

        def work():
            return process_batch(src, dst, to_zip, p, self.engine, prog, evt)

        first_use = not self.engine.is_loaded(p.model)
        self.run_task(work, self._batch_done,
                      "Preparing the model (first use downloads it)…" if first_use else "Processing…",
                      determinate=True)

    def _batch_progress(self, i: int, n: int, msg: str) -> None:
        if not self.busy:
            return
        self.progress.set(i / max(n, 1))
        if i:
            self.set_status(f"{i} of {n} done")
        self._log(msg)

    def _batch_done(self, res: Optional[BatchResult], err) -> None:
        if err:
            self.set_status("Batch failed.", error=True)
            self._log(f"Error: {err}")
            messagebox.showerror("Batch failed", self.friendly(err))
            return
        if not res.cancelled:
            self.progress.configure(progress_color=C.accent)
            self.progress.set(1)
        parts = [f"{res.ok} saved"]
        if res.failed:
            parts.append(f"{len(res.failed)} failed")
        if res.cancelled:
            parts.append("cancelled")
        self.set_status(f"Batch finished: {', '.join(parts)}.", error=bool(res.failed))
        self._log(f"Finished. Results are in {res.dest}")


def main() -> None:
    ctk.set_appearance_mode("dark")
    app = App()
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        app.after(200, lambda: app.open_image(sys.argv[1]))
    app.mainloop()


if __name__ == "__main__":
    main()