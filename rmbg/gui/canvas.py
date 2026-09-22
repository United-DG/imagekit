"""The preview surface, and the transform that makes a brush possible."""
from __future__ import annotations

import tkinter as tk
from typing import Optional

import customtkinter as ctk
from PIL import Image, ImageTk

from ..imaging import checkerboard
from .theme import C, checker_colors, is_dark, pick
from .widgets import button


class ImageCanvas(ctk.CTkFrame):
    """Draws a preview and remembers exactly how it drew it.

    The original app sized and centred its preview with no record of the mapping, so
    nothing interactive was possible. `transform` is that record; `to_source` is what
    makes the brush land where the user clicked.

    Layout is the image centred in the canvas, scaled to fit with a cap. On top of that:

    - `zoom` is a multiplier on the fit scale, so `zoom = 1` means "the whole image,
      fitted" whatever the window size, and the wheel moves it between `MIN_ZOOM` and
      `MAX_ZOOM`. `pan` is in canvas pixels away from the centred position.
    - Only the part of the image that is on screen is ever built. Drawing the whole frame
      is fine at fit — that is what a preview is — but at 16x a 12 MP image would be a
      half-gigabyte RGBA buffer, and all but a few percent of it lies outside the canvas.

    The mapping is a 6-tuple, `(scale, ox, oy, crop_x, crop_y, pscale)`, and it stays one:
    zoom and pan change only how `scale` and `ox`/`oy` are *computed*. `to_source` and
    everything built on it (`display_per_source`, the brush ring, the paint itself) never
    learn that they exist.
    """

    MARGIN = 12
    FIT_CAP = 4.0        # the fit rule's own ceiling, so a small image is not blown up
    MAX_ZOOM = 16.0      # user zoom, relative to fit. This is what the wheel moves.
    MIN_ZOOM = 0.1
    ZOOM_STEP = 1.25     # per wheel notch
    KEEP_VISIBLE = 48    # px of the image that must stay on screen, so it can be panned back

    def __init__(self, app, master, *, on_press=None, on_drag=None, on_release=None,
                 on_hover=None, on_configure=None, on_wheel=None, on_zoom=None):
        super().__init__(master, fg_color="transparent")
        self.app = app
        self._on_press, self._on_drag, self._on_release = on_press, on_drag, on_release
        self._on_hover, self._on_configure = on_hover, on_configure
        self._on_wheel, self._on_zoom = on_wheel, on_zoom

        self.canvas = tk.Canvas(self, bg=pick(C.panel), highlightthickness=0, bd=0)
        self.canvas.pack(fill="both", expand=True)

        self._img: Optional[Image.Image] = None
        self._crop: Optional[tuple[int, int, int, int]] = None
        self._pscale = 1.0
        self._photo = None
        self._chk_key = None
        self._chk_img = None
        self._ring = None
        # (scale, origin_x, origin_y, crop_x, crop_y, pscale); None until something is drawn
        self.transform: Optional[tuple] = None

        self.zoom = 1.0
        self.pan = [0.0, 0.0]
        self.space_held = False          # held down: a left-drag pans instead of painting
        self._fit_scale = 0.0            # the scale `zoom = 1` resolves to, for the readout
        self._panning = False
        self._pan_origin: Optional[tuple[float, float]] = None
        self._hint_used = False

        self._hint = self._make_hint()

        self.empty = ctk.CTkFrame(self, fg_color="transparent")
        ctk.CTkLabel(self.empty, text="Open an image to remove its background",
                     font=app.fonts.head, text_color=C.text).pack()
        ctk.CTkLabel(self.empty,
                     text="PNG, JPG, WebP, BMP and TIFF work.\n"
                          "Ctrl+O opens a file, Ctrl+V pastes from the clipboard.",
                     font=app.fonts.small, text_color=C.muted,
                     justify="center").pack(pady=(6, 14))
        button(app, self.empty, "Open image", app.open_image, primary=True, width=160).pack()
        self.show_empty()

        self.canvas.bind("<Configure>", self._configured)
        for seq, fn in (("<Button-1>", self._pressed), ("<B1-Motion>", self._dragged),
                        ("<ButtonRelease-1>", self._released),
                        ("<Button-2>", self._pressed), ("<B2-Motion>", self._dragged),
                        ("<ButtonRelease-2>", self._released),
                        ("<Button-3>", self._pressed), ("<B3-Motion>", self._dragged),
                        ("<ButtonRelease-3>", self._released),
                        ("<MouseWheel>", self._wheel), ("<Button-4>", self._wheel),
                        ("<Button-5>", self._wheel),
                        ("<Motion>", self._hovered), ("<Leave>", self._left)):
            self.canvas.bind(seq, fn)

    # ------------------------------------------------------------------ content
    def set_image(self, img: Optional[Image.Image], *, crop=None, pscale: float = 1.0) -> None:
        """Hand over a new preview. The zoom and pan are deliberately untouched: re-rendering
        after an edit must not throw away where the user is looking."""
        self._img, self._crop, self._pscale = img, crop, pscale
        if img is None:
            self.clear()
            return
        self.hide_empty()
        self.redraw()

    def clear(self) -> None:
        self._img = None
        self.transform = None
        self.canvas.delete("img")
        self.show_empty()

    def show_empty(self) -> None:
        self.empty.place(relx=0.5, rely=0.55, anchor="center")

    def hide_empty(self) -> None:
        self.empty.place_forget()

    def refresh_theme(self) -> None:
        self.canvas.configure(bg=pick(C.panel))
        if self._hint is not None:
            self.canvas.itemconfigure(self._hint, fill=pick(C.muted))
        self._chk_key = None
        self.redraw()

    # ------------------------------------------------------------------ view state
    def fit(self) -> None:
        """Back to the whole image, centred. `zoom = 1` is fit by definition."""
        self.zoom = 1.0
        self.pan = [0.0, 0.0]
        self.redraw()
        self._zoom_changed()

    def reset_view(self) -> None:
        """Same as `fit`, but without telling anyone: for a freshly loaded image, where the
        re-render is already happening and a second one would be wasted work."""
        self.zoom = 1.0
        self.pan = [0.0, 0.0]

    def zoom_percent(self) -> float:
        """What the readout shows: screen pixels per *source* pixel, as a percentage.

        The fit scale is per proxy pixel, so at fit this is the fit percentage rather than
        100 — which is the honest answer, and what every other editor reports.
        """
        return self.display_per_source() * 100.0

    def set_zoom(self, zoom: float, anchor: Optional[tuple[float, float]] = None) -> None:
        """Magnify, keeping the source pixel under `anchor` where it is.

        Holding the point under the cursor still is what makes wheel zoom usable: without it
        the thing being examined slides away as it grows, and the wheel becomes something to
        be avoided rather than the way to look at an edge.
        """
        zoom = min(max(float(zoom), self.MIN_ZOOM), self.MAX_ZOOM)
        if abs(zoom - self.zoom) < 1e-9:
            return
        before = self.to_source(*anchor) if anchor is not None else None
        self.zoom = zoom
        if anchor is None:
            self.pan = [0.0, 0.0]
            self.redraw()
        else:
            self.redraw()
            after = self.to_source(*anchor)
            if before is not None and after is not None and self.transform is not None:
                scale, _ox, _oy, _cx, _cy, pscale = self.transform
                # Move the image by however far that pixel drifted, in canvas pixels.
                self.pan[0] += (after[0] - before[0]) * scale * pscale
                self.pan[1] += (after[1] - before[1]) * scale * pscale
                self.redraw()
        self._zoom_changed()

    def zoom_by(self, factor: float, anchor: Optional[tuple[float, float]] = None) -> None:
        self.set_zoom(self.zoom * factor, anchor=anchor)

    def set_zoom_percent(self, percent: float) -> None:
        """100% is one screen pixel per source pixel, which is what a person means by it."""
        if self._fit_scale <= 0 or self._pscale <= 0:
            return
        self.set_zoom(percent / 100.0 / (self._fit_scale * self._pscale))

    def pan_by(self, dx: float, dy: float) -> None:
        self.pan[0] += dx
        self.pan[1] += dy
        self.redraw()

    def magnification(self) -> float:
        """Screen pixels per source pixel at this zoom, independent of the proxy.

        Derived from the fit rule and the source size rather than from the live transform,
        because the transform *depends on* the proxy resolution and the caller is asking in
        order to decide what that resolution should be.
        """
        source = getattr(self.app, "original", None)
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if source is None or cw < 60 or ch < 60:
            return 0.0
        w, h = source.size
        if not w or not h:
            return 0.0
        m = self.MARGIN
        return self.zoom * min((cw - 2 * m) / w, (ch - 2 * m) / h)

    # ------------------------------------------------------------------ drawing
    def redraw(self) -> None:
        if self._img is None:
            return
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if cw < 60 or ch < 60:
            return                       # not laid out yet; <Configure> will call again
        im = self._img
        m = self.MARGIN
        self._fit_scale = max(1e-9, min((cw - 2 * m) / im.width,
                                        (ch - 2 * m) / im.height, self.FIT_CAP))
        scale = self._fit_scale * self.zoom
        full_w, full_h = im.width * scale, im.height * scale
        self._clamp_pan(cw, ch, full_w, full_h)
        cx, cy = cw // 2 + self.pan[0], ch // 2 + self.pan[1]
        # Where image pixel (0, 0) lands. Everything else is this plus a scaled offset, which
        # is what lets the visible window be worked out before anything is allocated.
        ox, oy = cx - full_w / 2.0, cy - full_h / 2.0

        # The visible window, in image pixels. Rounding outwards and clipping to the image
        # keeps it inside `im`, and the +2 is slack against rounding so no edge shows a seam.
        x0 = max(0, int((0 - ox) / scale) - 2)
        y0 = max(0, int((0 - oy) / scale) - 2)
        x1 = min(im.width, int((cw - ox) / scale) + 3)
        y1 = min(im.height, int((ch - oy) / scale) + 3)

        crop_x, crop_y = (self._crop[0], self._crop[1]) if self._crop else (0, 0)
        self.transform = (scale, ox, oy, crop_x, crop_y, self._pscale)
        self.canvas.delete("img")
        self._photo = None
        if x1 <= x0 or y1 <= y0:
            self._place_hint(ch)
            return                       # panned entirely off screen; the transform still holds

        patch = im.crop((x0, y0, x1, y1))
        dw, dh = max(1, round(patch.width * scale)), max(1, round(patch.height * scale))
        if (patch.width, patch.height) == (dw, dh):
            # At 100% the patch is already the size it will be drawn at, and Pillow's resample
            # is not guaranteed bit-exact at a factor of one. Handing the pixels straight
            # through is what makes the magnified view *be* the export at that resolution
            # rather than a near copy of it.
            disp = patch
        else:
            disp = patch.resize((dw, dh), Image.LANCZOS if scale < 1 else Image.BICUBIC)
        # `.copy()` because Pillow may hand back a view onto the cached board's buffer for a
        # whole-region crop, and `alpha_composite` writes through it.
        comp = self._checker_at_least(dw, dh).crop((0, 0, dw, dh)).copy()
        comp.alpha_composite(disp.convert("RGBA"))
        # `master=` is load-bearing, not decoration. With no master, Pillow hands the photo
        # to tkinter's process-wide default root (`_get_default_root`, and `Tk._loadtk` only
        # claims that title while it is still None). If that root is ever a different
        # interpreter from this canvas, the photo simply does not exist here and
        # `create_image` dies with `image "pyimageN" does not exist`. Naming the canvas
        # makes the two impossible to separate.
        self._photo = ImageTk.PhotoImage(comp, master=self.canvas)
        self.canvas.create_image(ox + x0 * scale, oy + y0 * scale, image=self._photo,
                                 anchor="nw", tags="img")
        self.canvas.tag_raise("ring")
        self._place_hint(ch)

    def _checker_at_least(self, w: int, h: int) -> Image.Image:
        """A transparency grid at least `w` by `h`, cached in whole 12 px cells.

        Rounding up is what keeps this off the drag path: panning changes the patch by a pixel
        or two per motion event, and rebuilding an 860x660 board for each of those would be
        the most expensive thing in the gesture.
        """
        w, h = max(12, -(-int(w) // 12) * 12), max(12, -(-int(h) // 12) * 12)
        key = (w, h, is_dark())
        if key != self._chk_key:
            self._chk_img = checkerboard(w, h, *checker_colors()).convert("RGBA")
            self._chk_key = key
        return self._chk_img

    def _clamp_pan(self, cw: int, ch: int, full_w: float, full_h: float) -> None:
        """Stop the image being dragged out of reach: keep a handhold on screen.

        Expressing it as a bound on where the image's left edge may sit keeps the two axes the
        same calculation, and never binds when the image is at fit and fully visible.
        """
        if full_w <= 0 or full_h <= 0:
            return
        keep_x = min(self.KEEP_VISIBLE, full_w / 2.0, cw / 3.0)
        keep_y = min(self.KEEP_VISIBLE, full_h / 2.0, ch / 3.0)
        left = cw // 2 + self.pan[0] - full_w / 2.0
        top = ch // 2 + self.pan[1] - full_h / 2.0
        left = min(max(left, keep_x - full_w), cw - keep_x)
        top = min(max(top, keep_y - full_h), ch - keep_y)
        self.pan[0] = left - cw // 2 + full_w / 2.0
        self.pan[1] = top - ch // 2 + full_h / 2.0

    def _configured(self, _event=None) -> None:
        self.redraw()
        if self._on_configure:
            self._on_configure()

    # ------------------------------------------------------------------ hint line
    def _make_hint(self):
        """The one-line instruction along the bottom of the stage, in the canvas itself.

        A text item rather than a widget: it can never swallow a click, and it sits above the
        image without having to be kept in step with a layout.
        """
        try:
            font = self.app.fonts.small
        except Exception:                                    # pragma: no cover
            font = ("Segoe UI", 9)
        try:
            return self.canvas.create_text(10, 8, text="", anchor="sw", fill=pick(C.muted),
                                           font=font, tags="hint")
        except Exception:                                    # pragma: no cover - odd Tk build
            return None

    def set_hint(self, text: str) -> None:
        """The tools replace this with their own line; pass "" to clear it."""
        self._hint_used = False
        if self._hint is not None:
            self.canvas.itemconfigure(self._hint, text=text)

    def _place_hint(self, ch: int) -> None:
        """Bottom-left, until the interaction it describes has been used once."""
        if self._hint is None:
            return
        if self._hint_used:
            self.canvas.itemconfigure(self._hint, state="hidden")
            return
        self.canvas.coords(self._hint, 10, ch - 8)
        self.canvas.tag_raise("hint")

    def _used_hint(self) -> None:
        self._hint_used = True
        if self._hint is not None:
            self.canvas.itemconfigure(self._hint, state="hidden")

    # ------------------------------------------------------------------ mapping
    def to_source(self, x: float, y: float) -> Optional[tuple[float, float]]:
        """Canvas coordinates -> full-resolution source pixel. None when nothing is drawn."""
        t = self.transform
        if t is None:
            return None
        scale, ox, oy, crop_x, crop_y, pscale = t
        if pscale <= 0 or scale <= 0:
            return None
        return (((x - ox) / scale + crop_x) / pscale,
                ((y - oy) / scale + crop_y) / pscale)

    def display_per_source(self) -> float:
        """How many canvas pixels one source pixel currently occupies."""
        t = self.transform
        if t is None:
            return 1.0
        scale, _ox, _oy, _cx, _cy, pscale = t
        return max(1e-6, scale * pscale)

    def source_per_display(self) -> float:
        return 1.0 / self.display_per_source()

    # ------------------------------------------------------------------ brush ring
    def show_ring(self, x: float, y: float, radius_display: float, color) -> None:
        r = max(2.0, radius_display)
        coords = (x - r, y - r, x + r, y + r)
        if self._ring is None:
            self._ring = self.canvas.create_oval(*coords, outline=color, width=1, tags="ring")
        else:
            self.canvas.coords(self._ring, *coords)
            self.canvas.itemconfigure(self._ring, outline=color)
        self.canvas.tag_raise("ring")

    def hide_ring(self) -> None:
        if self._ring is not None:
            self.canvas.delete(self._ring)
            self._ring = None

    def set_cursor(self, name: str = "") -> None:
        """`""` restores the default. Tools name their own; the hand uses `fleur`."""
        try:
            self.canvas.configure(cursor=name or "arrow")
        except Exception:                                    # pragma: no cover - odd Tk build
            pass

    # ------------------------------------------------------------------ events
    def _pressed(self, e):
        if e.num == 2 or self.space_held:
            # Panning outranks the active tool, and outranks it silently: the tool never sees
            # a press, so it has no stroke to finish and a drag cannot half-happen.
            self._panning = True
            self._pan_origin = (e.x, e.y)
            self.set_cursor("fleur")
            return
        if self._on_press:
            self._on_press(e.x, e.y, e.num)

    def _dragged(self, e):
        if self._panning and self._pan_origin is not None:
            ox, oy = self._pan_origin
            self._pan_origin = (e.x, e.y)
            self.pan_by(e.x - ox, e.y - oy)
            self._used_hint()
            return
        if self._on_drag:
            self._on_drag(e.x, e.y, e.num)

    def _released(self, e):
        if self._panning:
            self._panning = False
            self._pan_origin = None
            self.set_cursor("fleur" if self.space_held else "")
            return
        if self._on_release:
            self._on_release(e.x, e.y, e.num)

    def _wheel(self, e):
        """Zoom about the cursor. A tool gets first refusal — the brush may want the wheel."""
        if self._on_wheel and self._on_wheel(e):
            return "break"
        delta = getattr(e, "delta", 0) or 0
        if not delta:                    # X11 reports the wheel as buttons 4 and 5
            delta = 120 if getattr(e, "num", 0) == 4 else -120 if getattr(e, "num", 0) == 5 else 0
        if not delta:
            return None
        self.zoom_by(self.ZOOM_STEP ** (delta / 120.0), anchor=(e.x, e.y))
        self._used_hint()
        return "break"

    def _hovered(self, e):
        if self._on_hover:
            self._on_hover(e.x, e.y)

    def _left(self, _e=None):
        self.hide_ring()

    def _zoom_changed(self) -> None:
        """Tell the page the magnification moved.

        The page is what knows the source size, so it is what decides whether the proxy is
        still good enough to look at — see `SinglePage._wanted_pscale`. Anything that needs
        real work must debounce: this fires on every notch of the wheel.
        """
        if self._on_zoom:
            self._on_zoom()
