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

    Layout is the same as before: the image is centred in the canvas, scaled to fit with
    a 4x cap. The one addition is the trim offset — when the image has been cropped, the
    preview no longer lines up with the source, so the crop origin has to be added back
    before dividing by the proxy scale.
    """

    MARGIN = 12
    MAX_ZOOM = 4.0

    def __init__(self, app, master, *, on_press=None, on_drag=None, on_release=None,
                 on_hover=None, on_configure=None):
        super().__init__(master, fg_color="transparent")
        self.app = app
        self._on_press, self._on_drag, self._on_release = on_press, on_drag, on_release
        self._on_hover, self._on_configure = on_hover, on_configure

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
                        ("<Button-3>", self._pressed), ("<B3-Motion>", self._dragged),
                        ("<ButtonRelease-3>", self._released),
                        ("<Motion>", self._hovered), ("<Leave>", self._left)):
            self.canvas.bind(seq, fn)

    # ------------------------------------------------------------------ content
    def set_image(self, img: Optional[Image.Image], *, crop=None, pscale: float = 1.0) -> None:
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
        self._chk_key = None
        self.redraw()

    # ------------------------------------------------------------------ drawing
    def redraw(self) -> None:
        if self._img is None:
            return
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if cw < 60 or ch < 60:
            return                       # not laid out yet; <Configure> will call again
        im = self._img
        m = self.MARGIN
        scale = max(1e-6, min((cw - 2 * m) / im.width, (ch - 2 * m) / im.height, self.MAX_ZOOM))
        dw = max(1, int(im.width * scale))
        dh = max(1, int(im.height * scale))
        disp = im.resize((dw, dh), Image.LANCZOS if scale < 1 else Image.BICUBIC).convert("RGBA")

        key = (dw, dh, is_dark())
        if key != self._chk_key:
            self._chk_img = checkerboard(dw, dh, *checker_colors()).convert("RGBA")
            self._chk_key = key
        comp = self._chk_img.copy()
        comp.alpha_composite(disp)
        # `master=` is load-bearing, not decoration. With no master, Pillow hands the photo
        # to tkinter's process-wide default root (`_get_default_root`, and `Tk._loadtk` only
        # claims that title while it is still None). If that root is ever a different
        # interpreter from this canvas, the photo simply does not exist here and
        # `create_image` dies with `image "pyimageN" does not exist`. Naming the canvas
        # makes the two impossible to separate.
        self._photo = ImageTk.PhotoImage(comp, master=self.canvas)

        cx, cy = cw // 2, ch // 2
        self.canvas.delete("img")
        self.canvas.create_image(cx, cy, image=self._photo, tags="img")
        self.canvas.tag_raise("ring")

        crop_x, crop_y = (self._crop[0], self._crop[1]) if self._crop else (0, 0)
        self.transform = (scale, cx - dw / 2.0, cy - dh / 2.0, crop_x, crop_y, self._pscale)

    def _configured(self, _event=None) -> None:
        self.redraw()
        if self._on_configure:
            self._on_configure()

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

    # ------------------------------------------------------------------ events
    def _pressed(self, e):
        if self._on_press:
            self._on_press(e.x, e.y, e.num)

    def _dragged(self, e):
        if self._on_drag:
            self._on_drag(e.x, e.y, e.num)

    def _released(self, e):
        if self._on_release:
            self._on_release(e.x, e.y, e.num)

    def _hovered(self, e):
        if self._on_hover:
            self._on_hover(e.x, e.y)

    def _left(self, _e=None):
        self.hide_ring()
