"""The single-image page: preview, tabs, and interactive touch-up."""
from __future__ import annotations

from pathlib import Path
from tkinter import colorchooser, filedialog, messagebox

import customtkinter as ctk
from PIL import Image

from ..config import (DEFAULT_MODEL, FORMATS, MODELS, MODEL_BY_ID, PREVIEW_MAX, label_for,
                      spec_for_label)
from ..edit import Segment, paint_segment
from ..imaging import load_image, mask_preview
from ..params import RANGES
from ..pipeline import render
from .canvas import ImageCanvas
from .theme import C, pick
from .widgets import SliderRow, button, label, segment, switch

VIEWS = ["Original", "Result", "Mask"]
TOOLS = ["Erase", "Restore", "Remove leftover"]


class SinglePage(ctk.CTkFrame):
    """Everything for one image: the stage on the left, the setting tabs on the right."""

    def __init__(self, app, master) -> None:
        super().__init__(master, fg_color="transparent")
        self.app = app

        # preview state
        self.pscale = 1.0
        self.prev_orig: Image.Image | None = None
        self.prev_rgb: Image.Image | None = None
        self.prev_alpha: Image.Image | None = None
        self._refresh_job = None

        # touch-up state
        self.tool = "erase"
        self.brush_size = 24.0        # in source pixels
        self.brush_softness = 0.5
        self._dragging = False
        self._drag_last: tuple[float, float] | None = None

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self._build_stage()
        self._build_side()

    # ================================================================== layout
    def _build_stage(self) -> None:
        stage = ctk.CTkFrame(self, fg_color=C.panel, corner_radius=14)
        stage.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        stage.grid_columnconfigure(0, weight=1)
        stage.grid_rowconfigure(1, weight=1)

        tb = ctk.CTkFrame(stage, fg_color="transparent")
        tb.grid(row=0, column=0, sticky="ew", padx=14, pady=(14, 8))
        tb.grid_columnconfigure(2, weight=1)

        self.open_btn = button(self.app, tb, "Open image", self.app.open_image, width=110)
        self.open_btn.grid(row=0, column=0)
        self.save_btn = button(self.app, tb, "Save", self.app.save_image, width=70)
        self.save_btn.grid(row=0, column=1, padx=(8, 0))
        self.file_lbl = ctk.CTkLabel(tb, text="", font=self.app.fonts.small,
                                     text_color=C.muted, anchor="w")
        self.file_lbl.grid(row=0, column=2, sticky="ew", padx=14)
        self.view_seg = segment(self.app, tb, VIEWS, command=lambda v: self.refresh())
        self.view_seg.set("Result")
        self.view_seg.grid(row=0, column=3)

        self.canvas = ImageCanvas(
            self.app, stage,
            on_press=self._press, on_drag=self._drag, on_release=self._release,
            on_hover=self._hover)
        self.canvas.grid(row=1, column=0, sticky="nsew", padx=14, pady=(0, 14))

        self.app.lockable += [self.open_btn, self.save_btn]

    def _build_side(self) -> None:
        side = ctk.CTkFrame(self, fg_color=C.panel, corner_radius=14, width=380)
        side.grid(row=0, column=1, sticky="ns")
        side.grid_propagate(False)
        side.grid_columnconfigure(0, weight=1)
        side.grid_rowconfigure(0, weight=1)
        self.tabs = ctk.CTkTabview(
            side, fg_color="transparent", border_width=0, corner_radius=10,
            segmented_button_fg_color=C.field, segmented_button_selected_color=C.sel,
            segmented_button_selected_hover_color=C.sel,
            segmented_button_unselected_color=C.field,
            segmented_button_unselected_hover_color=C.line, text_color=C.text)
        self.tabs.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        for name in ("Remove", "Refine", "Touch up", "Backdrop", "Export"):
            self.tabs.add(name).grid_columnconfigure(0, weight=1)
        self._tab_remove(self.tabs.tab("Remove"))
        self._tab_refine(self.tabs.tab("Refine"))
        self._tab_touch(self.tabs.tab("Touch up"))
        self._tab_backdrop(self.tabs.tab("Backdrop"))
        self._tab_export(self.tabs.tab("Export"))

    # ================================================================== tabs
    def _tab_remove(self, t) -> None:
        label(self.app, t, "Model").grid(row=0, column=0, sticky="w", pady=(6, 4))
        labels = [m.label for m in MODELS]
        self.model_var = ctk.StringVar(value=label_for(self.app.p.model))
        self.model_menu = ctk.CTkOptionMenu(
            t, values=labels, variable=self.model_var, command=self._model_changed,
            font=self.app.fonts.body, dropdown_font=self.app.fonts.body, fg_color=C.field,
            button_color=C.line, button_hover_color=C.line, text_color=C.text,
            dropdown_fg_color=C.panel, dropdown_text_color=C.text,
            dropdown_hover_color=C.field, height=34, corner_radius=8, anchor="w")
        self.model_menu.grid(row=1, column=0, sticky="ew")
        self.model_note = label(self.app, t, self._model_blurb(), muted=True, wrap=330)
        self.model_note.grid(row=2, column=0, sticky="w", pady=(6, 16))

        self.matting_sw = switch(self.app, t, "Alpha matting (hair, fur, soft edges)",
                                 command=self._matting_changed)
        self.matting_sw.grid(row=3, column=0, sticky="w")
        label(self.app, t, "Keeps fine strands but is much slower. Raise the foreground "
                           "value if the subject looks eaten away.", muted=True,
              wrap=330).grid(row=4, column=0, sticky="w", pady=(4, 10))
        self.fg_row = SliderRow(self.app, t, "Foreground threshold",
                                *RANGES["matting_fg"], 240,
                                on_change=lambda v: self._set("matting_fg", int(v)))
        self.fg_row.grid(row=5, column=0, sticky="ew", pady=(0, 8))
        self.bg_row = SliderRow(self.app, t, "Background threshold",
                                *RANGES["matting_bg"], 10,
                                on_change=lambda v: self._set("matting_bg", int(v)))
        self.bg_row.grid(row=6, column=0, sticky="ew", pady=(0, 8))
        self.erode_row = SliderRow(self.app, t, "Matting erode",
                                   *RANGES["matting_erode"], 10,
                                   on_change=lambda v: self._set("matting_erode", int(v)))
        self.erode_row.grid(row=7, column=0, sticky="ew")

        self.post_sw = switch(self.app, t, "Clean up the mask (recommended)",
                              command=lambda: self._set("post_process", bool(self.post_sw.get())))
        self.post_sw.select()
        self.post_sw.grid(row=8, column=0, sticky="w", pady=(14, 0))

        self.remove_btn = button(self.app, t, "Remove background", self.app.remove_bg,
                                 primary=True)
        self.remove_btn.grid(row=9, column=0, sticky="ew", pady=(20, 0))
        self.app.lockable.append(self.remove_btn)

    def _tab_refine(self, t) -> None:
        label(self.app, t, "Leftover edges", muted=True).grid(row=0, column=0, sticky="w",
                                                              pady=(6, 8))
        self.thresh_row = SliderRow(
            self.app, t, "Cut weak edge", *RANGES["mask_threshold"], 0,
            steps=RANGES["mask_threshold"][1],     # one step per alpha value, no snapping
            on_change=lambda v: self._set("mask_threshold", int(v)),
            note="The big one. The model leaves faint haze around a subject and no amount "
                 "of eroding removes it, because it is genuine alpha. Raising this snaps "
                 "it to fully transparent.")
        self.thresh_row.grid(row=1, column=0, sticky="ew", pady=(0, 14))
        self.hard_row = SliderRow(
            self.app, t, "Edge hardness", *RANGES["mask_hardness"], 1.0, fmt="{:.2f}",
            steps=75, on_change=lambda v: self._set("mask_hardness", round(v, 2)),
            note="Steepens the falloff so a smeared, ghostly edge becomes a crisp one.")
        self.hard_row.grid(row=2, column=0, sticky="ew", pady=(0, 14))
        self.speck_row = SliderRow(
            self.app, t, "Remove leftover islands", *RANGES["despeckle"], 0,
            fmt="{:.0f} px", steps=100,
            on_change=lambda v: self._set("despeckle", int(v)),
            note="Deletes detached blobs of background smaller than this. Raise it until "
                 "stray patches disappear, but stay below the size of the subject.")
        self.speck_row.grid(row=3, column=0, sticky="ew", pady=(0, 14))
        self.hole_row = SliderRow(
            self.app, t, "Fill holes in the subject", *RANGES["fill_holes"], 0,
            fmt="{:.0f} px", steps=100,
            on_change=lambda v: self._set("fill_holes", int(v)),
            note="Repairs transparent gaps punched inside the subject.")
        self.hole_row.grid(row=4, column=0, sticky="ew", pady=(0, 18))

        label(self.app, t, "Edges and colour", muted=True).grid(row=5, column=0, sticky="w",
                                                                pady=(0, 8))
        self.shrink_row = SliderRow(
            self.app, t, "Erode / grow edge", *RANGES["shrink"], 0, fmt="{:+.0f} px",
            steps=20, on_change=lambda v: self._set("shrink", int(round(v))),
            note="Positive eats into the subject to kill a halo; negative grows it to "
                 "cover a rim of background the model left behind.")
        self.shrink_row.grid(row=6, column=0, sticky="ew", pady=(0, 14))
        self.feather_row = SliderRow(self.app, t, "Feather", *RANGES["feather"], 0,
                                     fmt="{:.1f} px", steps=40,
                                     on_change=lambda v: self._set("feather", round(v, 2)))
        self.feather_row.grid(row=7, column=0, sticky="ew", pady=(0, 14))
        self.fringe_row = SliderRow(
            self.app, t, "De-fringe", *RANGES["defringe"], 0, fmt="{:.0f} px", steps=8,
            on_change=lambda v: self._set("defringe", int(v)),
            note="Rebuilds the colour of soft edge pixels from opaque neighbours, which "
                 "removes the ring of old background colour.")
        self.fringe_row.grid(row=8, column=0, sticky="ew", pady=(0, 16))

        self.bri_row = SliderRow(self.app, t, "Brightness", *RANGES["brightness"], 1.0,
                                 fmt="{:.2f}", steps=100,
                                 on_change=lambda v: self._set("brightness", round(v, 3)))
        self.bri_row.grid(row=9, column=0, sticky="ew", pady=(0, 10))
        self.con_row = SliderRow(self.app, t, "Contrast", *RANGES["contrast"], 1.0,
                                 fmt="{:.2f}", steps=100,
                                 on_change=lambda v: self._set("contrast", round(v, 3)))
        self.con_row.grid(row=10, column=0, sticky="ew", pady=(0, 10))
        self.sat_row = SliderRow(self.app, t, "Saturation", *RANGES["saturation"], 1.0,
                                 fmt="{:.2f}", steps=100,
                                 on_change=lambda v: self._set("saturation", round(v, 3)))
        self.sat_row.grid(row=11, column=0, sticky="ew", pady=(0, 14))

        self.trim_sw = switch(self.app, t, "Trim transparent border",
                              command=lambda: self._set("trim", bool(self.trim_sw.get())))
        self.trim_sw.grid(row=12, column=0, sticky="w")
        button(self.app, t, "Reset refinements", self._reset_refine).grid(
            row=13, column=0, sticky="ew", pady=(16, 0))

    def _tab_touch(self, t) -> None:
        label(self.app, t, "Fix what the model got wrong", muted=True).grid(
            row=0, column=0, sticky="w", pady=(6, 6))
        self.tool_seg = segment(self.app, t, TOOLS, command=self._tool_changed)
        self.tool_seg.set("Erase")
        self.tool_seg.grid(row=1, column=0, sticky="ew")
        self.tool_note = label(self.app, t, "", muted=True, wrap=330)
        self.tool_note.grid(row=2, column=0, sticky="w", pady=(8, 16))

        self.size_row = SliderRow(self.app, t, "Brush size", *RANGES["brush_size"],
                                  int(self.brush_size),
                                  fmt="{:.0f} px", steps=200,
                                  on_change=self._size_changed,
                                  note="In source-image pixels, so it means the same thing "
                                       "whatever the zoom.")
        self.size_row.grid(row=3, column=0, sticky="ew", pady=(0, 12))
        self.soft_row = SliderRow(self.app, t, "Softness", *RANGES["softness"],
                                  self.brush_softness,
                                  fmt="{:.2f}", steps=20,
                                  on_change=lambda v: setattr(self, "brush_softness",
                                                              round(v, 2)))
        self.soft_row.grid(row=4, column=0, sticky="ew", pady=(0, 18))

        undo_row = ctk.CTkFrame(t, fg_color="transparent")
        undo_row.grid(row=5, column=0, sticky="ew")
        undo_row.grid_columnconfigure((0, 1), weight=1)
        self.undo_btn = button(self.app, undo_row, "Undo", self.app.undo)
        self.undo_btn.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        self.redo_btn = button(self.app, undo_row, "Redo", self.app.redo)
        self.redo_btn.grid(row=0, column=1, sticky="ew", padx=(5, 0))
        button(self.app, t, "Reset all edits", self.app.reset_edits).grid(
            row=6, column=0, sticky="ew", pady=(10, 16))

        self.edit_state = label(self.app, t, "", muted=True, wrap=330)
        self.edit_state.grid(row=7, column=0, sticky="w")

        # Widgets that take `state` directly, and rows that need the slider treatment.
        self.touch_widgets = [self.tool_seg, self.undo_btn, self.redo_btn]
        self.touch_rows = [self.size_row, self.soft_row]
        self._tool_changed(TOOLS[0])

    def _tab_backdrop(self, t) -> None:
        label(self.app, t, "Behind the subject", muted=True).grid(row=0, column=0,
                                                                  sticky="w", pady=(6, 6))
        self.bd_seg = segment(self.app, t, ["Transparent", "Colour", "Image"],
                              command=self._backdrop_mode)
        self.bd_seg.set("Transparent")
        self.bd_seg.grid(row=1, column=0, sticky="ew")

        label(self.app, t, "Colour", muted=True).grid(row=2, column=0, sticky="w",
                                                     pady=(20, 6))
        row = ctk.CTkFrame(t, fg_color="transparent")
        row.grid(row=3, column=0, sticky="ew")
        self.swatch = ctk.CTkButton(row, text="", width=38, height=38, corner_radius=8,
                                    border_width=1, border_color=C.line,
                                    fg_color=self.app.p.bg_color,
                                    hover_color=self.app.p.bg_color, command=self._pick_color)
        self.swatch.grid(row=0, column=0, padx=(0, 10))
        button(self.app, row, "Custom colour", self._pick_color, width=120).grid(row=0,
                                                                                column=1)
        presets = ctk.CTkFrame(t, fg_color="transparent")
        presets.grid(row=4, column=0, sticky="w", pady=(12, 0))
        for i, hexv in enumerate(["#ffffff", "#f1efe9", "#111111", "#2f6fed", "#00b140"]):
            ctk.CTkButton(presets, text="", width=28, height=28, corner_radius=14,
                          border_width=1, border_color=C.line, fg_color=hexv,
                          hover_color=hexv,
                          command=lambda h=hexv: self._set_color(h)).grid(
                row=0, column=i, padx=(0, 8))

        label(self.app, t, "Image", muted=True).grid(row=5, column=0, sticky="w",
                                                    pady=(20, 6))
        button(self.app, t, "Choose backdrop image", self._pick_backdrop_image).grid(
            row=6, column=0, sticky="ew")
        self.bd_file = label(self.app, t, "No image chosen", muted=True, wrap=330)
        self.bd_file.grid(row=7, column=0, sticky="w", pady=(6, 0))

    def _tab_export(self, t) -> None:
        label(self.app, t, "Format", muted=True).grid(row=0, column=0, sticky="w",
                                                     pady=(6, 6))
        self.fmt_seg = segment(self.app, t, FORMATS, command=self._format_changed)
        self.fmt_seg.set(self.app.p.fmt)
        self.fmt_seg.grid(row=1, column=0, sticky="ew")
        label(self.app, t, "PNG and WebP keep transparency. JPG fills it with white.",
              muted=True, wrap=330).grid(row=2, column=0, sticky="w", pady=(6, 14))
        self.q_row = SliderRow(self.app, t, "Quality (WebP, JPG)", *RANGES["quality"], 95,
                               steps=40,
                               on_change=lambda v: self._set("quality", int(v),
                                                             redraw=False))
        self.q_row.grid(row=3, column=0, sticky="ew", pady=(0, 18))

        label(self.app, t, "Upscale", muted=True).grid(row=4, column=0, sticky="w",
                                                      pady=(0, 6))
        self.up_seg = segment(self.app, t, ["1×", "2×", "3×", "4×"],
                              command=self._upscale_changed)
        self.up_seg.set(f"{self.app.p.upscale}×")
        self.up_seg.grid(row=5, column=0, sticky="ew")
        label(self.app, t, "Sharp resize (Lanczos). Adds pixels, not new detail.",
              muted=True, wrap=330).grid(row=6, column=0, sticky="w", pady=(6, 14))
        self.size_lbl = label(self.app, t, "Output size: no image yet")
        self.size_lbl.grid(row=7, column=0, sticky="w")
        self.export_btn = button(self.app, t, "Save image", self.app.save_image, primary=True)
        self.export_btn.grid(row=8, column=0, sticky="ew", pady=(22, 0))
        self.app.lockable.append(self.export_btn)

    # ================================================================== params
    def _set(self, key: str, value, redraw: bool = True) -> None:
        setattr(self.app.p, key, value)
        if redraw:
            self.schedule_refresh()

    def _model_blurb(self) -> str:
        try:
            return spec_for_label(label_for(self.app.p.model)).blurb
        except KeyError:                          # a model this version no longer offers
            return ""

    def _model_changed(self, value: str) -> None:
        spec = spec_for_label(value)
        self.app.p.model = spec.id
        size = f" Download is about {spec.size_mb} MB." if spec.size_mb else ""
        self.model_note.configure(text=spec.blurb + size)
        self.app.save_settings()

    def _matting_changed(self) -> None:
        on = bool(self.matting_sw.get())
        self._set("matting", on, redraw=False)
        for row in (self.fg_row, self.bg_row, self.erode_row):
            row.set_enabled(on)

    def _reset_refine(self) -> None:
        p = self.app.p
        for row, value in ((self.thresh_row, 0), (self.hard_row, 1.0), (self.speck_row, 0),
                           (self.hole_row, 0), (self.shrink_row, 0), (self.feather_row, 0),
                           (self.fringe_row, 0), (self.bri_row, 1.0), (self.con_row, 1.0),
                           (self.sat_row, 1.0)):
            row.set(value)
        self.trim_sw.deselect()
        for key, value in (("mask_threshold", 0), ("mask_hardness", 1.0), ("despeckle", 0),
                           ("fill_holes", 0), ("shrink", 0), ("feather", 0.0),
                           ("defringe", 0), ("brightness", 1.0), ("contrast", 1.0),
                           ("saturation", 1.0), ("trim", False)):
            setattr(p, key, value)
        self.refresh()

    def _backdrop_mode(self, value: str) -> None:
        self.app.p.bg_mode = {"Transparent": "transparent", "Colour": "color",
                              "Image": "image"}[value]
        if self.app.p.bg_mode == "image" and not self.app.p.bg_image:
            self._pick_backdrop_image()
            return
        self.refresh()

    def _set_color(self, hexv: str) -> None:
        self.app.p.bg_color = hexv
        self.swatch.configure(fg_color=hexv, hover_color=hexv)
        self.app.p.bg_mode = "color"
        self.bd_seg.set("Colour")
        self.view_seg.set("Result")
        self.refresh()

    def _pick_color(self) -> None:
        _, hexv = colorchooser.askcolor(color=self.app.p.bg_color, parent=self.app,
                                       title="Backdrop colour")
        if hexv:
            self._set_color(hexv)

    def _pick_backdrop_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose backdrop image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff")])
        if not path:
            if not self.app.p.bg_image:
                self.app.p.bg_mode = "transparent"
                self.bd_seg.set("Transparent")
            return
        self.set_backdrop_image(path)

    def set_backdrop_image(self, path: str) -> None:
        try:
            load_image(path)
        except Exception as e:
            messagebox.showerror("Can't open image",
                                 f"{Path(path).name} couldn't be opened.\n\n{e}", parent=self.app)
            return
        self.app.bgcache.reset()
        self.app.p.bg_image = path
        self.app.p.bg_mode = "image"
        self.bd_file.configure(text=Path(path).name)
        self.bd_seg.set("Image")
        self.view_seg.set("Result")
        self.refresh()

    def _format_changed(self, value: str) -> None:
        self.app.p.fmt = value
        self.q_row.set_enabled(value != "PNG")

    def _upscale_changed(self, value: str) -> None:
        self.app.p.upscale = int(value[0])
        self.refresh()

    # ================================================================== touch up
    def _tool_changed(self, value: str) -> None:
        self.tool = {"Erase": "erase", "Restore": "restore",
                     "Remove leftover": "remove"}[value]
        notes = {
            "erase": "Drag to delete anything that is not the subject. Right-drag restores.",
            "restore": "Drag to bring back something the model removed by mistake, colour "
                       "included. Right-drag erases.",
            "remove": "Click a leftover blob to delete the whole connected piece in one go. "
                      "The main subject is refused — use Erase for that.",
        }
        self.tool_note.configure(text=notes[self.tool])

    def _size_changed(self, value: float) -> None:
        self.brush_size = float(value)

    def _active_tool(self, button: int) -> str:
        """Left button uses the picked tool; the right button inverts it."""
        if self.tool == "remove":
            return "erase" if button == 3 else "remove"
        if button == 3:
            return "restore" if self.tool == "erase" else "erase"
        return self.tool

    def _press(self, x: float, y: float, button: int) -> None:
        app = self.app
        if app.busy or app.edits is None:
            return
        if self.view_seg.get() == "Original":
            app.set_status("You are looking at the original. Switch to Result to edit it.",
                           error=True)
            return
        pos = self.canvas.to_source(x, y)
        if pos is None:
            return

        tool = self._active_tool(button)
        if tool == "remove":
            ok, msg = app.edits.remove_island_at(*pos)
            app.set_status(msg, error=not ok, ok=ok)
            if ok:
                self.on_edit_committed()
            return

        self._dragging = True
        self._drag_last = pos
        self._paint_to(pos, pos, tool)      # a click with no drag still paints a dot

    def _drag(self, x: float, y: float, button: int) -> None:
        if not self._dragging or self._drag_last is None:
            return
        pos = self.canvas.to_source(x, y)
        if pos is None:
            return
        self._paint_to(self._drag_last, pos, self._active_tool(button))
        self._drag_last = pos
        self.canvas.show_ring(x, y, self.brush_size * self.canvas.display_per_source(),
                              self._ring_color(button))

    def _release(self, x: float, y: float, button: int) -> None:
        if not self._dragging:
            return
        self._dragging = False
        self._drag_last = None
        if self.app.edits is not None and self.app.edits.end_stroke():
            self.on_edit_committed()

    def _paint_to(self, a: tuple[float, float], b: tuple[float, float], tool: str) -> None:
        """Paint at full resolution, and mirror it onto the preview for live feedback."""
        app = self.app
        erase = tool == "erase"
        app.edits.move(Segment(a[0], a[1], b[0], b[1], self.brush_size,
                               self.brush_softness, erase))
        if self.prev_alpha is not None:
            # Painting the proxy is what keeps a drag responsive: re-rendering the full
            # 12 MP layer on every mouse event would be unusable. The full-resolution
            # layer is authoritative and is resynced when the stroke ends.
            s = self.pscale
            paint_segment(
                self.prev_alpha, self.prev_rgb, self.prev_orig,
                Segment(a[0] * s, a[1] * s, b[0] * s, b[1] * s,
                        max(1.0, self.brush_size * s), self.brush_softness, erase),
                (self.prev_alpha.width, self.prev_alpha.height))
        self.schedule_refresh()

    def _ring_color(self, button: int):
        return pick(C.brush) if self._active_tool(button) == "erase" else pick(C.ok)

    def _hover(self, x: float, y: float) -> None:
        app = self.app
        if app.busy or app.edits is None or self.view_seg.get() == "Original":
            self.canvas.hide_ring()
            return
        if self.tool == "remove":
            self.canvas.show_ring(x, y, 6, pick(C.danger))
            return
        self.canvas.show_ring(x, y, self.brush_size * self.canvas.display_per_source(),
                              self._ring_color(1))

    def sync_touch(self) -> None:
        """The touch-up tools only make sense once a cutout exists."""
        edits = self.app.edits
        ready = edits is not None
        for w in self.touch_widgets:
            w.configure(state="normal" if ready else "disabled")
        for row in self.touch_rows:
            row.set_enabled(ready)
        if not ready:
            self.edit_state.configure(text="Remove the background first.",
                                      text_color=C.muted)
            return
        count = edits.count
        if count:
            plural = "" if count == 1 else "s"
            self.edit_state.configure(
                text=f"{count} edit{plural} applied — Ctrl+Z steps back through them.",
                text_color=C.muted)
        else:
            self.edit_state.configure(text="No manual edits yet.", text_color=C.muted)

    # ================================================================== preview
    def _make_image_proxies(self) -> None:
        app = self.app
        w, h = app.original.size
        self.pscale = min(1.0, PREVIEW_MAX / max(w, h))
        size = (max(1, round(w * self.pscale)), max(1, round(h * self.pscale)))
        if self.pscale < 1:
            self.prev_orig = app.original.convert("RGB").resize(size, Image.LANCZOS)
        else:
            self.prev_orig = app.original.convert("RGB")

    def _make_edit_proxies(self) -> None:
        app = self.app
        if app.edits is None:
            self.prev_rgb = self.prev_alpha = None
            return
        w, h = app.edits.size
        size = (max(1, round(w * self.pscale)), max(1, round(h * self.pscale)))
        if self.pscale < 1:
            self.prev_rgb = app.edits.rgb.resize(size, Image.LANCZOS)
            self.prev_alpha = app.edits.alpha.resize(size, Image.LANCZOS)
        else:
            self.prev_rgb = app.edits.rgb.copy()
            self.prev_alpha = app.edits.alpha.copy()

    def on_new_image(self) -> None:
        """A different file was opened."""
        self._dragging = False
        self._drag_last = None
        self._make_image_proxies()
        self._make_edit_proxies()
        self.sync_touch()
        self.canvas.set_image(None)
        self.refresh()

    def on_new_cutout(self) -> None:
        """The background was just removed (or the model re-run)."""
        self._make_edit_proxies()
        self.sync_touch()
        self.view_seg.set("Result")
        self.refresh()

    def on_edit_committed(self) -> None:
        """Resync the proxies from the full-resolution layer, which is authoritative."""
        self._make_edit_proxies()
        self.sync_touch()
        self.refresh()

    def schedule_refresh(self) -> None:
        if self._refresh_job:
            self.after_cancel(self._refresh_job)
        self._refresh_job = self.after(40, self.refresh)

    def refresh(self) -> None:
        self._refresh_job = None
        app = self.app
        if app.original is None:
            self.canvas.set_image(None)
            self._update_size_label(None)
            return

        r = None
        if self.prev_alpha is not None:
            try:
                r = render(self.prev_rgb, self.prev_alpha, app.p, scale=self.pscale,
                           cache=app.bgcache)
            except Exception as e:
                app.p.bg_mode = "transparent"
                self.bd_seg.set("Transparent")
                app.set_status(f"Backdrop failed: {e}", error=True)

        view = self.view_seg.get()
        if view == "Original" or r is None:
            self.canvas.set_image(self.prev_orig, crop=None, pscale=self.pscale)
        elif view == "Mask":
            self.canvas.set_image(mask_preview(r.alpha) if r.alpha is not None else r.img,
                                  crop=r.crop, pscale=self.pscale)
        else:
            self.canvas.set_image(r.img, crop=r.crop, pscale=self.pscale)
        self._update_size_label(r)

    def _update_size_label(self, r) -> None:
        if r is not None:
            w, h = r.img.size
            approx = "about " if r.crop else ""
        elif self.prev_orig is not None:
            w, h = self.prev_orig.size
            approx = ""
        else:
            self.size_lbl.configure(text="Output size: no image yet")
            return
        up = self.app.p.upscale
        self.size_lbl.configure(
            text=f"Output size: {approx}{round(w / self.pscale) * up} × "
                 f"{round(h / self.pscale) * up} px")

    # ================================================================== sync
    def load_params(self, p) -> None:
        """Push a Params object into every widget. Used once, at startup."""
        # Anything from a settings file is untrusted: a value written by an older version
        # must be corrected here rather than blowing up later inside a file dialog.
        if p.model not in MODEL_BY_ID:
            p.model = DEFAULT_MODEL
        self.model_var.set(label_for(p.model))
        self.model_note.configure(text=self._model_blurb())

        if p.matting:
            self.matting_sw.select()
        else:
            self.matting_sw.deselect()
        for row, value in ((self.fg_row, p.matting_fg), (self.bg_row, p.matting_bg),
                           (self.erode_row, p.matting_erode)):
            row.set(value)
            row.set_enabled(p.matting)
        if p.post_process:
            self.post_sw.select()
        else:
            self.post_sw.deselect()

        for row, value in ((self.thresh_row, p.mask_threshold),
                           (self.hard_row, p.mask_hardness),
                           (self.speck_row, p.despeckle), (self.hole_row, p.fill_holes),
                           (self.shrink_row, p.shrink), (self.feather_row, p.feather),
                           (self.fringe_row, p.defringe),
                           (self.bri_row, p.brightness), (self.con_row, p.contrast),
                           (self.sat_row, p.saturation), (self.q_row, p.quality)):
            row.set(value)
        self.q_row.set_enabled(p.fmt != "PNG")

        if p.trim:
            self.trim_sw.select()
        else:
            self.trim_sw.deselect()

        # Values coming from a settings file are untrusted: a stale entry must not raise
        # inside the option menu, so only apply what the widget actually offers.
        if p.fmt not in FORMATS:
            self.app.p.fmt = "PNG"
        self.fmt_seg.set(self.app.p.fmt)
        self.q_row.set_enabled(self.app.p.fmt != "PNG")
        if p.upscale in (1, 2, 3, 4):
            self.up_seg.set(f"{p.upscale}×")
        else:
            self.app.p.upscale = 1
        if p.bg_mode not in ("transparent", "color", "image"):
            self.app.p.bg_mode = "transparent"
        self.bd_seg.set({"transparent": "Transparent", "color": "Colour",
                         "image": "Image"}[self.app.p.bg_mode])
        self.swatch.configure(fg_color=p.bg_color, hover_color=p.bg_color)
        if p.bg_image and Path(p.bg_image).is_file():
            self.bd_file.configure(text=Path(p.bg_image).name)
        elif self.app.p.bg_mode == "image":
            # The remembered backdrop is gone; fall back rather than showing a blank panel.
            self.app.p.bg_mode = "transparent"
            self.bd_seg.set("Transparent")

        self.sync_touch()          # nothing to touch up until a cutout exists
