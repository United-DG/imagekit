"""Smoke tests for the real window.

Nobody can *look* at the GUI from a test, but building it catches the whole class of
mistakes that would otherwise only surface on launch: a mistyped keyword, a missing
widget, a KeyError inside a tab, a slider whose bounds disagree with the CLI.

The brush tests matter most. They drive the same `_press`/`_drag`/`_release` handlers the
mouse does, so a broken coordinate transform fails here instead of in front of the user.
"""
from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw

from rmbg.edit import EditLayer
from rmbg.params import Params, brush_range, from_dict
from rmbg.pipeline import render
from tests.support import scene

try:                                             # a window needs a display of some kind
    import tkinter
    _probe = tkinter.Tk()
    _probe.destroy()
    HAVE_DISPLAY = True
except Exception:                                # pragma: no cover - headless CI
    HAVE_DISPLAY = False

try:                                             # imported for the tests, not the probe
    import customtkinter as ctk
    from rmbg.gui.single import TABS
    from rmbg.gui.widgets import SliderRow
except Exception:                                # pragma: no cover - headless CI
    ctk = TABS = SliderRow = None

needs_display = unittest.skipUnless(HAVE_DISPLAY, "no display for tkinter")


@needs_display
class Window(unittest.TestCase):
    def setUp(self):
        from rmbg.gui.app import App
        # Building the window prints nothing, but `_open_from_command_line` can be tempted
        # to open a file named after a test argument; sys.argv is left alone on purpose so
        # that path is exercised too.
        with redirect_stdout(io.StringIO()):
            self.app = App()
        self.app.geometry("1280x820")
        self.app.update()
        self.canvas = self.app.single.canvas

    def tearDown(self):
        try:
            self.app.destroy()
        except Exception:                        # pragma: no cover - already gone
            pass

    # ------------------------------------------------------------------ helpers
    def load(self, size=(200, 150), box=(80, 60, 120, 90)):
        """A centred subject: source (100, 75) is the middle of the image and of the box."""
        orig, cut = scene(size[0], size[1], box)
        self.app.set_original(orig, "", "test")
        self.app.edits = EditLayer(orig, cut)
        self.app.single.on_new_cutout()
        self.app.update()
        self.canvas.redraw()
        return orig, cut

    def to_canvas(self, sx, sy):
        """Source pixel -> canvas coordinate, by inverting the stored transform."""
        scale, ox, oy, crop_x, crop_y, pscale = self.canvas.transform
        return ((sx * pscale - crop_x) * scale + ox, (sy * pscale - crop_y) * scale + oy)

    def centre(self):
        return self.canvas.canvas.winfo_width() // 2, self.canvas.canvas.winfo_height() // 2

    # ------------------------------------------------------------------ structure
    def test_it_builds_with_every_tab(self):
        for name in ("Remove", "Refine", "Touch up", "Backdrop", "Export"):
            self.assertIsNotNone(self.app.single.tabs.tab(name))

    def test_both_pages_switch_without_error(self):
        self.app._show_page("Batch")
        self.app.update()
        self.app._show_page("Single image")
        self.app.update()

    def test_the_brush_starts_disabled_until_there_is_a_cutout(self):
        self.assertIsNone(self.app.edits)
        self.app.single.sync_touch()
        self.assertEqual(self.app.single.tool_seg.cget("state"), "disabled")
        self.load()
        self.assertEqual(self.app.single.tool_seg.cget("state"), "normal")

    def test_the_touch_tab_counts_the_edits(self):
        self.load()
        self.app.single.tool = "erase"
        self.app.single.brush_size = 10
        cx, cy = self.centre()
        self.app.single._press(cx, cy, 1)
        self.app.single._release(cx, cy, 1)
        self.app.update()
        self.assertIn("1 edit", self.app.single.edit_state.cget("text"))

    def test_the_theme_can_be_toggled(self):
        for on in (False, True, True, False):
            if on:
                self.app.theme_sw.select()
            else:
                self.app.theme_sw.deselect()
            self.app._toggle_theme()
            self.app.update()

    # ------------------------------------------------------------------ params
    def test_params_reach_the_sliders(self):
        p = Params(mask_threshold=45, despeckle=300, shrink=-6, feather=3.0, fmt="JPG",
                   upscale=2)
        self.app.p = p
        self.app.single.load_params(p)
        self.app.update()
        self.assertEqual(self.app.single.thresh_row.get(), 45)
        self.assertEqual(self.app.single.speck_row.get(), 300)
        self.assertEqual(self.app.single.shrink_row.get(), -6)
        self.assertAlmostEqual(self.app.single.feather_row.get(), 3.0)
        self.assertEqual(self.app.single.fmt_seg.get(), "JPG")
        self.assertEqual(self.app.single.up_seg.get(), "2×")

    def test_a_stale_settings_file_does_not_crash_the_window(self):
        """A settings file written by an older version is untrusted input."""
        p = from_dict({"fmt": "TIFF", "upscale": 99, "bg_mode": "nonsense",
                       "quality": "not a number", "model": "deleted-model"})
        self.app.p = p
        self.app.single.load_params(p)
        self.app.update()

    def test_matting_sliders_are_disabled_until_matting_is_on(self):
        p = Params(matting=False)
        self.app.p = p
        self.app.single.load_params(p)
        self.app.update()
        self.assertEqual(self.app.single.fg_row.slider.cget("state"), "disabled")
        p.matting = True
        self.app.p = p
        self.app.single.load_params(p)
        self.app.update()
        self.assertEqual(self.app.single.fg_row.slider.cget("state"), "normal")

    def test_reset_refinements_puts_every_slider_back(self):
        self.load()
        self.app.p.mask_threshold = 200
        self.app.p.despeckle = 900
        self.app.single.load_params(self.app.p)
        self.app.single._reset_refine()
        self.app.update()
        self.assertEqual(self.app.p.mask_threshold, 0)
        self.assertEqual(self.app.p.despeckle, 0)
        self.assertEqual(self.app.single.thresh_row.get(), 0)

    def test_settings_are_written_where_they_belong(self):
        with tempfile.TemporaryDirectory() as d:
            target = Path(d) / "sub" / "settings.json"
            self.app.p.mask_threshold = 77
            with patch("rmbg.gui.app.settings_path", lambda: target):
                self.app.save_settings()
            data = json.loads(target.read_text(encoding="utf-8"))
            self.assertEqual(data["params"]["mask_threshold"], 77)
            self.assertIn("dark", data)

    # ------------------------------------------------------------------ mapping
    def test_a_click_lands_on_the_pixel_under_the_cursor(self):
        """Without this the eraser paints somewhere other than where it is pointing."""
        self.load()
        x, y = self.canvas.to_source(*self.centre())
        self.assertAlmostEqual(x, 100, delta=2.5)
        self.assertAlmostEqual(y, 75, delta=2.5)

    def test_trimming_does_not_shift_the_brush(self):
        """Trimming slides the image under the cursor, so the crop origin has to be added
        back before the proxy scale is divided out. Get it wrong and the brush only
        misbehaves with trimming on, which is the worst kind of bug to find by hand."""
        self.load()
        self.app.p.trim = True
        self.app.single.refresh()
        self.app.update()
        x, y = self.canvas.to_source(*self.centre())
        self.assertAlmostEqual(x, 100, delta=3.0)
        self.assertAlmostEqual(y, 75, delta=3.0)

    def test_round_tripping_a_source_point_through_the_transform(self):
        self.load()
        for sx, sy in ((80, 60), (100, 75), (120, 90)):
            cx, cy = self.to_canvas(sx, sy)
            back = self.canvas.to_source(cx, cy)
            self.assertAlmostEqual(back[0], sx, delta=1.0)
            self.assertAlmostEqual(back[1], sy, delta=1.0)

    def test_a_click_outside_the_image_is_ignored(self):
        self.load()
        self.assertIsNotNone(self.canvas.to_source(-5000, -5000))   # the maths still answers
        self.app.single._press(-5000, -5000, 1)                     # but nothing should break
        self.assertEqual(self.app.edits.count, 0)

    # ------------------------------------------------------------------ navigation
    def test_the_wheel_zooms_about_the_cursor(self):
        """The whole reason to zoom about the cursor: the pixel under it must not move.

        Anchor the zoom anywhere else — the middle of the canvas, say — and the thing being
        examined slides away as it grows, which makes the wheel useless for the one job it
        exists for: deciding whether an edge is clean.
        """
        self.load()
        self.canvas.set_zoom(2.5)
        self.app.update()
        spot = (140, 120)
        before = self.canvas.to_source(*spot)
        self.canvas.zoom_by(1.25, anchor=spot)
        self.app.update()
        after = self.canvas.to_source(*spot)
        self.assertAlmostEqual(after[0], before[0], delta=1.0)
        self.assertAlmostEqual(after[1], before[1], delta=1.0)
        self.assertGreater(self.canvas.zoom, 2.5, "the wheel did not zoom at all")

    def test_the_transform_still_round_trips_while_zoomed_and_panned(self):
        """Zoom and pan only change how `scale` and the origin are computed; the mapping is
        still the same 6-tuple. If it were not, the brush, the ring and the size readout would
        all have to learn about zooming — and the brush would land somewhere else."""
        self.load()
        self.canvas.set_zoom(6.0)
        self.canvas.pan_by(-140, 90)
        self.app.update()
        self.assertEqual(len(self.canvas.transform), 6)
        for sx, sy in ((80, 60), (100, 75), (119, 89)):
            cx, cy = self.to_canvas(sx, sy)
            back = self.canvas.to_source(cx, cy)
            self.assertAlmostEqual(back[0], sx, delta=1.0)
            self.assertAlmostEqual(back[1], sy, delta=1.0)

    def test_fit_undoes_the_zoom_and_the_pan(self):
        self.load()
        self.canvas.set_zoom(4.0)
        self.canvas.pan_by(-200, 120)
        self.canvas.fit()
        self.app.update()
        self.assertAlmostEqual(self.canvas.zoom, 1.0)
        self.assertAlmostEqual(self.canvas.pan[0], 0.0, places=6)
        self.assertAlmostEqual(self.canvas.pan[1], 0.0, places=6)
        x, y = self.canvas.to_source(*self.centre())
        self.assertAlmostEqual(x, 100, delta=2.5)
        self.assertAlmostEqual(y, 75, delta=2.5)

    def test_the_image_cannot_be_panned_out_of_reach(self):
        """A pan that can lose the image makes the hand a trap rather than a tool."""
        self.load()
        self.canvas.set_zoom(8.0)
        for _ in range(20):
            self.canvas.pan_by(400, 300)
        self.app.update()
        scale, ox, oy, _cx, _cy, _ps = self.canvas.transform
        im = self.canvas._img
        cw = self.canvas.canvas.winfo_width()
        ch = self.canvas.canvas.winfo_height()
        self.assertLess(ox, cw, "the image was pushed off the right edge")
        self.assertLess(oy, ch, "the image was pushed off the bottom edge")
        self.assertGreater(ox + im.width * scale, 0, "the image was pushed off the left edge")
        self.assertGreater(oy + im.height * scale, 0, "the image was pushed off the top edge")

    def test_a_magnified_view_is_rendered_at_the_resolution_the_screen_shows(self):
        """Past proxy resolution the preview used to be an upscaled proxy.

        On this 4000 px photo that is 2.9x of blur at 100%, which makes judging an edge — the
        one thing this app is for — actively misleading. So the proxy is sized to the
        magnification instead: as fine as the screen shows, never finer than the source.
        """
        self.load(size=(4000, 3000))
        page = self.app.single
        self.assertLess(page.pscale, 1.0, "a 4000 px image should start on a proxy")
        self.canvas.set_zoom_percent(100.0)
        page._apply_detail()                   # what the deferred call would have done
        self.app.update()
        self.assertAlmostEqual(page.pscale, 1.0, delta=0.02,
                               msg="100% is still showing an upscaled proxy")
        self.assertAlmostEqual(self.canvas.zoom_percent(), 100.0, delta=1.0)
        # and back out again: full resolution must not be paid for when it is not on screen
        self.canvas.fit()
        page._apply_detail()
        self.app.update()
        self.assertAlmostEqual(page.pscale, 1400 / 4000, delta=0.02)

    def test_the_zoom_readout_is_the_magnification_of_the_source(self):
        """100% has to mean one screen pixel per source pixel, not per proxy pixel, or the
        figure is a lie the moment the image is bigger than the proxy."""
        self.load(size=(4000, 3000))
        self.canvas.fit()
        self.app.update()
        fitted = self.canvas.zoom_percent()
        self.assertGreater(fitted, 0.0)
        self.assertLess(fitted, 100.0, "a 4000 px image cannot be at 100% while fitted")
        self.canvas.set_zoom_percent(100.0)
        self.app.update()
        self.assertAlmostEqual(self.canvas.zoom_percent(), 100.0, delta=1.0)

    def test_the_magnified_view_shows_the_export_pixels(self):
        """Parity: at 100% the preview has to *be* the export, not an impression of it.

        Preview and export run the same pipeline on the same pixels; the only thing the zoom
        changes is the resolution that pipeline runs at. So the pixel under a given point on
        screen and the pixel an export writes at the same place must be equal. If they drift,
        an edge examined at 100% is not the edge being saved, and looking closer tells you
        nothing — which is the whole reason to look closer.
        """
        self.load(size=(4000, 3000))
        page = self.app.single
        self.canvas.set_zoom_percent(100.0)
        page._apply_detail()                   # what the deferred call would have done
        self.app.update()
        self.assertAlmostEqual(page.pscale, 1.0, delta=0.02,
                               msg="the preview is still a proxy, so this compares nothing")
        export = render(self.app.edits.rgb.copy(), self.app.edits.alpha.copy(),
                        self.app.p, scale=1.0)
        self.assertIsNone(export.crop)
        for sx, sy in ((900, 700), (2000, 1500), (3100, 2400)):
            x, y = self.to_canvas(sx, sy)
            back = self.canvas.to_source(x, y)
            self.assertAlmostEqual(back[0], sx, delta=1.0)
            self.assertEqual(page.canvas._img.getpixel((sx, sy)), export.img.getpixel((sx, sy)),
                             f"the preview differs from the export at source ({sx}, {sy})")

    # ------------------------------------------------------------------ brush
    def test_the_brush_slider_spends_its_travel_where_the_sizes_are_useful(self):
        """The arithmetic behind "if u move it even a small distance it becomes too big".

        On a 1200 px photo the radii anyone actually paints with are about 6 to 86 px. On the
        old linear 2..400 track that band was slider positions 1%..21%, and everything past it
        was a brush wider than a quarter of the picture — so the only usable place to leave the
        handle was just off its stop. The row is per-image and geometric now.
        """
        lo, hi = brush_range(1200)
        row = SliderRow(self.app, self.app, "size", lo, hi, 24, steps=None, curve="log")
        try:
            linear = (86 - 6) / (400 - 2)          # what the old fixed linear row gave: 0.20
            self.assertAlmostEqual(linear, 0.20, places=2,
                                   msg="the old range no longer looks the way it is described")
            band = row._to_pos(86) - row._to_pos(6)
            self.assertGreater(band, 0.5,
                               f"the useful band is only {band:.0%} of the travel")
            # and the default has to land somewhere it can be nudged in both directions
            self.assertGreater(row._to_pos(24), 0.25)
            self.assertLess(row._to_pos(24), 0.8)
        finally:
            row.destroy()

    def test_a_curved_slider_round_trips_its_value(self):
        """`load_params` writes values in and `get` reads them back, so the curve has to be
        exactly invertible rather than merely close."""
        lo, hi = brush_range(1200)
        row = SliderRow(self.app, self.app, "size", lo, hi, 24, steps=None, curve="log")
        try:
            for value in (2, 6, 12, 24, 37, 86, 239):
                row.set(value)
                self.assertAlmostEqual(row.get(), value, delta=0.01)
        finally:
            row.destroy()

    def test_the_brush_bounds_follow_the_image(self):
        """How large a sensible brush is depends on the image, so opening one re-ranges it."""
        self.load()
        self.assertEqual(self.app.single.size_row.bounds(), brush_range(200))
        self.load(size=(1000, 750))
        self.assertEqual(self.app.single.size_row.bounds(), brush_range(1000))

    def test_the_brush_ring_matches_what_it_paints(self):
        """Ring, readout and paint are three expressions of one quantity.

        `brush_size` goes to `show_ring` as a display radius and to the brush as a paint radius
        in source pixels. If either conversion is wrong the cursor lies about the brush, which
        is the kind of fault that makes a tool feel broken while nothing actually raises.
        """
        self.load()
        page = self.app.single
        page.tool = "erase"
        page.brush_softness = 0.0
        page.brush_size = 12
        cx, cy = self.centre()
        page._press(cx, cy, 1)
        page._release(cx, cy, 1)
        self.app.update()

        alpha = self.app.edits.alpha
        self.assertEqual(alpha.getpixel((100, 75)), 0, "the click painted nothing")
        reach = max(r for r in range(1, 60) if alpha.getpixel((100 + r, 75)) == 0)
        self.assertAlmostEqual(reach, 12, delta=1.0,
                               msg="the painted radius is not the brush size")

        page._hover(*self.to_canvas(100, 75))
        coords = self.canvas.canvas.coords(self.canvas._ring)
        self.assertTrue(coords, "no brush ring was drawn")
        shown = (coords[2] - coords[0]) / 2.0
        self.assertAlmostEqual(shown, 12 * self.canvas.display_per_source(), delta=1.0,
                               msg="the ring does not show the size the brush paints")

    # ------------------------------------------------------------------ panel layout
    def test_every_tab_body_scrolls(self):
        """The Refine tab wanted ~920 px of content in the ~690 px the fixed panel offers, so
        brightness, contrast, saturation, trim and reset sat past the bottom edge and were
        unreachable at any window size — which is how a working slider came to look missing."""
        for name in TABS:
            self.assertIsInstance(self.app.single._bodies[name], ctk.CTkScrollableFrame,
                                  f"the {name} tab has no scrolling body")

    def test_the_refine_tab_can_be_scrolled_to_its_last_controls(self):
        self.load()
        self.app.geometry("1080x740")            # minsize: the worst case for the panel
        self.app.single.tabs.set("Refine")
        self.app.update()
        body = self.app.single._bodies["Refine"]
        self.assertNotEqual(body._parent_canvas.yview(), (0.0, 1.0),
                            "the Refine tab now fits — if its content shrank, this test needs "
                            "a new example of a tab that overflows")
        self.assertTrue(body._scrollbar.winfo_ismapped(),
                        "the Refine tab overflows but offers no scrollbar to reach the rest")
        bottom = body._parent_canvas.bbox("all")[3]
        for row in (self.app.single.bri_row, self.app.single.con_row, self.app.single.sat_row,
                    self.app.single.trim_sw):
            self.assertLessEqual(row.winfo_y() + row.winfo_height(), bottom + 1,
                                 "a control sits outside the scrollable region")

    def test_a_tab_that_fits_does_not_carry_a_scrollbar(self):
        self.load()
        self.app.single.tabs.set("Backdrop")
        self.app.update()
        body = self.app.single._bodies["Backdrop"]
        self.assertEqual(body._parent_canvas.yview(), (0.0, 1.0),
                         "the Backdrop tab no longer fits; pick another short tab")
        self.assertFalse(body._scrollbar.winfo_ismapped(),
                         "a tab that fits should not show a scrollbar")

    def test_erasing_clears_the_source_pixel_under_the_cursor(self):
        self.load()
        self.assertEqual(self.app.edits.alpha.getpixel((100, 75)), 255)
        self.app.single.tool = "erase"
        self.app.single.brush_size = 10
        cx, cy = self.centre()
        self.app.single._press(cx, cy, 1)
        self.app.single._release(cx, cy, 1)
        self.app.update()
        self.assertEqual(self.app.edits.alpha.getpixel((100, 75)), 0)
        self.assertEqual(self.app.edits.count, 1)

    def test_a_whole_drag_is_one_undo_step(self):
        self.load()
        self.app.single.tool = "erase"
        self.app.single.brush_size = 8
        cw, cy = self.canvas.canvas.winfo_width(), self.centre()[1]
        self.app.single._press(cw // 2 - 30, cy, 1)
        for dx in range(-25, 26, 5):
            self.app.single._drag(cw // 2 + dx, cy, 1)
        self.app.single._release(cw // 2 + 25, cy, 1)
        self.app.update()
        self.assertEqual(self.app.edits.count, 1)

    def test_undo_puts_the_pixel_back(self):
        self.load()
        self.app.single.tool = "erase"
        self.app.single.brush_size = 10
        cx, cy = self.centre()
        self.app.single._press(cx, cy, 1)
        self.app.single._release(cx, cy, 1)
        self.app.undo()
        self.app.update()
        self.assertEqual(self.app.edits.alpha.getpixel((100, 75)), 255)

    def test_the_right_button_restores_instead_of_erasing(self):
        """Right-drag is the undo-of-a-single-mistake escape hatch, so it has to invert."""
        orig, cut = self.load()
        self.app.single.tool = "erase"
        self.app.single.brush_size = 10
        self.assertEqual(self.app.edits.alpha.getpixel((30, 30)), 0)
        cx, cy = self.to_canvas(30, 30)
        self.app.single._press(cx, cy, 3)
        self.app.single._release(cx, cy, 3)
        self.app.update()
        self.assertEqual(self.app.edits.alpha.getpixel((30, 30)), 255)
        self.assertEqual(self.app.edits.rgb.getpixel((30, 30)),
                         self.app.edits.orig_rgb.getpixel((30, 30)))

    def test_click_to_remove_deletes_a_leftover_but_not_the_subject(self):
        self.load()
        ImageDraw.Draw(self.app.edits.alpha).rectangle((20, 20, 30, 30), fill=255)
        self.app.single.on_edit_committed()
        self.app.update()
        self.app.single.tool = "remove"

        cx, cy = self.to_canvas(25, 25)
        self.app.single._press(cx, cy, 1)
        self.app.update()
        self.assertEqual(self.app.edits.alpha.getpixel((25, 25)), 0)

        cx, cy = self.to_canvas(100, 75)             # the subject itself
        self.app.single._press(cx, cy, 1)
        self.app.update()
        self.assertEqual(self.app.edits.alpha.getpixel((100, 75)), 255)
        self.assertIn("subject", self.app.status.cget("text"))

    def test_painting_is_refused_while_looking_at_the_original(self):
        self.load()
        self.app.single.view_seg.set("Original")
        self.app.single.refresh()
        self.app.update()
        cx, cy = self.centre()
        self.app.single._press(cx, cy, 1)
        self.assertEqual(self.app.edits.count, 0)
        self.assertIn("original", self.app.status.cget("text"))

    # ------------------------------------------------------------------ views
    def test_every_view_renders(self):
        self.load()
        for view in ("Original", "Result", "Mask"):
            self.app.single.view_seg.set(view)
            self.app.single.refresh()
            self.app.update()

    def test_the_preview_photo_belongs_to_the_canvas_interpreter(self):
        """The preview photo must be created *by* the canvas that draws it.

        A master-less `ImageTk.PhotoImage` goes to tkinter's process-wide default root, and
        that title is only claimed while it is None — so in a process that has built and
        dropped a window before, the photo can land in a different Tcl interpreter. It then
        does not exist where it is drawn and `create_image` raises
        `image "pyimageN" does not exist`, which is a failure with no visible cause in the
        drawing code. Naming the canvas as master is what rules it out.
        """
        self.load()
        self.assertTrue(self.canvas.canvas.find_withtag("img"),
                        "the preview should have drawn an image item")
        inner = getattr(self.canvas._photo, "_PhotoImage__photo", None)
        self.assertIsNotNone(inner, "Pillow renamed PhotoImage.__photo")
        self.assertIs(inner.tk, self.canvas.canvas.tk,
                      "the preview photo lives in another interpreter than the canvas")

    def test_a_mask_preview_exists_for_the_mask_view(self):
        self.load()
        self.app.single.view_seg.set("Mask")
        self.app.single.refresh()
        self.app.update()
        self.assertIsNotNone(self.canvas.transform)
        self.assertIsNotNone(self.canvas._img)

    def test_a_backdrop_colour_renders(self):
        self.load()
        self.app.single._set_color("#00ff00")
        self.app.update()
        self.assertEqual(self.app.p.bg_color, "#00ff00")
        self.assertEqual(self.app.p.bg_mode, "color")

    def test_the_output_size_label_tracks_the_upscale(self):
        self.load()
        self.app.p.upscale = 2
        self.app.single.refresh()
        self.app.update()
        self.assertIn("400", self.app.single.size_lbl.cget("text"))

    # ------------------------------------------------------------------ output
    def test_output_layers_come_from_the_edited_layer(self):
        self.load()
        rgb, alpha = self.app.output_layers()
        self.assertEqual(rgb.size, (200, 150))
        self.assertEqual(alpha.size, (200, 150))
        self.assertIs(alpha, self.app.edits.alpha)

    def test_output_layers_are_the_original_before_any_cutout(self):
        orig, _cut = scene()
        self.app.set_original(orig, "", "test")
        rgb, alpha = self.app.output_layers()
        self.assertEqual(rgb.mode, "RGB")
        self.assertIsNone(alpha)

    def test_re_running_the_model_does_not_lose_the_touch_up(self):
        """The whole reason `rebuild` keeps a snapshot: a stray Ctrl+R must not cost an
        hour of brush work with no way back."""
        orig, cut = self.load()
        self.app.single.tool = "erase"
        self.app.single.brush_size = 10
        cx, cy = self.centre()
        self.app.single._press(cx, cy, 1)
        self.app.single._release(cx, cy, 1)
        edited = self.app.edits.alpha.tobytes()

        self.app.edits.rebuild(orig, cut)            # what `remove_bg`'s callback does
        self.app.single.on_new_cutout()
        self.app.update()
        self.assertNotEqual(self.app.edits.alpha.tobytes(), edited)

        self.app.undo()
        self.app.update()
        self.assertEqual(self.app.edits.alpha.tobytes(), edited)


if __name__ == "__main__":                       # pragma: no cover
    unittest.main()
