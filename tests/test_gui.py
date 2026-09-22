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
from rmbg.params import Params, from_dict
from tests.support import scene

try:                                             # a window needs a display of some kind
    import tkinter
    _probe = tkinter.Tk()
    _probe.destroy()
    HAVE_DISPLAY = True
except Exception:                                # pragma: no cover - headless CI
    HAVE_DISPLAY = False

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

    # ------------------------------------------------------------------ brush
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
