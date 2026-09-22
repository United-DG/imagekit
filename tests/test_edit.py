"""The brush, click-to-remove, and the undo journal."""
from __future__ import annotations

import unittest

from PIL import Image, ImageDraw

import rmbg.edit as editmod
from rmbg.edit import EditLayer, Segment, _densify, paint_segment
from rmbg.mask import HAVE_SCIPY
from tests.support import scene

needs_scipy = unittest.skipUnless(HAVE_SCIPY, "scipy is required for island removal")


def new_layer(size=(60, 60), box=(20, 20, 40, 40), softness=0.0, erase=True, radius=6,
              at=(30, 30)) -> EditLayer:
    orig, cut = scene(size[0], size[1], box)
    layer = EditLayer(orig, cut)
    layer.move(Segment(at[0], at[1], at[0], at[1], radius, softness, erase))
    return layer


class Painting(unittest.TestCase):
    def test_a_fresh_layer_matches_the_model_output(self):
        orig, cut = scene()
        layer = EditLayer(orig, cut)
        self.assertFalse(layer.has_edits)
        self.assertEqual(layer.count, 0)
        self.assertEqual(layer.alpha.tobytes(), cut.getchannel("A").tobytes())

    def test_erase_clears_alpha_under_the_brush(self):
        layer = new_layer()
        self.assertTrue(layer.end_stroke())
        self.assertEqual(layer.alpha.getpixel((30, 30)), 0)
        self.assertEqual(layer.count, 1)

    def test_erase_leaves_the_rest_of_the_subject_alone(self):
        layer = new_layer()
        layer.end_stroke()
        self.assertEqual(layer.alpha.getpixel((22, 22)), 255)
        self.assertEqual(layer.alpha.getpixel((38, 38)), 255)

    def test_an_empty_stroke_records_nothing(self):
        orig, cut = scene()
        self.assertFalse(EditLayer(orig, cut).end_stroke())

    def test_one_drag_is_one_undo_step(self):
        orig, cut = scene()
        layer = EditLayer(orig, cut)
        for x in range(24, 37):
            layer.move(Segment(x, 30, x, 30, 3, 0.0, True))
        layer.end_stroke()
        self.assertEqual(layer.count, 1)

    def test_a_soft_brush_leaves_partial_alpha(self):
        layer = new_layer(softness=1.0, radius=10)
        layer.end_stroke()
        self.assertTrue(any(layer.alpha.histogram()[1:255]),
                        "a soft brush should not cut a hard edge")

    def test_a_fast_drag_does_not_leave_gaps(self):
        orig, cut = scene()
        layer = EditLayer(orig, cut)
        layer.move(Segment(22, 30, 38, 30, 3, 0.0, True))     # one long jump
        layer.end_stroke()
        for x in range(23, 38):
            self.assertEqual(layer.alpha.getpixel((x, 30)), 0,
                             f"a gap was left at x={x}")

    def test_restore_brings_back_colour_as_well_as_alpha(self):
        """Outside the subject the model's colour is black, so a restored pixel has to be
        visibly different from what was there before."""
        orig, cut = scene()
        layer = EditLayer(orig, cut)
        self.assertEqual(layer.alpha.getpixel((5, 5)), 0)
        self.assertEqual(layer.rgb.getpixel((5, 5)), (0, 0, 0))
        layer.move(Segment(5, 5, 5, 5, 6, 0.0, False))
        layer.end_stroke()
        self.assertEqual(layer.alpha.getpixel((5, 5)), 255)
        self.assertEqual(layer.rgb.getpixel((5, 5)), layer.orig_rgb.getpixel((5, 5)))

    def test_a_restore_does_not_paint_colour_outside_the_brush(self):
        orig, cut = scene()
        layer = EditLayer(orig, cut)
        layer.move(Segment(5, 5, 5, 5, 4, 0.0, False))
        layer.end_stroke()
        self.assertEqual(layer.rgb.getpixel((25, 25)), cut.convert("RGB").getpixel((25, 25)))


class UndoRedo(unittest.TestCase):
    def test_undo_restores_the_alpha_exactly(self):
        orig, cut = scene()
        layer = EditLayer(orig, cut)
        before = layer.alpha.tobytes()
        layer.move(Segment(30, 30, 30, 30, 6, 0.0, True))
        layer.end_stroke()
        layer.undo()
        self.assertEqual(layer.alpha.tobytes(), before)

    def test_undo_restores_the_colour_exactly(self):
        orig, cut = scene()
        layer = EditLayer(orig, cut)
        before = layer.rgb.tobytes()
        layer.move(Segment(5, 5, 5, 5, 6, 0.0, False))
        layer.end_stroke()
        layer.undo()
        self.assertEqual(layer.rgb.tobytes(), before)

    def test_redo_replays_the_action_bit_for_bit(self):
        layer = new_layer()
        layer.end_stroke()
        after = layer.alpha.tobytes()
        layer.undo()
        layer.redo()
        self.assertEqual(layer.alpha.tobytes(), after)

    def test_undoing_nothing_reports_nothing(self):
        orig, cut = scene()
        layer = EditLayer(orig, cut)
        self.assertIsNone(layer.undo())
        self.assertIsNone(layer.redo())

    def test_undo_describes_what_it_undid(self):
        layer = new_layer()
        layer.end_stroke()
        self.assertIn("Eras", layer.undo())

    def test_a_new_action_clears_the_redo_stack(self):
        layer = new_layer()
        layer.end_stroke()
        layer.undo()
        self.assertTrue(layer.can_redo)
        layer.move(Segment(22, 22, 22, 22, 4, 0.0, True))
        layer.end_stroke()
        self.assertFalse(layer.can_redo)

    def test_history_overflow_bakes_the_state_instead_of_losing_it(self):
        """Past MAX_HISTORY the oldest actions are folded into the base. The pixels must
        survive that even though the undo stack no longer lists them."""
        original = editmod.MAX_HISTORY
        editmod.MAX_HISTORY = 3
        try:
            orig, cut = scene()
            layer = EditLayer(orig, cut)
            spots = [(22, 22), (26, 26), (30, 30), (34, 34)]
            for x, y in spots:
                layer.move(Segment(x, y, x, y, 3, 0.0, True))
                layer.end_stroke()
            for x, y in spots:
                self.assertEqual(layer.alpha.getpixel((x, y)), 0,
                                 f"the stroke at {x},{y} was lost when history overflowed")
            self.assertEqual(layer.count, 0)
            self.assertFalse(layer.can_undo)
        finally:
            editmod.MAX_HISTORY = original

    def test_reset_returns_to_the_model_output_and_forgets_everything(self):
        orig, cut = scene()
        layer = EditLayer(orig, cut)
        base = layer.alpha.tobytes()
        layer.move(Segment(30, 30, 30, 30, 6, 0.0, True))
        layer.end_stroke()
        layer.reset()
        self.assertEqual(layer.alpha.tobytes(), base)
        self.assertFalse(layer.has_edits)
        self.assertFalse(layer.can_undo)


class Rebuild(unittest.TestCase):
    """Re-running the model on a touched-up image must not silently destroy the work."""

    def other_cut(self, size):
        cut = Image.new("RGBA", size, (0, 0, 0, 0))
        ImageDraw.Draw(cut).rectangle((2, 2, 50, 50), fill=(9, 9, 9, 255))
        return cut

    def test_the_new_alpha_replaces_the_old_one(self):
        orig, cut = scene()
        layer = EditLayer(orig, cut)
        layer.move(Segment(30, 30, 30, 30, 6, 0.0, True))
        layer.end_stroke()
        edited = layer.alpha.tobytes()
        layer.rebuild(orig, self.other_cut(cut.size))
        self.assertNotEqual(layer.alpha.tobytes(), edited)

    def test_the_old_state_stays_undoable(self):
        orig, cut = scene()
        layer = EditLayer(orig, cut)
        layer.move(Segment(30, 30, 30, 30, 6, 0.0, True))
        layer.end_stroke()
        edited = layer.alpha.tobytes()
        layer.rebuild(orig, self.other_cut(cut.size))
        self.assertTrue(layer.can_undo)
        layer.undo()
        self.assertEqual(layer.alpha.tobytes(), edited)

    def test_rebuilding_repeatedly_does_not_stack_snapshots(self):
        orig, cut = scene()
        layer = EditLayer(orig, cut)
        layer.move(Segment(30, 30, 30, 30, 6, 0.0, True))
        layer.end_stroke()
        for _ in range(3):
            layer.rebuild(orig, cut)
        self.assertEqual(layer.count, 1)


@needs_scipy
class ClickToRemove(unittest.TestCase):
    def build(self) -> EditLayer:
        size = (60, 60)
        orig = Image.new("RGB", size, (30, 60, 200))
        cut = Image.new("RGBA", size, (0, 0, 0, 0))
        d = ImageDraw.Draw(cut)
        d.rectangle((20, 20, 40, 40), fill=(220, 60, 40, 255))     # subject, 21x21
        d.rectangle((50, 50, 54, 54), fill=(220, 60, 40, 255))     # leftover, 5x5
        return EditLayer(orig, cut)

    def test_removes_the_blob_under_the_click(self):
        layer = self.build()
        ok, msg = layer.remove_island_at(52, 52)
        self.assertTrue(ok)
        self.assertEqual(layer.alpha.getpixel((52, 52)), 0)
        self.assertEqual(layer.alpha.getpixel((30, 30)), 255)
        self.assertEqual(layer.count, 1)
        self.assertIn("25", msg)

    def test_refuses_the_main_subject(self):
        layer = self.build()
        ok, msg = layer.remove_island_at(30, 30)
        self.assertFalse(ok)
        self.assertIn("subject", msg)
        self.assertEqual(layer.alpha.getpixel((30, 30)), 255)
        self.assertEqual(layer.count, 0, "a refused click must not enter the journal")

    def test_a_removed_island_can_be_undone(self):
        layer = self.build()
        before = layer.alpha.tobytes()
        layer.remove_island_at(52, 52)
        layer.undo()
        self.assertEqual(layer.alpha.tobytes(), before)


class Helpers(unittest.TestCase):
    def test_densify_fills_the_gaps_in_a_long_segment(self):
        points = _densify(Segment(0, 0, 100, 0, 4, 0.0, True), 4)
        self.assertGreater(len(points), 50)
        gaps = [points[i + 1][0] - points[i][0] for i in range(len(points) - 1)]
        self.assertLessEqual(max(gaps), 4.0, "stamps must overlap, not dot along")

    def test_densify_handles_a_zero_length_segment(self):
        points = _densify(Segment(5, 5, 5, 5, 4, 0.0, True), 4)
        self.assertTrue(points)
        self.assertTrue(all(p == (5.0, 5.0) for p in points))

    def test_paint_segment_only_touches_its_own_box(self):
        alpha = Image.new("L", (40, 40), 255)
        rgb = Image.new("RGB", (40, 40), (10, 10, 10))
        paint_segment(alpha, rgb, None, Segment(10, 10, 10, 10, 3, 0.0, True), (40, 40))
        self.assertEqual(alpha.getpixel((10, 10)), 0)
        self.assertEqual(alpha.getpixel((35, 35)), 255)

    def test_paint_segment_clips_at_the_image_border(self):
        """A stroke near the edge must not raise or wrap around."""
        alpha = Image.new("L", (20, 20), 255)
        rgb = Image.new("RGB", (20, 20), (10, 10, 10))
        paint_segment(alpha, rgb, None, Segment(0, 0, 1, 1, 8, 0.5, True), (20, 20))
        self.assertEqual(alpha.getpixel((0, 0)), 0)

    def test_a_tiny_radius_still_paints(self):
        alpha = Image.new("L", (20, 20), 255)
        rgb = Image.new("RGB", (20, 20), (10, 10, 10))
        paint_segment(alpha, rgb, None, Segment(10, 10, 10, 10, 0.2, 0.0, True), (20, 20))
        self.assertEqual(alpha.getpixel((10, 10)), 0)


if __name__ == "__main__":                       # pragma: no cover
    unittest.main()
