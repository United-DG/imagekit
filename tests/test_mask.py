"""The alpha post-processing levers."""
from __future__ import annotations

import unittest

from PIL import Image, ImageDraw

from rmbg.mask import (HAVE_SCIPY, apply_curve, despeckle, drop_island, island_at,
                       remove_island_at)
from tests.support import mask_with

needs_scipy = unittest.skipUnless(HAVE_SCIPY, "scipy is required for component work")


class Curve(unittest.TestCase):
    def test_identity_at_the_defaults(self):
        a = Image.new("L", (4, 4), 100)
        self.assertIs(apply_curve(a, 0, 1.0), a)

    def test_threshold_clears_weak_alpha(self):
        """This is the lever for haze: the model leaves faint but *real* alpha, and no
        amount of eroding removes it because it is not an artefact."""
        out = apply_curve(Image.new("L", (4, 4), 40), 45, 1.0)
        self.assertEqual(out.getpixel((0, 0)), 0)

    def test_threshold_keeps_strong_alpha(self):
        self.assertEqual(apply_curve(Image.new("L", (4, 4), 255), 45, 1.0).getpixel((0, 0)),
                         255)

    def test_the_ramp_above_the_threshold_is_preserved(self):
        """Cutting hard at the threshold would make every anti-aliased edge jagged, so
        values above it are rescaled rather than clipped."""
        a = Image.new("L", (2, 1))
        a.putpixel((0, 0), 45)
        a.putpixel((1, 0), 150)
        out = apply_curve(a, 45, 1.0)
        self.assertEqual(out.getpixel((0, 0)), 0)
        self.assertGreater(out.getpixel((1, 0)), 1)
        self.assertLess(out.getpixel((1, 0)), 255)

    def test_hardness_pushes_values_toward_the_extremes(self):
        a = Image.new("L", (2, 1))
        a.putpixel((0, 0), 100)
        a.putpixel((1, 0), 160)
        out = apply_curve(a, 0, 2.0)
        self.assertLess(out.getpixel((0, 0)), 100)
        self.assertGreater(out.getpixel((1, 0)), 160)

    def test_hardness_below_one_softens_the_edge(self):
        a = Image.new("L", (2, 1))
        a.putpixel((0, 0), 60)
        a.putpixel((1, 0), 200)
        out = apply_curve(a, 0, 0.5)
        self.assertGreater(out.getpixel((0, 0)), 60)
        self.assertLess(out.getpixel((1, 0)), 200)

    def test_the_threshold_is_clamped_to_a_usable_range(self):
        a = Image.new("L", (2, 2), 200)
        self.assertEqual(apply_curve(a, 9999, 1.0).getpixel((0, 0)), 0)


@needs_scipy
class Despeckle(unittest.TestCase):
    SIZE = (60, 60)

    def subjects(self) -> Image.Image:
        """A big subject (41x41 = 1681 px) and a small blob (5x5 = 25 px)."""
        return mask_with(self.SIZE, (5, 5, 45, 45), (50, 50, 54, 54))

    def test_drops_a_small_island(self):
        self.assertEqual(despeckle(self.subjects(), min_area=100).getpixel((52, 52)), 0)

    def test_keeps_the_subject(self):
        self.assertEqual(despeckle(self.subjects(), min_area=100).getpixel((25, 25)), 255)

    def test_keeps_an_island_above_the_threshold(self):
        self.assertEqual(despeckle(self.subjects(), min_area=5).getpixel((52, 52)), 255)

    def test_a_blob_touching_the_edge_is_still_a_speck(self):
        """Deliberate. Leftover background very often sits against the frame edge, so
        sparing border-touching blobs would make this control useless on exactly the
        images that need it. The size threshold is the only protection."""
        m = mask_with((30, 30), (0, 0, 3, 3), (12, 12, 25, 25))
        out = despeckle(m, min_area=100)
        self.assertEqual(out.getpixel((1, 1)), 0)
        self.assertEqual(out.getpixel((18, 18)), 255)

    def test_no_op_when_both_areas_are_zero(self):
        m = self.subjects()
        self.assertIs(despeckle(m, 0, 0), m)

    def test_fills_a_small_interior_hole(self):
        m = mask_with(self.SIZE, (5, 5, 45, 45))
        ImageDraw.Draw(m).rectangle((20, 20, 24, 24), fill=0)
        self.assertEqual(despeckle(m, fill_area=100).getpixel((22, 22)), 255)

    def test_keeps_a_large_interior_hole(self):
        m = mask_with(self.SIZE, (5, 5, 45, 45))
        ImageDraw.Draw(m).rectangle((15, 15, 35, 35), fill=0)
        self.assertEqual(despeckle(m, fill_area=100).getpixel((25, 25)), 0)

    def test_never_fills_the_outside(self):
        """The surround touches the border, so it is background rather than a hole.
        Filling it would turn the whole frame opaque."""
        m = mask_with(self.SIZE, (20, 20, 40, 40))
        out = despeckle(m, fill_area=99_999)
        self.assertEqual(out.getpixel((0, 0)), 0)
        self.assertEqual(out.getpixel((30, 30)), 255)

    def test_returns_a_greyscale_image(self):
        self.assertEqual(despeckle(self.subjects(), min_area=100).mode, "L")


@needs_scipy
class Islands(unittest.TestCase):
    def alpha(self) -> Image.Image:
        return mask_with((60, 60), (5, 5, 45, 45), (50, 50, 54, 54))

    def test_reports_the_area_and_whether_it_is_the_subject(self):
        info = island_at(self.alpha(), 25, 25)
        self.assertIsNotNone(info)
        self.assertEqual(info.area, 41 * 41)
        self.assertTrue(info.is_largest)

    def test_a_small_blob_is_not_the_subject(self):
        info = island_at(self.alpha(), 52, 52)
        self.assertEqual(info.area, 25)
        self.assertFalse(info.is_largest)

    def test_a_transparent_pixel_has_no_island(self):
        self.assertIsNone(island_at(self.alpha(), 0, 0))

    def test_a_point_outside_the_frame_has_no_island(self):
        self.assertIsNone(island_at(self.alpha(), 999, 999))

    def test_remove_deletes_the_clicked_blob(self):
        new, msg = remove_island_at(self.alpha(), 52, 52)
        self.assertIsNotNone(new)
        self.assertEqual(new.getpixel((52, 52)), 0)
        self.assertEqual(new.getpixel((25, 25)), 255)
        self.assertIn("25", msg)

    def test_remove_refuses_the_subject(self):
        """Deleting the subject on a stray click is the worst thing this app could do."""
        original = self.alpha()
        new, msg = remove_island_at(original, 25, 25)
        self.assertIsNone(new)
        self.assertIn("subject", msg)
        self.assertEqual(original.getpixel((25, 25)), 255)

    def test_remove_takes_the_subject_when_asked_not_to_protect_it(self):
        new, _ = remove_island_at(self.alpha(), 25, 25, protect_largest=False)
        self.assertIsNotNone(new)
        self.assertEqual(new.getpixel((25, 25)), 0)

    def test_remove_on_empty_space_explains_itself(self):
        new, msg = remove_island_at(self.alpha(), 0, 0)
        self.assertIsNone(new)
        self.assertIn("transparent", msg)

    def test_drop_island_leaves_everything_else_alone(self):
        a = self.alpha()
        out = drop_island(a, island_at(a, 52, 52).label)
        self.assertEqual(out.getpixel((52, 52)), 0)
        self.assertEqual(out.getpixel((25, 25)), 255)


if __name__ == "__main__":                       # pragma: no cover
    unittest.main()
