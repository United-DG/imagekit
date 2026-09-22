"""The bounds the GUI sliders and the CLI share, and the one that varies per image."""
from __future__ import annotations

import unittest

from rmbg.params import RANGES, brush_range


class BrushRange(unittest.TestCase):
    """`brush_size` is the only control whose useful span depends on the image."""

    def test_a_big_image_gets_the_full_ceiling(self):
        self.assertEqual(brush_range(6000), (2, 400))

    def test_a_small_image_gets_a_small_one(self):
        """A 400 px ceiling is meaningless on a small image.

        Every size a person would actually use ends up crammed into the first sliver of the
        slider, which is the half of the reported brush bug that is not about the curve.
        """
        self.assertEqual(brush_range(200), (2, 40))
        self.assertEqual(brush_range(1000), (2, 200))

    def test_the_floor_holds_for_a_tiny_image(self):
        self.assertEqual(brush_range(60), (2, 24))
        self.assertEqual(brush_range(1), (2, 24))

    def test_the_default_brush_is_always_inside_the_range_it_is_shown_on(self):
        """`SinglePage.brush_size` starts at 24; a range that excludes it would clamp on load."""
        for edge in (1, 7, 24, 60, 200, 400, 1000, 1200, 4032, 20000):
            lo, hi = brush_range(edge)
            self.assertGreater(lo, 0)
            self.assertLess(lo, hi)
            self.assertLessEqual(hi, 400)
            self.assertTrue(lo <= 24 <= hi, f"24 is outside {brush_range(edge)}")

    def test_it_stays_inside_the_absolute_range_the_cli_knows_about(self):
        """`RANGES` is the outer envelope both front ends agree on; this only narrows it."""
        lo, hi = RANGES["brush_size"]
        for edge in (1, 200, 1000, 6000):
            r_lo, r_hi = brush_range(edge)
            self.assertGreaterEqual(r_lo, lo)
            self.assertLessEqual(r_hi, hi)


if __name__ == "__main__":                       # pragma: no cover
    unittest.main()
