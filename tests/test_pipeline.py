"""The render pipeline, and the invariant that keeps the preview honest."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

from rmbg.mask import HAVE_SCIPY
from rmbg.params import Params
from rmbg.pipeline import (BackdropCache, compose, defringe, export_image, make_output,
                           produce, refine, render)
from tests.support import flat, mask_with, opaque_pixels

needs_scipy = unittest.skipUnless(HAVE_SCIPY, "scipy is required for despeckle")


class NoCutout(unittest.TestCase):
    def test_none_alpha_returns_the_plain_image(self):
        r = render(flat((12, 12), (9, 9, 9)), None, Params())
        self.assertEqual(r.img.mode, "RGB")
        self.assertIsNone(r.alpha)
        self.assertIsNone(r.crop)

    def test_a_cutout_comes_back_at_the_image_size(self):
        r = render(flat((16, 12)), mask_with((16, 12), (4, 4, 10, 10)), Params())
        self.assertEqual(r.alpha.size, (16, 12))
        self.assertEqual(r.alpha.getpixel((7, 7)), 255)
        self.assertEqual(r.img.mode, "RGBA")


class Colour(unittest.TestCase):
    def test_identity_at_the_defaults(self):
        out, _ = refine(flat((8, 8), (10, 20, 30)), Image.new("L", (8, 8), 255), Params())
        self.assertEqual(out.getpixel((0, 0)), (10, 20, 30))

    def test_brightness_lifts_the_image(self):
        out, _ = refine(flat((4, 4), (100, 100, 100)), Image.new("L", (4, 4), 255),
                        Params(brightness=1.2))
        self.assertGreater(out.getpixel((0, 0))[0], 100)

    def test_saturation_at_zero_greys_the_image(self):
        out, _ = refine(flat((4, 4), (200, 40, 40)), Image.new("L", (4, 4), 255),
                        Params(saturation=0.0))
        r, g, b = out.getpixel((0, 0))
        self.assertAlmostEqual(r, g, delta=2)
        self.assertAlmostEqual(g, b, delta=2)


class Edges(unittest.TestCase):
    def setUp(self):
        self.alpha = mask_with((40, 40), (10, 10, 30, 30))
        self.rgb = flat((40, 40))

    def test_shrink_eats_into_the_subject(self):
        _, out = refine(self.rgb, self.alpha, Params(shrink=2))
        self.assertLess(opaque_pixels(out), opaque_pixels(self.alpha))

    def test_negative_shrink_grows_the_subject(self):
        """Widened from the original 0..10 range: growing the subject is how you cover a
        fringe of original background the model left just outside the edge."""
        _, out = refine(self.rgb, self.alpha, Params(shrink=-2))
        self.assertGreater(opaque_pixels(out), opaque_pixels(self.alpha))

    def test_feather_produces_partial_alpha(self):
        _, out = refine(self.rgb, self.alpha, Params(feather=2.0))
        self.assertTrue(any(out.histogram()[1:255]),
                        "feathering should leave values between 0 and 255")

    def test_a_shrink_below_the_rounding_threshold_does_nothing(self):
        _, out = refine(self.rgb, self.alpha, Params(shrink=0), scale=0.5)
        self.assertEqual(out.tobytes(), self.alpha.tobytes())


class Defringe(unittest.TestCase):
    def test_an_opaque_image_is_returned_untouched(self):
        rgb = flat((20, 20), (200, 30, 30))
        self.assertIs(defringe(rgb, Image.new("L", (20, 20), 255), 2), rgb)

    def test_a_soft_rim_picks_up_the_colour_next_to_it(self):
        """The rim under a soft edge holds old background colour; defringe rebuilds it
        from the opaque neighbours, which is what removes a visible halo."""
        rgb = Image.new("RGB", (20, 20), (0, 0, 250))                    # blue surround
        ImageDraw.Draw(rgb).rectangle((6, 6, 14, 14), fill=(240, 30, 30))  # red subject
        alpha = Image.new("L", (20, 20), 0)
        ImageDraw.Draw(alpha).rectangle((7, 7, 13, 13), fill=255)
        alpha = alpha.filter(ImageFilter.GaussianBlur(1.2))

        box = (3, 3, 17, 17)
        # tobytes()[0::3] is the red channel of an RGB image, and avoids getdata(), which
        # Pillow deprecates for removal in 14.
        before = sum(rgb.crop(box).tobytes()[0::3])
        after = sum(defringe(rgb, alpha, 2).crop(box).tobytes()[0::3])
        self.assertGreater(after, before, "the rim should have moved toward the subject red")


class Compose(unittest.TestCase):
    def test_transparent_mode_is_a_pass_through(self):
        img = Image.new("RGBA", (4, 4), (1, 2, 3, 128))
        self.assertIs(compose(img, Params(), BackdropCache()), img)

    def test_colour_mode_fills_behind_the_subject(self):
        img = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
        out = compose(img, Params(bg_mode="color", bg_color="#ff0000"), BackdropCache())
        self.assertEqual(out.getpixel((2, 2)), (255, 0, 0, 255))

    def test_colour_mode_also_works_before_there_is_a_cutout(self):
        """An image with no cutout yet is plain RGB, and it still has to take a backdrop."""
        out = compose(flat((4, 4), (10, 20, 30)),
                      Params(bg_mode="color", bg_color="#00ff00"), BackdropCache())
        self.assertEqual(out.mode, "RGBA")
        self.assertEqual(out.getpixel((2, 2)), (10, 20, 30, 255))

    def test_the_backdrop_shows_through_a_transparent_subject(self):
        img = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
        out = compose(img, Params(bg_mode="color", bg_color="#0000ff"), BackdropCache())
        self.assertEqual(out.getpixel((1, 1)), (0, 0, 255, 255))

    def test_image_mode_cover_fits_the_frame(self):
        with tempfile.TemporaryDirectory() as d:
            bg = Path(d) / "bg.png"
            Image.new("RGB", (100, 10), (0, 200, 0)).save(bg)
            out = compose(Image.new("RGBA", (20, 20), (0, 0, 0, 0)),
                          Params(bg_mode="image", bg_image=str(bg)), BackdropCache())
        self.assertEqual(out.size, (20, 20))
        self.assertEqual(out.getpixel((10, 10))[:3], (0, 200, 0))


class Trim(unittest.TestCase):
    def test_reports_a_crop_and_keeps_the_alpha_aligned(self):
        r = render(flat((40, 40), (10, 20, 30)), mask_with((40, 40), (10, 10, 25, 20)),
                   Params(trim=True))
        self.assertEqual(r.crop, (10, 10, 26, 21))
        self.assertEqual(r.img.size, (16, 11))
        self.assertEqual(r.alpha.size, r.img.size)

    def test_no_trim_means_no_crop(self):
        r = render(flat((12, 12)), mask_with((12, 12), (2, 2, 9, 9)), Params())
        self.assertIsNone(r.crop)
        self.assertEqual(r.img.size, (12, 12))


@needs_scipy
class PreviewParity(unittest.TestCase):
    """The preview runs this same code on a downscaled proxy.

    Every length scales by `scale` and every area by `scale ** 2`. Getting the area wrong
    makes the preview delete islands the export keeps, which would quietly break the one
    thing this app is for.
    """

    SUBJECT_FULL = (10, 10, 110, 110)     # 101 x 101 = 10201 px
    BLOB_FULL = (150, 150, 170, 170)      # 21 x 21 = 441 px
    SUBJECT_HALF = (5, 5, 55, 55)         # 51 x 51 = 2601 px
    BLOB_HALF = (75, 75, 85, 85)          # 11 x 11 = 121 px

    def full(self) -> Image.Image:
        return mask_with((200, 200), self.SUBJECT_FULL, self.BLOB_FULL)

    def half(self) -> Image.Image:
        return mask_with((100, 100), self.SUBJECT_HALF, self.BLOB_HALF)

    def test_a_blob_kept_at_full_size_is_also_kept_by_the_preview(self):
        p = Params(despeckle=200)         # below both 441 (full) and 121 (proxy, x0.25)
        r_full = render(flat((200, 200)), self.full(), p, scale=1.0)
        r_half = render(flat((100, 100)), self.half(), p, scale=0.5)
        self.assertEqual(r_full.alpha.getpixel((160, 160)), 255)
        self.assertEqual(r_half.alpha.getpixel((80, 80)), 255,
                         "the preview removed an island the export keeps — "
                         "despeckle's area is not scaling by scale ** 2")

    def test_a_blob_dropped_at_full_size_is_also_dropped_by_the_preview(self):
        p = Params(despeckle=300)         # above both 169 (full) and 49 (proxy, x0.25)
        full = mask_with((200, 200), self.SUBJECT_FULL, (150, 150, 162, 162))
        half = mask_with((100, 100), self.SUBJECT_HALF, (75, 75, 81, 81))
        self.assertEqual(render(flat((200, 200)), full, p, scale=1.0).alpha.getpixel((156, 156)), 0)
        self.assertEqual(render(flat((100, 100)), half, p, scale=0.5).alpha.getpixel((78, 78)), 0)

    def test_the_subject_survives_on_both_scales(self):
        p = Params(despeckle=200)
        self.assertEqual(render(flat((200, 200)), self.full(), p, scale=1.0)
                         .alpha.getpixel((60, 60)), 255)
        self.assertEqual(render(flat((100, 100)), self.half(), p, scale=0.5)
                         .alpha.getpixel((30, 30)), 255)

    def test_the_curve_runs_before_despeckle_so_a_leftover_can_disconnect(self):
        """Haze often bridges the subject to a leftover. Breaking that bridge is what lets
        despeckle see two separate pieces at all; run despeckle first and the bridge makes
        them one large component, so the leftover survives."""
        a = Image.new("L", (60, 30), 0)
        d = ImageDraw.Draw(a)
        d.rectangle((2, 2, 30, 28), fill=255)     # subject, 29 x 27 = 783 px
        d.rectangle((50, 2, 56, 8), fill=255)     # leftover, 7 x 7 = 49 px
        d.rectangle((31, 4, 49, 12), fill=150)    # hazy bridge, above the opaque cutoff
        r = render(flat((60, 30)), a, Params(mask_threshold=160, despeckle=400))
        self.assertEqual(r.alpha.getpixel((20, 15)), 255, "the subject should survive")
        self.assertEqual(r.alpha.getpixel((53, 5)), 0,
                         "the leftover should have gone once the bridge was cleared")


class Produce(unittest.TestCase):
    def setUp(self):
        self.rgb = flat((20, 10))
        self.alpha = mask_with((20, 10), (2, 2, 18, 8))

    def test_upscale_multiplies_the_output_size(self):
        res = produce(self.rgb, self.alpha, Params(upscale=2))
        self.assertEqual(res.size, (40, 20))
        self.assertTrue(res.data.startswith(b"\x89PNG"))

    def test_the_mask_comes_back_at_the_output_size(self):
        res = produce(self.rgb, self.alpha, Params(upscale=2), with_alpha=True)
        self.assertEqual(res.alpha.size, (40, 20))

    def test_no_mask_unless_asked(self):
        self.assertIsNone(produce(self.rgb, self.alpha, Params()).alpha)

    def test_no_alpha_means_no_mask_even_when_asked(self):
        self.assertIsNone(produce(self.rgb, None, Params(), with_alpha=True).alpha)

    def test_make_output_returns_bytes_and_a_size(self):
        data, size = make_output(self.rgb, self.alpha, Params())
        self.assertIsInstance(data, bytes)
        self.assertEqual(size, (20, 10))

    def test_export_image_writes_a_readable_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "out.png"
            size = export_image(self.rgb, self.alpha, Params(), path)
            self.assertEqual(size, (20, 10))
            with Image.open(path) as im:
                self.assertEqual(im.size, (20, 10))
                self.assertEqual(im.mode, "RGBA")

    def test_a_jpg_export_loses_the_alpha_channel(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "out.jpg"
            export_image(self.rgb, self.alpha, Params(fmt="JPG"), path)
            with Image.open(path) as im:
                self.assertEqual(im.mode, "RGB")


if __name__ == "__main__":                       # pragma: no cover
    unittest.main()
