"""Loading, cropping, resizing, encoding."""
from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from rmbg.imaging import (checkerboard, encode, flatten_to_rgb, is_fully_transparent,
                          load_image, mask_preview, trim_transparent, upscale)


class LoadImage(unittest.TestCase):
    def test_returns_rgba(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "a.png"
            Image.new("RGB", (10, 6), (200, 30, 30)).save(path)
            img = load_image(path)
        self.assertEqual(img.mode, "RGBA")
        self.assertEqual(img.size, (10, 6))

    def test_honours_exif_orientation(self):
        """A phone photo is stored unrotated with an EXIF flag; reading it raw gives the
        wrong shape and every later crop would be off by a quarter turn."""
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "rot.jpg"
            img = Image.new("RGB", (40, 20), (10, 120, 30))
            for x in range(40):                       # asymmetry, so a rotation is visible
                for y in range(20):
                    if x < 8:
                        img.putpixel((x, y), (250, 250, 0))
            exif = Image.Exif()
            exif[274] = 6                             # 6 = rotate 90 CW on display
            img.save(path, "JPEG", exif=exif)
            loaded = load_image(path)
        self.assertEqual(loaded.size, (20, 40))

    def test_accepts_a_file_like_object(self):
        buf = io.BytesIO()
        Image.new("RGB", (5, 7), (0, 0, 0)).save(buf, "PNG")
        buf.seek(0)
        self.assertEqual(load_image(buf).size, (5, 7))


class Flatten(unittest.TestCase):
    def test_passes_rgb_straight_through(self):
        img = Image.new("RGB", (4, 4), (1, 2, 3))
        self.assertIs(flatten_to_rgb(img), img)

    def test_composites_transparency_onto_the_background(self):
        img = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
        img.putpixel((0, 0), (10, 20, 30, 255))
        out = flatten_to_rgb(img, (255, 255, 255))
        self.assertEqual(out.mode, "RGB")
        self.assertEqual(out.getpixel((0, 0)), (10, 20, 30))
        self.assertEqual(out.getpixel((3, 3)), (255, 255, 255))

    def test_fully_opaque_skips_the_paste(self):
        img = Image.new("RGBA", (4, 4), (5, 6, 7, 255))
        self.assertEqual(flatten_to_rgb(img).getpixel((2, 2)), (5, 6, 7))


class FullyTransparent(unittest.TestCase):
    def test_true_for_an_empty_layer(self):
        self.assertTrue(is_fully_transparent(Image.new("RGBA", (4, 4), (0, 0, 0, 0))))

    def test_false_when_anything_is_opaque(self):
        img = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
        img.putpixel((1, 1), (9, 9, 9, 1))
        self.assertFalse(is_fully_transparent(img))

    def test_false_for_images_without_alpha(self):
        self.assertFalse(is_fully_transparent(Image.new("RGB", (4, 4))))


class Trim(unittest.TestCase):
    def test_crops_and_reports_the_box(self):
        img = Image.new("RGBA", (20, 20), (0, 0, 0, 0))
        img.paste(Image.new("RGBA", (5, 5), (9, 9, 9, 255)), (6, 7))
        out, box = trim_transparent(img)
        self.assertEqual(box, (6, 7, 11, 12))
        self.assertEqual(out.size, (5, 5))

    def test_reports_none_when_nothing_was_cropped(self):
        img = Image.new("RGBA", (8, 8), (1, 2, 3, 255))
        out, box = trim_transparent(img)
        self.assertIsNone(box)
        self.assertIs(out, img)

    def test_ignores_alpha_below_the_threshold(self):
        """Near-transparent speckle must not defeat the crop."""
        img = Image.new("RGBA", (20, 20), (0, 0, 0, 0))
        img.paste(Image.new("RGBA", (4, 4), (9, 9, 9, 255)), (2, 2))
        img.putpixel((19, 19), (255, 255, 255, 3))
        out, box = trim_transparent(img)
        self.assertEqual(box, (2, 2, 6, 6))
        self.assertEqual(out.size, (4, 4))


class Upscale(unittest.TestCase):
    def test_one_is_a_no_op(self):
        img = Image.new("RGBA", (8, 8), (1, 2, 3, 255))
        self.assertIs(upscale(img, 1), img)

    def test_doubles_the_size_and_keeps_alpha(self):
        img = Image.new("RGBA", (8, 6), (1, 2, 3, 200))
        out = upscale(img, 2)
        self.assertEqual(out.size, (16, 12))
        self.assertEqual(out.mode, "RGBA")
        self.assertEqual(out.getchannel("A").getpixel((8, 6)), 200)

    def test_refuses_an_absurd_result(self):
        """The guard is checked before any resize, so this must not need to allocate the
        enormous image it is refusing — the limit is patched down instead."""
        img = Image.new("RGBA", (100, 100), (0, 0, 0, 255))
        with patch("rmbg.imaging.MAX_OUTPUT_PIXELS", 100_000):
            with self.assertRaises(ValueError):
                upscale(img, 4)
        self.assertEqual(upscale(img, 2).size, (200, 200))     # a sane factor still works


class Encode(unittest.TestCase):
    def test_png_round_trips_transparency(self):
        img = Image.new("RGBA", (4, 4), (10, 20, 30, 128))
        out = Image.open(io.BytesIO(encode(img, "PNG")))
        self.assertEqual(out.mode, "RGBA")
        self.assertEqual(out.getpixel((1, 1)), (10, 20, 30, 128))

    def test_jpg_flattens_to_white(self):
        img = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
        out = Image.open(io.BytesIO(encode(img, "JPG", 95)))
        self.assertEqual(out.mode, "RGB")
        self.assertGreater(out.getpixel((4, 4))[0], 240)

    def test_webp_is_accepted(self):
        data = encode(Image.new("RGBA", (4, 4), (1, 1, 1, 255)), "WEBP", 90)
        self.assertTrue(data.startswith(b"RIFF"))

    def test_unknown_format_raises(self):
        with self.assertRaises(ValueError):
            encode(Image.new("RGBA", (4, 4)), "TIFF")


class Previews(unittest.TestCase):
    def test_checkerboard_is_opaque_and_the_right_size(self):
        chk = checkerboard(20, 12, (255, 255, 255), (200, 200, 200))
        self.assertEqual(chk.size, (20, 12))
        self.assertEqual(chk.mode, "RGB")

    def test_checkerboard_survives_a_zero_sized_request(self):
        chk = checkerboard(0, 0, (255, 255, 255), (200, 200, 200))
        self.assertEqual(chk.size, (1, 1))

    def test_mask_preview_is_greyscale_and_opaque(self):
        alpha = Image.new("L", (6, 4), 100)
        out = mask_preview(alpha)
        self.assertEqual(out.mode, "RGBA")
        self.assertEqual(out.getpixel((2, 2)), (100, 100, 100, 255))


if __name__ == "__main__":                       # pragma: no cover
    unittest.main()
