"""The command line: argument parsing, and real runs with the model stubbed out."""
from __future__ import annotations

import io
import json
import tempfile
import unittest
import zipfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from rmbg.cli import build_parser, main, params_from_args
from rmbg.params import Params
from tests.support import StubEngine


def parse(argv: list[str]):
    return build_parser().parse_args(argv)


class Parsing(unittest.TestCase):
    def test_bare_defaults_match_params(self):
        """The CLI and the dataclass must not drift: a new Params field with no flag would
        otherwise be silently unreachable from the command line."""
        self.assertEqual(params_from_args(parse([])), Params())

    def test_every_quality_flag_reaches_params(self):
        p = params_from_args(parse([
            "--model", "isnet-general-use", "--matting", "--matting-fg", "200",
            "--matting-bg", "20", "--matting-erode", "4", "--no-post-process",
            "--threshold", "45", "--hardness", "1.8", "--despeckle", "300",
            "--fill-holes", "120", "--shrink", "-3", "--feather", "1.5",
            "--defringe", "2", "--brightness", "1.1", "--contrast", "1.2",
            "--saturation", "0.8", "--trim", "--fmt", "WebP", "--quality", "80",
            "--upscale", "2"]))
        self.assertEqual(p.model, "isnet-general-use")
        self.assertTrue(p.matting)
        self.assertEqual((p.matting_fg, p.matting_bg, p.matting_erode), (200, 20, 4))
        self.assertFalse(p.post_process)
        self.assertEqual(p.mask_threshold, 45)
        self.assertAlmostEqual(p.mask_hardness, 1.8)
        self.assertEqual((p.despeckle, p.fill_holes), (300, 120))
        self.assertEqual(p.shrink, -3)
        self.assertAlmostEqual(p.feather, 1.5)
        self.assertEqual(p.defringe, 2)
        self.assertEqual((p.brightness, p.contrast, p.saturation), (1.1, 1.2, 0.8))
        self.assertTrue(p.trim)
        self.assertEqual((p.fmt, p.quality, p.upscale), ("WebP", 80, 2))

    def test_backdrop_flags(self):
        p = params_from_args(parse(["--bg", "color", "--bg-color", "#123456"]))
        self.assertEqual((p.bg_mode, p.bg_color), ("color", "#123456"))

    def test_a_backdrop_image_implies_image_mode(self):
        p = params_from_args(parse(["--bg-image", "wall.png"]))
        self.assertEqual(p.bg_mode, "image")
        self.assertEqual(p.bg_image, "wall.png")

    def test_an_explicit_transparent_backdrop_wins_over_a_leftover_image(self):
        p = params_from_args(parse(["--bg", "transparent"]))
        self.assertEqual((p.bg_mode, p.bg_image), ("transparent", ""))

    def test_an_unknown_model_is_rejected(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse(["--model", "not-a-model"])

    def test_an_unknown_format_is_rejected(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse(["--fmt", "GIF"])

    def test_upscale_is_bounded(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse(["--upscale", "9"])

    def test_version_exits_cleanly(self):
        with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as cm:
            parse(["--version"])
        self.assertEqual(cm.exception.code, 0)


class Runs(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.src = self.dir / "in.png"
        Image.new("RGB", (40, 30), (200, 30, 30)).save(self.src)

    def tearDown(self):
        self._tmp.cleanup()

    def run_cli(self, argv, engine=None):
        """Patch the engine so no model is downloaded and no inference happens."""
        with patch("rmbg.cli.Engine", lambda **kw: engine or StubEngine()):
            with redirect_stdout(io.StringIO()) as out, \
                    redirect_stderr(io.StringIO()) as err:
                code = main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_single_image_writes_the_default_name(self):
        code, _, _ = self.run_cli([str(self.src), "-q"])
        self.assertEqual(code, 0)
        out = self.dir / "in_nobg.png"
        self.assertTrue(out.is_file())
        with Image.open(out) as im:
            self.assertEqual(im.mode, "RGBA")
            self.assertEqual(im.size, (40, 30))

    def test_the_output_name_can_be_chosen(self):
        target = self.dir / "chosen.png"
        code, _, _ = self.run_cli([str(self.src), "-o", str(target), "-q"])
        self.assertEqual(code, 0)
        self.assertTrue(target.is_file())

    def test_the_mask_is_written_when_asked(self):
        mask = self.dir / "m.png"
        code, _, _ = self.run_cli([str(self.src), "-o", str(self.dir / "o.png"),
                                   "--mask", str(mask), "-q"])
        self.assertEqual(code, 0)
        with Image.open(mask) as im:
            self.assertEqual(im.mode, "L")
            self.assertEqual(im.size, (40, 30))

    def test_the_quality_flags_change_the_mask(self):
        """The whole point of the quality controls: the mask on disk must reflect them."""
        plain = self.dir / "plain.png"
        cut = self.dir / "cut.png"
        self.run_cli([str(self.src), "--mask", str(plain), "-q"])
        self.run_cli([str(self.src), "--mask", str(cut), "--threshold", "200",
                      "--shrink", "2", "-q"])
        with Image.open(plain) as a, Image.open(cut) as b:
            self.assertNotEqual(a.tobytes(), b.tobytes())

    def test_json_output_is_machine_readable(self):
        code, out, _ = self.run_cli([str(self.src), "-o", str(self.dir / "o.png"),
                                     "--json"])
        self.assertEqual(code, 0)
        payload = json.loads(out.strip().splitlines()[-1])
        self.assertEqual(payload["size"], [40, 30])
        self.assertEqual(payload["model"], "u2net")
        self.assertTrue(Path(payload["output"]).is_file())

    def test_upscale_is_applied(self):
        code, _, _ = self.run_cli([str(self.src), "-o", str(self.dir / "big.png"),
                                   "--upscale", "2", "-q"])
        self.assertEqual(code, 0)
        with Image.open(self.dir / "big.png") as im:
            self.assertEqual(im.size, (80, 60))

    def test_the_backdrop_colour_is_applied(self):
        code, _, _ = self.run_cli([str(self.src), "-o", str(self.dir / "bg.png"),
                                   "--bg", "color", "--bg-color", "#ff0000", "-q"])
        self.assertEqual(code, 0)
        with Image.open(self.dir / "bg.png") as im:
            self.assertEqual(im.getpixel((0, 0))[:3], (255, 0, 0))

    def test_a_missing_file_fails_with_a_message(self):
        code, _, err = self.run_cli([str(self.dir / "nope.png")])
        self.assertEqual(code, 1)
        self.assertIn("no such file", err)

    def test_no_arguments_prints_help(self):
        code, out, _ = self.run_cli([])
        self.assertEqual(code, 2)
        self.assertIn("usage:", out)

    def test_listing_models_needs_no_input(self):
        code, out, _ = self.run_cli(["--models"])
        self.assertEqual(code, 0)
        self.assertIn("u2net", out)

    def test_listing_models_as_json(self):
        code, out, _ = self.run_cli(["--models", "--json"])
        self.assertEqual(code, 0)
        data = json.loads(out)
        self.assertTrue(any(m["id"] == "birefnet-general-lite" for m in data))
        self.assertTrue(all("cached" in m for m in data))

    def test_batch_writes_into_a_folder(self):
        srcdir = self.dir / "in"
        srcdir.mkdir()
        for i in range(2):
            Image.new("RGB", (20, 20), (10, 40, 90)).save(srcdir / f"p{i}.png")
        outdir = self.dir / "out"
        code, _, _ = self.run_cli(["--batch", str(srcdir), str(outdir), "-q"])
        self.assertEqual(code, 0)
        self.assertEqual(sorted(p.name for p in outdir.iterdir()),
                         ["p0_nobg.png", "p1_nobg.png"])

    def test_batch_to_a_zip(self):
        srcdir = self.dir / "in"
        srcdir.mkdir()
        Image.new("RGB", (20, 20)).save(srcdir / "p0.png")
        zpath = self.dir / "out.zip"
        code, _, _ = self.run_cli(["--batch", str(srcdir), str(zpath), "-q"])
        self.assertEqual(code, 0)
        with zipfile.ZipFile(zpath) as zf:
            self.assertEqual(zf.namelist(), ["p0_nobg.png"])

    def test_batch_reports_failures_in_json(self):
        srcdir = self.dir / "in"
        srcdir.mkdir()
        Image.new("RGB", (20, 20)).save(srcdir / "good.png")
        (srcdir / "bad.png").write_text("not an image", encoding="utf-8")
        code, out, _ = self.run_cli(["--batch", str(srcdir), str(self.dir / "out"),
                                     "--json", "-q"])
        self.assertEqual(code, 1)                  # a failure is a non-zero exit
        payload = json.loads(out)
        self.assertEqual((payload["total"], payload["ok"]), (2, 1))
        self.assertEqual(payload["failed"][0]["name"], "bad.png")

    def test_batch_on_an_empty_folder_fails(self):
        empty = self.dir / "empty"
        empty.mkdir()
        code, _, err = self.run_cli(["--batch", str(empty), str(self.dir / "out")])
        self.assertEqual(code, 1)
        self.assertIn("No supported images", err)

    def test_batch_on_a_missing_source_fails(self):
        code, _, err = self.run_cli(["--batch", str(self.dir / "nope"),
                                     str(self.dir / "out")])
        self.assertEqual(code, 1)
        self.assertIn("folder or a .zip", err)


if __name__ == "__main__":                       # pragma: no cover
    unittest.main()
