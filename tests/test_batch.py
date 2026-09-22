"""Listing inputs and running a batch, with the model stubbed out."""
from __future__ import annotations

import io
import tempfile
import threading
import unittest
import zipfile
from pathlib import Path

from PIL import Image

from rmbg.batch import _unique, list_inputs, process_batch
from rmbg.params import Params
from tests.support import StubEngine


def png_bytes(size=(12, 9), colour=(200, 30, 30)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, colour).save(buf, "PNG")
    return buf.getvalue()


def make_folder(root: Path, n: int = 3, sub: str | None = None) -> Path:
    """`n` images plus the things a listing must learn to ignore."""
    root.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        Image.new("RGB", (24, 18), (200, 30, 30)).save(root / f"img{i}.png")
    (root / "notes.txt").write_text("not an image", encoding="utf-8")
    (root / ".hidden.png").write_bytes(b"")
    if sub:
        deep = root / sub
        deep.mkdir()
        Image.new("RGB", (10, 10), (0, 0, 0)).save(deep / "deep.png")
    return root


class Listing(unittest.TestCase):
    def test_finds_images_and_skips_everything_else(self):
        with tempfile.TemporaryDirectory() as d:
            make_folder(Path(d))
            with list_inputs(d) as src:
                names = [e.name for e in src]
        self.assertEqual(names, ["img0.png", "img1.png", "img2.png"])

    def test_a_folder_listing_is_lazy(self):
        """Nothing should be decoded until the batch actually gets to that file."""
        with tempfile.TemporaryDirectory() as d:
            make_folder(Path(d), n=1)
            entries = list_inputs(d).entries
        self.assertEqual(entries[0].name, "img0.png")      # no exception, no pixels read

    def test_subfolders_are_ignored_unless_asked_for(self):
        with tempfile.TemporaryDirectory() as d:
            make_folder(Path(d), sub="inner")
            with list_inputs(d) as src:
                self.assertNotIn("deep.png", [e.name for e in src])
            with list_inputs(d, recursive=True) as src:
                names = [e.name for e in src]
        self.assertIn("inner_deep.png", names)

    def test_an_empty_folder_is_a_valid_empty_listing(self):
        with tempfile.TemporaryDirectory() as d:
            empty = Path(d) / "nothing"
            empty.mkdir()
            with list_inputs(empty) as src:
                self.assertEqual(len(src), 0)
                self.assertFalse(src)

    def test_something_that_is_neither_a_folder_nor_a_zip_raises(self):
        with tempfile.TemporaryDirectory() as d:
            junk = Path(d) / "x.txt"
            junk.write_text("hello", encoding="utf-8")
            with self.assertRaises(ValueError):
                list_inputs(junk)

    def test_a_zip_is_flattened_to_basenames(self):
        """Flattening is what stops an archive steering writes out of the destination."""
        with tempfile.TemporaryDirectory() as d:
            zpath = Path(d) / "in.zip"
            with zipfile.ZipFile(zpath, "w") as zf:
                zf.writestr("sub/a.png", png_bytes())
                zf.writestr("b.png", png_bytes())
                zf.writestr("__MACOSX/._a.png", b"junk")
                zf.writestr(".hidden.png", png_bytes())
                zf.writestr("~$draft.png", png_bytes())
                zf.writestr("notes.txt", "hello")
            with list_inputs(zpath) as src:
                names = sorted(e.name for e in src)
        self.assertEqual(names, ["a.png", "b.png"])

    def test_a_zip_entry_reads_back_as_an_image(self):
        with tempfile.TemporaryDirectory() as d:
            zpath = Path(d) / "in.zip"
            with zipfile.ZipFile(zpath, "w") as zf:
                zf.writestr("a.png", png_bytes((7, 5)))
            with list_inputs(zpath) as src:
                img = src.entries[0].read()
        self.assertEqual(img.size, (7, 5))

    def test_the_zip_handle_is_closed_when_the_listing_is(self):
        with tempfile.TemporaryDirectory() as d:
            zpath = Path(d) / "in.zip"
            with zipfile.ZipFile(zpath, "w") as zf:
                zf.writestr("a.png", png_bytes())
            src = list_inputs(zpath)
            src.close()
            self.assertIsNone(src._closer)


class Unique(unittest.TestCase):
    def test_returns_the_name_when_it_is_free(self):
        self.assertEqual(_unique("a.png", set(), None), "a.png")

    def test_avoids_a_name_already_used_in_this_run(self):
        self.assertEqual(_unique("a.png", {"a.png"}, None), "a_2.png")

    def test_avoids_overwriting_a_file_already_on_disk(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            (folder / "a.png").write_bytes(b"keep me")
            self.assertEqual(_unique("a.png", set(), folder), "a_2.png")

    def test_keeps_counting_past_the_first_collision(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            (folder / "a.png").write_bytes(b"")
            (folder / "a_2.png").write_bytes(b"")
            self.assertEqual(_unique("a.png", set(), folder), "a_3.png")


class BatchRun(unittest.TestCase):
    def test_writes_one_file_per_image(self):
        with tempfile.TemporaryDirectory() as d:
            src_dir = make_folder(Path(d) / "in")
            out = Path(d) / "out"
            engine = StubEngine()
            with list_inputs(src_dir) as src:
                res = process_batch(src, str(out), False, Params(), engine)
            self.assertEqual((res.total, res.ok, res.failed), (3, 3, []))
            self.assertEqual(sorted(p.name for p in out.iterdir()),
                             ["img0_nobg.png", "img1_nobg.png", "img2_nobg.png"])
            self.assertEqual(engine.calls, 3)

    def test_writes_a_zip_when_asked(self):
        with tempfile.TemporaryDirectory() as d:
            src_dir = make_folder(Path(d) / "in")
            zpath = Path(d) / "out.zip"
            with list_inputs(src_dir) as src:
                res = process_batch(src, str(zpath), True, Params(), StubEngine())
            self.assertEqual(res.ok, 3)
            with zipfile.ZipFile(zpath) as zf:
                self.assertEqual(sorted(zf.namelist()),
                                 ["img0_nobg.png", "img1_nobg.png", "img2_nobg.png"])

    def test_the_zip_is_deflated_not_stored(self):
        with tempfile.TemporaryDirectory() as d:
            src_dir = make_folder(Path(d) / "in", n=1)
            zpath = Path(d) / "out.zip"
            with list_inputs(src_dir) as src:
                process_batch(src, str(zpath), True, Params(), StubEngine())
            with zipfile.ZipFile(zpath) as zf:
                self.assertEqual(zf.infolist()[0].compress_type, zipfile.ZIP_DEFLATED)

    def test_the_output_format_comes_from_params(self):
        with tempfile.TemporaryDirectory() as d:
            src_dir = make_folder(Path(d) / "in", n=1)
            out = Path(d) / "out"
            with list_inputs(src_dir) as src:
                process_batch(src, str(out), False, Params(fmt="WebP"), StubEngine())
            self.assertEqual([p.name for p in out.iterdir()], ["img0_nobg.webp"])

    def test_a_bad_file_does_not_stop_the_batch(self):
        with tempfile.TemporaryDirectory() as d:
            src_dir = make_folder(Path(d) / "in")
            (src_dir / "broken.png").write_text("this is not a png", encoding="utf-8")
            out = Path(d) / "out"
            with list_inputs(src_dir) as src:
                res = process_batch(src, str(out), False, Params(), StubEngine())
            self.assertEqual(res.ok, 3)
            self.assertEqual(len(res.failed), 1)
            name, why = res.failed[0]
            self.assertEqual(name, "broken.png")
            self.assertIn("readable", why)

    def test_names_never_collide_with_what_is_already_there(self):
        with tempfile.TemporaryDirectory() as d:
            src_dir = make_folder(Path(d) / "in", n=1)
            out = Path(d) / "out"
            out.mkdir()
            (out / "img0_nobg.png").write_bytes(b"do not clobber me")
            with list_inputs(src_dir) as src:
                process_batch(src, str(out), False, Params(), StubEngine())
            self.assertEqual((out / "img0_nobg.png").read_bytes(), b"do not clobber me")
            self.assertTrue((out / "img0_nobg_2.png").is_file())

    def test_cancelling_before_the_first_image_writes_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            src_dir = make_folder(Path(d) / "in")
            out = Path(d) / "out"
            cancel = threading.Event()
            cancel.set()
            with list_inputs(src_dir) as src:
                res = process_batch(src, str(out), False, Params(), StubEngine(),
                                    cancel=cancel)
            self.assertTrue(res.cancelled)
            self.assertEqual(res.ok, 0)
            self.assertFalse(out.exists())

    def test_progress_reaches_the_last_image(self):
        with tempfile.TemporaryDirectory() as d:
            src_dir = make_folder(Path(d) / "in", n=2)
            out = Path(d) / "out"
            seen: list[tuple[int, int]] = []
            with list_inputs(src_dir) as src:
                process_batch(src, str(out), False, Params(), StubEngine(),
                              on_progress=lambda i, n, m: seen.append((i, n)))
            self.assertEqual(seen[0], (0, 2))
            self.assertEqual(seen[-1], (2, 2))

    def test_an_empty_source_is_a_no_op(self):
        with tempfile.TemporaryDirectory() as d:
            empty = Path(d) / "empty"
            empty.mkdir()
            out = Path(d) / "out"
            with list_inputs(empty) as src:
                res = process_batch(src, str(out), False, Params(), StubEngine())
            self.assertEqual(res.total, 0)
            self.assertFalse(out.exists())

    def test_a_path_can_be_passed_instead_of_a_listing(self):
        with tempfile.TemporaryDirectory() as d:
            src_dir = make_folder(Path(d) / "in", n=1)
            out = Path(d) / "out"
            res = process_batch(str(src_dir), str(out), False, Params(), StubEngine())
            self.assertEqual(res.ok, 1)

    def test_the_destination_folder_is_created(self):
        with tempfile.TemporaryDirectory() as d:
            src_dir = make_folder(Path(d) / "in", n=1)
            out = Path(d) / "deep" / "nested"
            with list_inputs(src_dir) as src:
                process_batch(src, str(out), False, Params(), StubEngine())
            self.assertTrue(out.is_dir())

    def test_the_result_reports_its_destination(self):
        with tempfile.TemporaryDirectory() as d:
            src_dir = make_folder(Path(d) / "in", n=1)
            out = Path(d) / "out"
            with list_inputs(src_dir) as src:
                res = process_batch(src, str(out), False, Params(), StubEngine())
            self.assertEqual(res.dest, str(out))


if __name__ == "__main__":                       # pragma: no cover
    unittest.main()
