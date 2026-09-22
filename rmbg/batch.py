"""Processing a folder or ZIP full of images."""
from __future__ import annotations

import io
import os
import threading
import zipfile
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path, PurePosixPath
from typing import Callable, Optional

from PIL import Image

from .config import FORMAT_EXT, IMAGE_EXTS, MAX_ZIP_ENTRY_BYTES
from .engine import Engine
from .imaging import load_image
from .params import Params
from .pipeline import BackdropCache, make_output


@dataclass
class Entry:
    """One input image. `read` is deferred so a listing never loads any pixels."""

    name: str
    read: Callable[[], Image.Image]


class Source:
    """A listing of images, plus whatever handle is needed to read them.

    A ZIP is opened once for the entire batch. Opening it per member re-parses the
    central directory every time, which on a large archive is most of the total work.
    """

    def __init__(self, entries, closer: Optional[Callable[[], None]] = None) -> None:
        self.entries = list(entries)
        self._closer = closer

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)

    def __bool__(self) -> bool:
        return bool(self.entries)

    def close(self) -> None:
        if self._closer is not None:
            self._closer()
            self._closer = None

    def __enter__(self) -> "Source":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _zip_reader(zf: zipfile.ZipFile, member: str, size: int) -> Image.Image:
    if size > MAX_ZIP_ENTRY_BYTES:
        raise ValueError("file is too large")
    return load_image(io.BytesIO(zf.read(member)))


def list_inputs(src: str | os.PathLike, recursive: bool = False) -> Source:
    """Images in a folder or a ZIP.

    ZIP member names are flattened to their basename, so an archive cannot steer writes
    outside the destination folder. Recursive folder walks flatten subfolders into the
    name with underscores for the same reason.
    """
    path = Path(src)
    if path.is_dir():
        entries: list[Entry] = []
        walker = path.rglob("*") if recursive else path.iterdir()
        for f in sorted(walker, key=lambda x: str(x).casefold()):
            if not f.is_file() or f.name.startswith("."):
                continue
            if f.suffix.lower() not in IMAGE_EXTS:
                continue
            name = f.name
            if recursive:
                name = str(f.relative_to(path)).replace(os.sep, "_").replace("/", "_")
            entries.append(Entry(name, partial(load_image, f)))
        return Source(entries)

    if path.is_file() and zipfile.is_zipfile(path):
        zf = zipfile.ZipFile(path)
        entries = []
        for info in sorted(zf.infolist(), key=lambda i: i.filename.casefold()):
            if info.is_dir() or "__MACOSX" in info.filename:
                continue
            base = PurePosixPath(info.filename.replace("\\", "/")).name
            if not base or base.startswith((".", "~")):
                continue
            if Path(base).suffix.lower() not in IMAGE_EXTS:
                continue
            entries.append(Entry(base, partial(_zip_reader, zf, info.filename, info.file_size)))
        return Source(entries, zf.close)

    raise ValueError("Source must be a folder or a .zip file.")


def _unique(name: str, used: set[str], folder: Optional[Path]) -> str:
    """A name that collides with neither `used` nor an existing file on disk."""
    stem, ext = os.path.splitext(name)
    cand, n = name, 1
    while cand.lower() in used or (folder is not None and (folder / cand).exists()):
        n += 1
        cand = f"{stem}_{n}{ext}"
    used.add(cand.lower())
    return cand


@dataclass
class BatchResult:
    total: int = 0
    ok: int = 0
    failed: list[tuple[str, str]] = field(default_factory=list)
    cancelled: bool = False
    dest: str = ""


def process_batch(source: Source | str | os.PathLike, dest: str, to_zip: bool, p: Params,
                  engine: Engine,
                  on_progress: Callable[[int, int, str], None] = lambda *a: None,
                  cancel: Optional[threading.Event] = None) -> BatchResult:
    """Run the pipeline over every entry.

    One bad file never stops the batch: failures are collected and reported at the end.
    """
    owned = None
    if isinstance(source, (str, os.PathLike)):
        source = owned = list_inputs(source)
    try:
        return _run(source, dest, to_zip, p, engine, on_progress, cancel or threading.Event())
    finally:
        if owned is not None:
            owned.close()


def _run(source: Source, dest: str, to_zip: bool, p: Params, engine: Engine,
         on_progress: Callable[[int, int, str], None], cancel: threading.Event) -> BatchResult:
    res = BatchResult(total=len(source), dest=str(dest))
    if not len(source):
        return res
    if cancel.is_set():
        # Checked before the destination is created: a batch cancelled before it starts
        # should leave the filesystem exactly as it found it, stray empty folder included.
        res.cancelled = True
        return res

    ext = FORMAT_EXT[p.fmt]
    used: set[str] = set()
    cache = BackdropCache()
    zf: Optional[zipfile.ZipFile] = None
    folder: Optional[Path] = None

    if to_zip:
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        zf = zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED)
    else:
        folder = Path(dest)
        folder.mkdir(parents=True, exist_ok=True)

    try:
        on_progress(0, res.total, "Loading model")
        engine.session(p.model)              # surface a download before the first file
        for i, entry in enumerate(source, 1):
            if cancel.is_set():
                res.cancelled = True
                break
            try:
                cut = engine.cutout(entry.read(), p)
                data, _ = make_output(cut.convert("RGB"), cut.getchannel("A"), p, cache=cache)
                out_name = _unique(f"{Path(entry.name).stem}_nobg{ext}", used, folder)
                if zf is not None:
                    zf.writestr(out_name, data)
                else:
                    (folder / out_name).write_bytes(data)
                res.ok += 1
                on_progress(i, res.total, f"Saved {out_name}")
            except Exception as e:            # keep going, report at the end
                why = ("not a readable image" if isinstance(e, Image.UnidentifiedImageError)
                       else (str(e) or type(e).__name__))
                res.failed.append((entry.name, why))
                on_progress(i, res.total, f"Failed {entry.name}: {why}")
    finally:
        if zf is not None:
            zf.close()
    return res
