"""Headless command line.

Imports the core only — never a GUI toolkit — so it runs on a server, in a script, or in
a test without a display.
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
from pathlib import Path

from . import __version__
from .batch import list_inputs, process_batch
from .config import (BG_MODES, FORMATS, MODELS, UPSCALES, cached_model_bytes,
                     is_model_cached)
from .engine import Engine
from .imaging import load_image
from .params import Params
from .pipeline import produce


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m rmbg",
        description="Remove image backgrounds. Headless; run main.py for the GUI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("input", nargs="?", help="image to process")
    ap.add_argument("-o", "--output", help="output file (default: <name>_nobg.<ext>)")
    ap.add_argument("--batch", nargs=2, metavar=("SRC", "DST"),
                    help="process a folder or ZIP of images into a folder or a .zip")
    ap.add_argument("--recursive", action="store_true", help="walk subfolders during --batch")
    ap.add_argument("--models", action="store_true", help="list the models and exit")
    ap.add_argument("--mask", metavar="PATH", help="also write the finished alpha channel")
    ap.add_argument("--json", action="store_true", help="print a machine-readable result")
    ap.add_argument("-q", "--quiet", action="store_true", help="only print errors")
    ap.add_argument("--cache-size", type=int, default=2, metavar="N",
                    help="how many models to keep loaded at once")
    ap.add_argument("--version", action="version", version=f"rmbg {__version__}")
    _add_params(ap)
    return ap


def _add_params(ap: argparse.ArgumentParser) -> None:
    d = Params()          # single source of truth for defaults

    g = ap.add_argument_group("removal")
    g.add_argument("--model", default=d.model, choices=[m.id for m in MODELS],
                   help="model id; see --models")
    g.add_argument("--matting", action="store_true", default=d.matting,
                   help="alpha matting — better on hair and fur, much slower")
    g.add_argument("--matting-fg", type=int, default=d.matting_fg,
                   help="raise this if matting eats into the subject")
    g.add_argument("--matting-bg", type=int, default=d.matting_bg)
    g.add_argument("--matting-erode", type=int, default=d.matting_erode)
    g.add_argument("--no-post-process", dest="post_process", action="store_false",
                   default=d.post_process, help="skip rembg's built-in mask cleanup")

    g = ap.add_argument_group("mask quality")
    g.add_argument("--threshold", dest="mask_threshold", type=int, default=d.mask_threshold,
                   metavar="0-254", help="alpha at or below this becomes 0 (kills haze)")
    g.add_argument("--hardness", dest="mask_hardness", type=float, default=d.mask_hardness,
                   metavar="0.25-4", help="edge contrast; above 1 sharpens a smeared edge")
    g.add_argument("--despeckle", type=int, default=d.despeckle, metavar="PX",
                   help="delete connected alpha islands smaller than this many pixels")
    g.add_argument("--fill-holes", type=int, default=d.fill_holes, metavar="PX",
                   help="fill transparent holes smaller than this many pixels")

    g = ap.add_argument_group("edges and colour")
    g.add_argument("--shrink", type=int, default=d.shrink, metavar="-10..10",
                   help="erode the edge; negative dilates to cover a leftover fringe")
    g.add_argument("--feather", type=float, default=d.feather, metavar="PX",
                   help="soften the edge")
    g.add_argument("--defringe", type=int, default=d.defringe, metavar="PX",
                   help="recolour soft edges from opaque neighbours (0 = off)")
    g.add_argument("--brightness", type=float, default=d.brightness)
    g.add_argument("--contrast", type=float, default=d.contrast)
    g.add_argument("--saturation", type=float, default=d.saturation)
    g.add_argument("--trim", action="store_true", default=d.trim,
                   help="crop away a fully transparent border")

    g = ap.add_argument_group("backdrop")
    g.add_argument("--bg", choices=list(BG_MODES), default=d.bg_mode,
                   help="what to put behind the subject")
    g.add_argument("--bg-color", default=d.bg_color, metavar="#RRGGBB")
    g.add_argument("--bg-image", default=d.bg_image, help="implies --bg image")

    g = ap.add_argument_group("export")
    g.add_argument("--fmt", choices=FORMATS, default=d.fmt, help="output format")
    g.add_argument("--quality", type=int, default=d.quality, help="WebP and JPG quality")
    g.add_argument("--upscale", type=int, default=d.upscale, choices=list(UPSCALES),
                   help="Lanczos upscale factor; adds pixels, not detail")


def params_from_args(a: argparse.Namespace) -> Params:
    """Explicit field-by-field mapping, so a new Params field cannot be silently ignored."""
    bg_mode = a.bg
    if a.bg_image and bg_mode == "transparent":
        bg_mode = "image"
    return Params(
        model=a.model,
        matting=a.matting,
        matting_fg=a.matting_fg,
        matting_bg=a.matting_bg,
        matting_erode=a.matting_erode,
        post_process=a.post_process,
        mask_threshold=a.mask_threshold,
        mask_hardness=a.mask_hardness,
        despeckle=a.despeckle,
        fill_holes=a.fill_holes,
        shrink=a.shrink,
        feather=a.feather,
        defringe=a.defringe,
        brightness=a.brightness,
        contrast=a.contrast,
        saturation=a.saturation,
        trim=a.trim,
        bg_mode=bg_mode,
        bg_color=a.bg_color,
        bg_image=a.bg_image or "",
        fmt=a.fmt,
        quality=a.quality,
        upscale=a.upscale,
    )


# --------------------------------------------------------------------------- output

def _say(args, message: str) -> None:
    if not args.quiet:
        print(message, flush=True)


def _fail(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)


def _list_models(args) -> int:
    if args.json:
        print(json.dumps([
            {"id": m.id, "label": m.label, "blurb": m.blurb, "size_mb": m.size_mb,
             "cached": is_model_cached(m.id), "bytes": cached_model_bytes(m.id)}
            for m in MODELS], indent=2))
        return 0
    print("Available models:\n")
    width = max(len(m.id) for m in MODELS)
    for m in MODELS:
        on_disk = cached_model_bytes(m.id)
        if on_disk:
            state = f"cached, {on_disk / 1048576:.0f} MB"
        elif m.size_mb:
            state = f"downloads ~{m.size_mb} MB"
        else:
            state = "not downloaded"
        print(f"  {m.id:<{width}}  {m.blurb}")
        print(f"  {'':<{width}}  {state}\n")
    return 0


# --------------------------------------------------------------------------- runs

def _run_single(args) -> int:
    p = params_from_args(args)
    src = Path(args.input)
    if not src.is_file():
        _fail(f"no such file: {src}")
        return 1
    out = Path(args.output) if args.output else src.with_name(f"{src.stem}_nobg{_ext(p.fmt)}")
    if out.parent != Path(""):
        out.parent.mkdir(parents=True, exist_ok=True)

    img = load_image(src)
    engine = Engine(cache_size=args.cache_size)
    if not engine.is_loaded(p.model) and not is_model_cached(p.model):
        _say(args, f"Downloading {p.model} (first use)… this can take a while.")
    _say(args, f"{src.name}: {img.width}x{img.height} px, model {p.model}")
    cut = engine.cutout(img, p)

    res = produce(cut.convert("RGB"), cut.getchannel("A"), p, with_alpha=bool(args.mask))
    out.write_bytes(res.data)
    if args.mask:
        Path(args.mask).parent.mkdir(parents=True, exist_ok=True)
        res.alpha.save(args.mask)

    payload = {"input": str(src), "output": str(out), "model": p.model,
               "size": list(res.size), "bytes": len(res.data),
               "mask": str(args.mask) if args.mask else None}
    if args.json:
        print(json.dumps(payload))
    else:
        _say(args, f"Saved {out}  {res.size[0]}x{res.size[1]} px, "
                   f"{len(res.data) / 1024:.0f} KB")
    return 0


def _run_batch(args) -> int:
    p = params_from_args(args)
    src, dst = args.batch
    to_zip = dst.lower().endswith(".zip")
    try:
        source = list_inputs(src, recursive=args.recursive)
    except Exception as e:
        _fail(str(e))
        return 1
    if not len(source):
        source.close()
        _fail("No supported images found in that source.")
        return 1

    cancel = threading.Event()
    _install_sigint(cancel)

    def progress(i: int, n: int, msg: str) -> None:
        if not args.quiet:
            print(msg if i == 0 else f"[{i}/{n}] {msg}", flush=True)

    engine = Engine(cache_size=args.cache_size)
    _say(args, f"{len(source)} image(s) -> {dst}")
    with source:
        res = process_batch(source, dst, to_zip, p, engine, progress, cancel)

    payload = {"total": res.total, "ok": res.ok, "cancelled": res.cancelled,
               "dest": res.dest, "failed": [{"name": n, "why": w} for n, w in res.failed]}
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        parts = [f"{res.ok} saved"]
        if res.failed:
            parts.append(f"{len(res.failed)} failed")
        if res.cancelled:
            parts.append("cancelled")
        _say(args, f"Done: {', '.join(parts)}. Results in {res.dest}")
        for name, why in res.failed:
            print(f"  failed: {name} — {why}", file=sys.stderr)

    if res.cancelled:
        return 130
    return 1 if res.failed else 0


def _install_sigint(cancel: threading.Event) -> None:
    """Let Ctrl+C finish the current image instead of killing a half-written archive."""
    def handler(signum, frame):
        cancel.set()
        print("\nCancelling after the current image…", file=sys.stderr, flush=True)

    try:
        signal.signal(signal.SIGINT, handler)
    except (ValueError, AttributeError):        # not the main thread, or no SIGINT
        pass


def _ext(fmt: str) -> str:
    from .config import FORMAT_EXT
    return FORMAT_EXT[fmt]


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.models:
            return _list_models(args)
        if args.batch:
            return _run_batch(args)
        if not args.input:
            build_parser().print_help()
            return 2
        return _run_single(args)
    except KeyboardInterrupt:
        return 130
    except Exception as e:
        _fail(str(e) or type(e).__name__)
        return 1


if __name__ == "__main__":                      # pragma: no cover
    sys.exit(main())
