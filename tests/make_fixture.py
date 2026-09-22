"""Generate `tests/fixtures/sample.png` — a fixture that shows the two failure modes.

    python tests/make_fixture.py

The point of this file is to make the quality controls demonstrable. A plain photo does
not show what is wrong; this one does, deliberately:

* **Haze** — a soft ring of real, faint alpha around the subject. No amount of eroding
  removes it, because it is a value, not an artefact. The `--threshold` lever is the fix.
* **A leftover island** — a detached blob the model would call foreground, sitting away
  from the subject. `--despeckle` (or a click in the Touch up tab) is the fix.
* **An interior hole** — a gap punched inside the subject. `--fill-holes` is the fix.

Run it through the CLI with and without the quality flags and compare the `--mask` output:

    python -m rmbg tests/fixtures/sample.png -o out.png --mask plain.png
    python -m rmbg tests/fixtures/sample.png -o out.png \\
        --threshold 40 --hardness 1.6 --despeckle 400 --fill-holes 400 --mask fixed.png
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

SIZE = (480, 360)
HERE = Path(__file__).resolve().parent


def build() -> Image.Image:
    """The photo: a subject on a background, before any cutout."""
    img = Image.new("RGB", SIZE, (58, 74, 96))                 # dull blue background
    d = ImageDraw.Draw(img)
    d.rectangle((70, 60, 300, 320), fill=(226, 96, 62))        # the subject
    d.ellipse((150, 90, 260, 200), fill=(250, 214, 160))       # a detail on it
    d.rectangle((380, 60, 440, 120), fill=(226, 96, 62))       # leftover background blob
    return img


def model_alpha() -> Image.Image:
    """What a model gets roughly right, plus the two mistakes worth fixing."""
    a = Image.new("L", SIZE, 0)
    d = ImageDraw.Draw(a)
    d.rectangle((70, 60, 300, 320), fill=255)                  # the subject, found
    d.rectangle((380, 60, 440, 120), fill=255)                 # ...and the leftover it kept
    d.rectangle((180, 240, 230, 290), fill=0)                  # a hole it punched inside
    haze = a.filter(ImageFilter.GaussianBlur(16))              # a soft ring of real alpha
    return Image.blend(a, haze, 0.55)


def main() -> int:
    out = HERE / "fixtures" / "sample.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    img = build().convert("RGBA")
    img.putalpha(model_alpha())
    img.save(out)
    print(f"Wrote {out}  ({SIZE[0]}x{SIZE[1]} px)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
