"""Manual touch-up: a brush, a click-to-remove tool, and undo.

This is the answer to "the model got it wrong and there is nothing I can do about it".
Everything paints onto a full-resolution working copy of the cutout.

Undo is a **journal**, not a stack of snapshots: it records *what the user did* (a handful
of numbers per stroke) and replays it from the model's original output. Snapshotting a
12-megapixel alpha and colour layer per stroke would cost tens of megabytes a step; a
journal costs a few hundred bytes and reproduces the result exactly.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from PIL import Image, ImageDraw, ImageFilter

from . import mask as maskops

# Beyond this many actions the oldest are baked into a new base state, which caps memory
# at the cost of not being able to undo arbitrarily far back.
MAX_HISTORY = 60


@dataclass(frozen=True)
class Segment:
    """One straight piece of a brush stroke, in source-image pixels."""

    x0: float
    y0: float
    x1: float
    y1: float
    radius: float
    softness: float          # 0 = hard edge, 1 = very soft
    erase: bool


class Action:
    """Something the user did, replayable against a layer."""

    def apply(self, layer: "EditLayer") -> None:
        raise NotImplementedError

    def describe(self) -> str:
        return "edit"


@dataclass
class BrushStroke(Action):
    segments: list[Segment] = field(default_factory=list)

    def apply(self, layer: "EditLayer") -> None:
        for seg in self.segments:
            layer._paint(seg)

    def describe(self) -> str:
        erase = bool(self.segments) and self.segments[0].erase
        return ("Erased" if erase else "Restored") + " across the image"


@dataclass
class RemoveIsland(Action):
    """Delete the connected blob under a point."""

    x: float
    y: float

    def apply(self, layer: "EditLayer") -> None:
        info = maskops.island_at(layer.alpha, self.x, self.y)
        if info is not None:
            layer.alpha = maskops.drop_island(layer.alpha, info.label)

    def describe(self) -> str:
        return "Removed a leftover island"


@dataclass
class Snapshot(Action):
    """A whole previous state, kept so a re-run of the model stays undoable."""

    rgb: Image.Image
    alpha: Image.Image
    note: str = "previous result"

    def __post_init__(self) -> None:
        # Copy on capture: the layer paints in place, so a live reference would keep
        # changing underneath the snapshot.
        self.rgb = self.rgb.copy()
        self.alpha = self.alpha.copy()

    def apply(self, layer: "EditLayer") -> None:
        layer.rgb = self.rgb.copy()
        layer.alpha = self.alpha.copy()

    def describe(self) -> str:
        return f"the {self.note}"


class EditLayer:
    """A working copy of a cutout: colour, plus an alpha the user can paint on.

    `_base_*` is the state the journal replays from. `rgb`/`alpha` are the current state,
    which is that base with the journal replayed onto it — so `state == base + history`
    holds at all times, and undo means "drop the last action and replay". That is what
    makes undo exact rather than approximate, and it is why a model re-run has to be
    encoded as an action too rather than as a new base (see `rebuild`).

    `_model_*` is the newest model output, which `reset` returns to. It is *not* the same
    as `_base_*` once the model has been re-run over a touched-up image.
    """

    def __init__(self, original: Image.Image, cut: Image.Image) -> None:
        self.orig_rgb = original.convert("RGB")
        self.size: tuple[int, int] = cut.size
        self._model_rgb = cut.convert("RGB")
        self._model_alpha = cut.getchannel("A").copy()
        self._base_rgb = self._model_rgb.copy()
        self._base_alpha = self._model_alpha.copy()
        self.rgb = self._base_rgb.copy()
        self.alpha = self._base_alpha.copy()
        self._history: list[Action] = []
        self._redo: list[Action] = []
        self._pending: BrushStroke | None = None

    # ------------------------------------------------------------------ state
    @property
    def has_edits(self) -> bool:
        return bool(self._history)

    @property
    def count(self) -> int:
        """How many actions are on the undo stack."""
        return len(self._history)

    @property
    def can_undo(self) -> bool:
        return bool(self._history)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    def rebuild(self, original: Image.Image, cut: Image.Image) -> None:
        """Point at a fresh model result.

        The state the user is looking at becomes the new **base**, and the model's output
        becomes a snapshot *action* on top of it. That keeps `state == base + history`
        true, so undoing a re-run is the ordinary pop-and-replay and redo puts the model's
        result back — no special case for either.

        The other encoding (base becomes the model output, snapshot holds the old state)
        looks equivalent but is not: popping the snapshot leaves the new base in place, so
        undo would return the fresh model result instead of the user's work — exactly the
        loss this method exists to prevent.
        """
        self._pending = None
        self.orig_rgb = original.convert("RGB")
        self.size = cut.size
        self._model_rgb = cut.convert("RGB")
        self._model_alpha = cut.getchannel("A").copy()

        if self._history or self._redo:
            # Something is on screen worth keeping: make it the base and replay the new
            # model output as an action on top.
            self._base_rgb = self.rgb.copy()
            self._base_alpha = self.alpha.copy()
            self._history = [Snapshot(self._model_rgb, self._model_alpha, "model re-run")]
        else:
            self._base_rgb = self._model_rgb.copy()
            self._base_alpha = self._model_alpha.copy()
            self._history = []
        self._redo.clear()
        self._replay()

    def reset(self) -> None:
        """Throw away every edit and go back to the newest model output."""
        self._history.clear()
        self._redo.clear()
        self._pending = None
        self._base_rgb = self._model_rgb.copy()
        self._base_alpha = self._model_alpha.copy()
        self.rgb = self._base_rgb.copy()
        self.alpha = self._base_alpha.copy()

    # ------------------------------------------------------------------ brush
    def move(self, seg: Segment) -> None:
        """Apply a segment immediately and remember it as part of the current stroke.

        Painting as the mouse moves is what makes the brush feel immediate; grouping the
        segments into one action on mouse-up is what makes one Ctrl+Z undo the whole drag
        rather than one mouse event.
        """
        self._paint(seg)
        if self._pending is None:
            self._pending = BrushStroke()
        self._pending.segments.append(seg)

    def end_stroke(self) -> bool:
        """Commit the stroke in progress. True when anything was recorded."""
        pending, self._pending = self._pending, None
        if pending is None or not pending.segments:
            return False
        self._record(pending)
        return True

    def remove_island_at(self, x: float, y: float, protect_largest: bool = True
                         ) -> tuple[bool, str]:
        """Delete the blob under (x, y) and journal it.

        The validated result is assigned directly rather than re-derived by replaying the
        action: `remove_island_at` already returns the new alpha, and labelling a 12 MP
        mask twice on a click is a second of lag for nothing. The journal entry still
        replays by coordinate, which is what keeps it correct after later edits.
        """
        new_alpha, message = maskops.remove_island_at(self.alpha, x, y, protect_largest)
        if new_alpha is None:
            return False, message
        self.alpha = new_alpha
        self._record(RemoveIsland(float(x), float(y)))
        return True, message

    # ------------------------------------------------------------------ undo
    def undo(self) -> str | None:
        if not self._history:
            return None
        act = self._history.pop()
        self._redo.append(act)
        self._replay()
        return f"Undid {act.describe()}."

    def redo(self) -> str | None:
        if not self._redo:
            return None
        act = self._redo.pop()
        self._history.append(act)
        self._replay()
        return f"Redid {act.describe()}."

    # ------------------------------------------------------------------ internals
    def _record(self, act: Action) -> None:
        self._history.append(act)
        self._redo.clear()
        if len(self._history) > MAX_HISTORY:
            # Bake the current state and forget the tail. Dropping the oldest entries
            # while keeping the rest would break replay: the retained actions assume the
            # dropped ones already happened.
            self._base_rgb = self.rgb.copy()
            self._base_alpha = self.alpha.copy()
            self._history.clear()

    def _replay(self) -> None:
        self.rgb = self._base_rgb.copy()
        self.alpha = self._base_alpha.copy()
        for act in self._history:
            act.apply(self)

    def _paint(self, seg: Segment) -> None:
        paint_segment(self.alpha, self.rgb, self.orig_rgb, seg, self.size)


def paint_segment(alpha: Image.Image, rgb: Image.Image, orig_rgb: Image.Image | None,
                  seg: Segment, size: tuple[int, int]) -> None:
    """Paint one segment onto an alpha, and onto a colour layer when restoring.

    Kept as a free function rather than a method so the GUI can run the identical code
    against the downscaled preview while a drag is in progress. Repainting a 12 MP frame
    on every mouse event would make the brush unusable; painting the proxy is instant, and
    the authoritative full-resolution layer is rebuilt when the stroke ends.
    """
    radius = max(1.0, float(seg.radius))
    points = _densify(seg, radius)
    pad = radius + 2

    left = max(0, int(math.floor(min(p[0] for p in points) - pad)))
    top = max(0, int(math.floor(min(p[1] for p in points) - pad)))
    right = min(size[0], int(math.ceil(max(p[0] for p in points) + pad)))
    bottom = min(size[1], int(math.ceil(max(p[1] for p in points) + pad)))
    if right <= left or bottom <= top:
        return
    box = (left, top, right, bottom)

    # Only ever touch the bounding box of this segment. A stroke across a large image
    # would otherwise allocate and blend the whole frame on every mouse event.
    stamp = Image.new("L", (right - left, bottom - top), 0)
    draw = ImageDraw.Draw(stamp)
    for px, py in points:
        draw.ellipse((px - left - radius, py - top - radius,
                      px - left + radius, py - top + radius), fill=255)

    softness = min(max(seg.softness, 0.0), 1.0)
    if softness > 0.01:
        stamp = stamp.filter(ImageFilter.GaussianBlur(max(0.5, radius * softness * 0.5)))

    # Image.composite blends by the stamp. paste() would treat the stamp as a copy mask
    # instead, so every soft brush would come out with a hard, jagged edge.
    target = Image.new("L", stamp.size, 0 if seg.erase else 255)
    alpha.paste(Image.composite(target, alpha.crop(box), stamp), box)

    if not seg.erase and orig_rgb is not None:
        # Restoring an area the model removed also has to bring the colour back: the
        # model's RGB under a transparent region is not the subject.
        rgb.paste(Image.composite(orig_rgb.crop(box), rgb.crop(box), stamp), box)


def _densify(seg: Segment, radius: float) -> list[tuple[float, float]]:
    """Walk a segment in steps small enough that the stamps overlap.

    Without this, a fast drag produces a handful of widely spaced events and the stroke
    comes out as a row of separate dots.
    """
    dx, dy = seg.x1 - seg.x0, seg.y1 - seg.y0
    distance = math.hypot(dx, dy)
    step = max(1.0, radius * 0.4)
    n = max(1, int(math.ceil(distance / step)))
    return [(seg.x0 + dx * i / n, seg.y0 + dy * i / n) for i in range(n + 1)]
