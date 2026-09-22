"""Small shared widgets and one-line builders so pages stay readable."""
from __future__ import annotations

import math
from typing import Callable, Optional

import customtkinter as ctk

from .theme import C


def button(app, parent, text, command, primary=False, **kw):
    if primary:
        return ctk.CTkButton(parent, text=text, command=command, font=app.fonts.head,
                             corner_radius=9, height=38, fg_color=C.accent,
                             hover_color=C.accent_hi, text_color=C.on_accent,
                             text_color_disabled=("#8a7440", "#7d6a3a"), **kw)
    return ctk.CTkButton(parent, text=text, command=command, font=app.fonts.body,
                         corner_radius=9, height=34, fg_color=C.field,
                         hover_color=C.line, text_color=C.text, **kw)


def segment(app, parent, values, command=None, **kw):
    return ctk.CTkSegmentedButton(
        parent, values=values, command=command, font=app.fonts.body, corner_radius=8,
        fg_color=C.field, selected_color=C.sel, selected_hover_color=C.sel,
        unselected_color=C.field, unselected_hover_color=C.line, text_color=C.text, **kw)


def label(app, parent, text, muted=False, wrap=0, **kw):
    return ctk.CTkLabel(parent, text=text, font=app.fonts.small if muted else app.fonts.body,
                        text_color=C.muted if muted else C.text, anchor="w",
                        justify="left", wraplength=wrap, **kw)


def entry(app, parent, var, placeholder=""):
    return ctk.CTkEntry(parent, textvariable=var, placeholder_text=placeholder,
                        font=app.fonts.body, fg_color=C.field, border_width=0,
                        text_color=C.text, height=34, corner_radius=8)


def switch(app, parent, text, command=None, **kw):
    return ctk.CTkSwitch(parent, text=text, font=app.fonts.body, text_color=C.text,
                         command=command, fg_color=C.line, progress_color=C.accent,
                         button_color=("#ffffff", "#e7f0ed"),
                         button_hover_color=("#ffffff", "#ffffff"), **kw)


class SliderRow(ctk.CTkFrame):
    """Label + live value + slider in one tidy row.

    Two things here are load-bearing rather than cosmetic.

    **The curve.** A slider interpolates linearly, which is right for a setting whose useful
    range is a small fraction of its bounds and wrong for one that spans two orders of
    magnitude. The brush is the second kind: it runs from a 2 px hair to a 400 px swathe, so
    on a linear track every usable size sits in the first fifth of the bar and the rest of the
    travel jumps straight to "wider than the subject". `curve="log"` drives the slider in
    [0, 1] and maps position to value geometrically, which spreads the useful band across the
    middle and makes the small end adjustable at all. Callers still speak only in values.

    **The wheel.** Every slider now lives inside a scrolling tab body, and customtkinter's
    scrollable frame binds `<MouseWheel>` process-wide to scroll that body. A slider that also
    reacted to the wheel would change value *and* scroll at the same time, so these opt out
    (`scroll_step=0`) and the panel keeps the gesture.
    """

    def __init__(self, app, master, title: str, lo: float, hi: float, value: float,
                 fmt: str = "{:.0f}", steps: Optional[int] = None, on_change=None,
                 note: str = "", curve: Optional[str] = None,
                 render: Optional[Callable[[float], str]] = None):
        super().__init__(master, fg_color="transparent")
        self.fmt, self.on_change, self.curve, self.render = fmt, on_change, curve, render
        self._lo, self._hi = float(lo), float(hi)
        self._silent = False
        self._last: float | None = None
        self.grid_columnconfigure(0, weight=1)
        self.lbl = ctk.CTkLabel(self, text=title, font=app.fonts.body, text_color=C.text,
                                anchor="w")
        self.lbl.grid(row=0, column=0, sticky="w")
        self.val = ctk.CTkLabel(self, text=self._text(value), font=app.fonts.small,
                                text_color=C.muted, anchor="e")
        self.val.grid(row=0, column=1, sticky="e")
        # A curved row hands the slider a normalised [0, 1] range and does the maths itself;
        # CTkSlider only knows how to interpolate linearly.
        lo_s, hi_s = (0.0, 1.0) if curve else (self._lo, self._hi)
        self.slider = ctk.CTkSlider(
            self, from_=lo_s, to=hi_s, number_of_steps=steps, command=self._changed,
            height=16, fg_color=C.field, progress_color=C.accent, button_color=C.accent,
            button_hover_color=C.accent_hi, scroll_step=0)
        self.slider.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        if note:
            ctk.CTkLabel(self, text=note, font=app.fonts.small, text_color=C.muted,
                         anchor="w", justify="left", wraplength=310).grid(
                row=2, column=0, columnspan=2, sticky="w")
        self.set(value)

    # ------------------------------------------------------------------ mapping
    def _to_value(self, pos: float) -> float:
        """Slider position -> the number the caller cares about."""
        if not self.curve:
            return pos
        return self._lo * (self._hi / self._lo) ** min(max(pos, 0.0), 1.0)

    def _to_pos(self, value: float) -> float:
        """The number the caller cares about -> slider position."""
        if not self.curve:
            return value
        v = min(max(float(value), self._lo), self._hi)
        return math.log(v / self._lo) / math.log(self._hi / self._lo)

    def _text(self, value: float) -> str:
        return self.render(value) if self.render else self.fmt.format(value)

    def _show(self, value: float) -> None:
        self.val.configure(text=self._text(value))

    # ------------------------------------------------------------------ api
    def _changed(self, v):
        value = self._to_value(v)
        # Ignore a callback that reports the value already held. The wheel is the reason:
        # it is bound process-wide to scroll the tab body, and a scroll over a slider still
        # lands here with an unchanged value. Re-running the pipeline for it would be waste.
        if self._last is not None and value == self._last:
            return
        self._last = value
        self._show(value)
        if self.on_change and not self._silent:
            self.on_change(value)

    def get(self) -> float:
        return self._to_value(self.slider.get())

    def set(self, v: float) -> None:
        """Move the slider to `v` **without** reporting it as a user change.

        `CTkSlider.set` happens to be silent, but the callers that use this (`load_params`,
        `_reset_refine`, `_nudge_brush`) set their own state directly and would double-write
        if that ever changed. The guard makes "programmatic, not an edit" a guarantee rather
        than a coincidence.
        """
        value = min(max(float(v), self._lo), self._hi) if self.curve else float(v)
        self._silent = True
        try:
            self.slider.set(self._to_pos(value))
        finally:
            self._silent = False
        # Record what the slider actually holds, not what was asked for: CTkSlider snaps to
        # `number_of_steps`, and a later drag onto that same snapped value is a no-op.
        self._last = self.get()
        self._show(self._last)

    def set_range(self, lo: float, hi: float) -> None:
        """Move the bounds, clamping the current value into them.

        The brush uses this: how large a sensible brush is depends entirely on how big the
        image is, and a fixed ceiling leaves the slider mostly useless on a small one.
        """
        value = self.get()
        self._lo, self._hi = float(lo), float(hi)
        if not self.curve:
            self.slider.configure(from_=self._lo, to=self._hi)
        self.set(value)

    def bounds(self) -> tuple[float, float]:
        return self._lo, self._hi

    def refresh(self) -> None:
        """Redraw the readout from the held value.

        For a row whose text depends on something outside it — the brush, which reports how
        large it looks on screen as well as in source pixels — that text goes stale when the
        zoom changes.
        """
        if self._last is not None:
            self._show(self._last)

    def set_enabled(self, on: bool) -> None:
        self.slider.configure(
            state="normal" if on else "disabled",
            progress_color=C.accent if on else C.line,
            button_color=C.accent if on else C.line,
            button_hover_color=C.accent_hi if on else C.line)
        self.lbl.configure(text_color=C.text if on else C.muted)
