"""Small shared widgets and one-line builders so pages stay readable."""
from __future__ import annotations

from typing import Optional

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
    """Label + live value + slider in one tidy row."""

    def __init__(self, app, master, title: str, lo: float, hi: float, value: float,
                 fmt: str = "{:.0f}", steps: Optional[int] = None, on_change=None,
                 note: str = ""):
        super().__init__(master, fg_color="transparent")
        self.fmt, self.on_change = fmt, on_change
        self._silent = False
        self.grid_columnconfigure(0, weight=1)
        self.lbl = ctk.CTkLabel(self, text=title, font=app.fonts.body, text_color=C.text,
                                anchor="w")
        self.lbl.grid(row=0, column=0, sticky="w")
        self.val = ctk.CTkLabel(self, text=fmt.format(value), font=app.fonts.small,
                                text_color=C.muted, anchor="e")
        self.val.grid(row=0, column=1, sticky="e")
        self.slider = ctk.CTkSlider(
            self, from_=lo, to=hi, number_of_steps=steps, command=self._changed, height=16,
            fg_color=C.field, progress_color=C.accent, button_color=C.accent,
            button_hover_color=C.accent_hi)
        self.slider.set(value)
        self.slider.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        if note:
            ctk.CTkLabel(self, text=note, font=app.fonts.small, text_color=C.muted,
                         anchor="w", justify="left", wraplength=310).grid(
                row=2, column=0, columnspan=2, sticky="w")

    def _changed(self, v):
        self.val.configure(text=self.fmt.format(v))
        if self.on_change and not self._silent:
            self.on_change(v)

    def get(self) -> float:
        return self.slider.get()

    def set(self, v: float) -> None:
        """Move the slider to `v` **without** reporting it as a user change.

        `CTkSlider.set` happens to be silent, but the callers that use this (`load_params`,
        `_reset_refine`, `_nudge_brush`) set their own state directly and would double-write
        if that ever changed. The guard makes "programmatic, not an edit" a guarantee rather
        than a coincidence.
        """
        self._silent = True
        try:
            self.slider.set(v)
        finally:
            self._silent = False
        self.val.configure(text=self.fmt.format(v))

    def set_enabled(self, on: bool) -> None:
        self.slider.configure(
            state="normal" if on else "disabled",
            progress_color=C.accent if on else C.line,
            button_color=C.accent if on else C.line,
            button_hover_color=C.accent_hi if on else C.line)
        self.lbl.configure(text_color=C.text if on else C.muted)
