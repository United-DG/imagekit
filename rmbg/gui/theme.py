"""Colours and fonts."""
from __future__ import annotations

import platform

import customtkinter as ctk

FAMILY = {"Windows": "Segoe UI", "Darwin": ".AppleSystemUIFont"}.get(platform.system(),
                                                                    "Noto Sans")
MONO = {"Windows": "Consolas", "Darwin": "Menlo"}.get(platform.system(), "monospace")


class C:
    """(light, dark) colour pairs. Amber is reserved for primary actions and sliders."""

    bg = ("#edf0ee", "#0f191d")
    panel = ("#ffffff", "#15242a")
    field = ("#e2e8e5", "#1d3037")
    line = ("#cfd8d4", "#28444e")
    sel = ("#ffffff", "#30505b")
    text = ("#13201d", "#e7f0ed")
    muted = ("#566762", "#8fa6a0")
    accent = ("#d99a00", "#f2b134")
    accent_hi = ("#bf8600", "#ffc85a")
    on_accent = "#1b1300"
    danger = ("#b3261e", "#ff8f85")
    ok = ("#1c6b3c", "#69d29a")
    brush = ("#2f6fed", "#7fb2ff")        # brush cursor ring
    checker_light = ((255, 255, 255), (222, 228, 225))
    checker_dark = ((52, 74, 82), (38, 56, 63))


def pick(pair):
    """Resolve a (light, dark) pair against the current appearance mode."""
    return pair[1] if is_dark() else pair[0]


def is_dark() -> bool:
    return ctk.get_appearance_mode() == "Dark"


def checker_colors():
    return C.checker_dark if is_dark() else C.checker_light


class Fonts:
    """Built once, after the root window exists."""

    def __init__(self) -> None:
        self.title = ctk.CTkFont(family=FAMILY, size=18, weight="bold")
        self.head = ctk.CTkFont(family=FAMILY, size=14, weight="bold")
        self.body = ctk.CTkFont(family=FAMILY, size=13)
        self.small = ctk.CTkFont(family=FAMILY, size=12)
        self.mono = ctk.CTkFont(family=MONO, size=11)
