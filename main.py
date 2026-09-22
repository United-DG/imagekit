"""Desktop entry point. `python main.py [image]` opens the GUI.

The implementation lives in the `rmbg` package; this file stays tiny so the command
already in your shell history keeps working. For headless use see `python -m rmbg`.
"""
from __future__ import annotations

import sys


def main() -> int:
    try:
        from rmbg.gui.app import App
    except ImportError as e:                     # a missing GUI dependency, not a crash
        print(f"The window needs customtkinter and Pillow, which are not importable: {e}",
              file=sys.stderr)
        print("Install them with:  pip install -r requirements.txt", file=sys.stderr)
        return 1
    App().mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
