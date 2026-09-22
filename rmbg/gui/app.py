"""The application shell: navigation, background work, file actions and settings.

Everything that touches the disk, the model, or another thread lives here. The two pages
only build widgets and ask the App to do things, which keeps them readable and keeps the
threading rules in one place.
"""
from __future__ import annotations

import json
import queue
import sys
import threading
import tkinter as tk
import traceback
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk
from PIL import Image

from ..config import FORMAT_EXT, settings_path
from ..edit import EditLayer
from ..engine import Engine
from ..imaging import load_image
from ..params import from_dict
from ..pipeline import BackdropCache, export_image
from .batch_page import BatchPage
from .single import VIEWS, SinglePage
from .theme import C, Fonts
from .widgets import button, segment, switch

try:                                     # optional: drag-and-drop needs tkinterdnd2
    from tkinterdnd2 import DND_FILES, TkinterDnD
    _DND = TkinterDnD.DnDWrapper
except Exception:                        # missing it costs only drag-and-drop
    DND_FILES = TkinterDnD = None

    class _DND:
        pass


def _read_settings() -> dict:
    try:
        return json.loads(settings_path().read_text(encoding="utf-8"))
    except Exception:
        return {}


class App(ctk.CTk, _DND):
    """Owns the state both pages share, and the worker thread that does the heavy work."""

    def __init__(self) -> None:
        self._saved = _read_settings()
        ctk.set_appearance_mode("dark" if self._saved.get("dark", True) else "light")
        super().__init__()

        self.title("Background remover")
        self.geometry("1320x870")
        self.minsize(1080, 740)
        self.configure(fg_color=C.bg)
        self.fonts = Fonts()

        # ---- state shared with the pages
        self.p = from_dict(self._saved.get("params", {}))
        self.engine = Engine(cache_size=2)
        self.bgcache = BackdropCache()
        self.original: Image.Image | None = None
        self.edits: EditLayer | None = None
        self.src_path = ""
        self.busy = False
        self.cancel_evt = threading.Event()
        self.q: queue.Queue = queue.Queue()
        self.lockable: list = []          # widgets disabled while a task runs
        self.drop_ready = False
        self._poll_job = None

        self._build_shell()
        self.single = SinglePage(self, self.body)
        self.batch = BatchPage(self, self.body)
        self._pages = {"Single image": self.single, "Batch": self.batch}
        self.single.load_params(self.p)
        self._show_page("Single image")

        self._bind_keys()
        self._install_drop()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._poll_job = self.after(50, self._poll)
        self._open_from_command_line()

    # =================================================================== shell
    def _build_shell(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        top = ctk.CTkFrame(self, fg_color="transparent")
        top.grid(row=0, column=0, sticky="ew", padx=20, pady=(16, 10))
        top.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(top, text="Background remover", font=self.fonts.title,
                     text_color=C.text).grid(row=0, column=0, sticky="w")
        self.nav = segment(self, top, ["Single image", "Batch"], command=self._show_page)
        self.nav.set("Single image")
        self.nav.grid(row=0, column=1)
        self.theme_sw = switch(self, top, "Dark", command=self._toggle_theme)
        if self._saved.get("dark", True):
            self.theme_sw.select()
        self.theme_sw.grid(row=0, column=2, sticky="e")

        self.body = ctk.CTkFrame(self, fg_color="transparent")
        self.body.grid(row=1, column=0, sticky="nsew", padx=20)
        self.body.grid_columnconfigure(0, weight=1)
        self.body.grid_rowconfigure(0, weight=1)

        bar = ctk.CTkFrame(self, fg_color="transparent")
        bar.grid(row=2, column=0, sticky="ew", padx=22, pady=(10, 14))
        bar.grid_columnconfigure(0, weight=1)
        self.status = ctk.CTkLabel(bar, text="Open an image to begin.", font=self.fonts.small,
                                   text_color=C.muted, anchor="w")
        self.status.grid(row=0, column=0, sticky="w")
        self.progress = ctk.CTkProgressBar(bar, width=200, height=6, fg_color=C.field,
                                           progress_color=C.accent, corner_radius=3)
        self.progress.set(0)
        self.progress.grid(row=0, column=1, sticky="e", padx=(12, 12))
        self.cancel_btn = button(self, bar, "Cancel", self.cancel_task, width=80)
        self.cancel_btn.grid(row=0, column=2, sticky="e")
        self.cancel_btn.configure(state="disabled")

    def _show_page(self, name: str) -> None:
        page = self._pages.get(name) or self.single
        for other in self._pages.values():
            other.grid_remove()
        page.grid(row=0, column=0, sticky="nsew")
        self.nav.set(name)
        if page is self.batch:
            self.batch.update_summary()
        elif self.original is not None:
            self.single.refresh()

    def _toggle_theme(self) -> None:
        ctk.set_appearance_mode("dark" if self.theme_sw.get() else "light")
        self.single.canvas.refresh_theme()
        self.single.refresh()

    # =================================================================== status
    def set_status(self, text: str, *, error: bool = False, ok: bool = False) -> None:
        self.status.configure(text=text,
                              text_color=(C.danger if error else C.ok if ok else C.muted))

    def set_busy(self, busy: bool, message: str = "", determinate: bool = False) -> None:
        self.busy = busy
        for w in self.lockable:
            try:
                w.configure(state="disabled" if busy else "normal")
            except Exception:
                pass
        try:
            self.cancel_btn.configure(state="normal" if (busy and determinate) else "disabled")
        except Exception:
            pass
        self.progress.stop()
        if busy and not determinate:
            self.progress.configure(mode="indeterminate", progress_color=C.accent)
            self.progress.start()
        else:
            self.progress.configure(mode="determinate", progress_color=C.field)
            self.progress.set(0)
        if message:
            self.set_status(message)

    def cancel_task(self) -> None:
        self.cancel_evt.set()
        self.set_status("Finishing the current image, then stopping…")

    # =================================================================== threading
    def run_task(self, work, done, message: str = "Working…", determinate: bool = False) -> None:
        """Run `work` off the Tk thread, then call `done(result, error)` back on it.

        Tk is single-threaded and not thread-safe, so the worker never touches a widget:
        it hands the result to a queue that `_poll` drains.
        """
        if self.busy:
            return
        self.cancel_evt.clear()
        self.set_busy(True, message, determinate)

        def runner():
            try:
                result = work()
            except Exception as e:
                self.q.put(("error", e, done))
            else:
                self.q.put(("done", result, done))

        threading.Thread(target=runner, daemon=True).start()

    def _poll(self) -> None:
        try:
            while True:
                msg = self.q.get_nowait()
                kind = msg[0]
                if kind == "progress":
                    _, i, n, text = msg
                    if n:
                        self.progress.set(i / n)
                        self.set_status(f"{text}  ({i}/{n})")
                    else:
                        self.set_status(text)
                elif kind == "done":
                    self._finish(msg[1], None, msg[2])
                else:
                    self._finish(None, msg[1], msg[2])
        except queue.Empty:
            pass
        self._poll_job = self.after(60, self._poll)

    def destroy(self) -> None:
        """Cancel the timers before tearing the window down.

        `_poll` reschedules itself every 60 ms and the preview debounces a repaint, so
        without this a callback lands on a destroyed interpreter and Tk prints
        `invalid command name ..._poll` to stderr.

        Nothing up to the `super()` call may raise. This runs on the way out, and anything
        that throws before `super().destroy()` leaves the window, its Tcl interpreter and
        tkinter's default root alive — which outlives this object and quietly poisons
        whatever is built next.
        """
        single = getattr(self, "single", None)
        for job in (getattr(self, "_poll_job", None),
                    getattr(single, "_refresh_job", None)):
            if job is not None:
                try:
                    self.after_cancel(job)
                except Exception:
                    pass
        self._poll_job = None
        if single is not None:
            single._refresh_job = None
        super().destroy()

    def _finish(self, result, err, done) -> None:
        self.set_busy(False, "")
        self.progress.configure(progress_color=C.field)
        try:
            done(result, err)
        except Exception:
            self.set_status("Something went wrong.", error=True)
            messagebox.showerror("Unexpected error", traceback.format_exc())

    @staticmethod
    def friendly(err: Exception) -> str:
        """A message a person can act on, rather than a traceback."""
        name = type(err).__name__
        if isinstance(err, FileNotFoundError):
            return "That file is gone."
        if isinstance(err, PermissionError):
            return "That file is open in another program, or not writable."
        if isinstance(err, Image.DecompressionBombError):
            return "That image is too large to open safely."
        if isinstance(err, Image.UnidentifiedImageError):
            return "That file is not an image this app can read."
        if "onnxruntime" in str(err) or name in ("InvalidProtobuf", "NoSuchFile"):
            return f"The model could not be loaded. Check your connection.\n\n{err}"
        return str(err) or name

    # =================================================================== files
    def _open_from_command_line(self) -> None:
        """`python main.py photo.png` opens that photo, so the old habit still works."""
        args = [a for a in sys.argv[1:] if not a.startswith("-")]
        if args and Path(args[0]).is_file():
            self.load_path(args[0])

    def open_image(self, path: str | None = None) -> None:
        if self.busy:
            return
        if not path:
            initial = str(Path(self.src_path).parent) if self.src_path else None
            path = filedialog.askopenfilename(
                title="Open image", initialdir=initial,
                filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff"),
                           ("All files", "*.*")])
        if path:
            self.load_path(path)

    def load_path(self, path: str) -> bool:
        try:
            img = load_image(path)
        except Exception as e:
            messagebox.showerror("Can't open image",
                                 f"{Path(path).name} couldn't be opened.\n\n{self.friendly(e)}")
            return False
        self.set_original(img, path)
        return True

    def set_original(self, img: Image.Image, path: str = "", label: str = "") -> None:
        """A brand-new image with no cutout yet."""
        self.original = img
        self.edits = None
        self.src_path = path
        self.bgcache.reset()
        self.single.on_new_image()
        name = label or (Path(path).name if path else "Untitled")
        self.single.file_lbl.configure(text=f"{name}  ·  {img.width} × {img.height} px")
        self.set_status("Ready. Remove the background when you are.")

    def paste_image(self) -> None:
        """Ctrl+V. PIL's clipboard support is Windows/macOS; on Linux it needs xclip."""
        if self.busy:
            return
        try:
            from PIL import ImageGrab
            data = ImageGrab.grabclipboard()
        except Exception as e:
            self.set_status(f"Clipboard unavailable: {e}", error=True)
            return
        if isinstance(data, Image.Image):
            self.set_original(data.convert("RGBA"), "", "Clipboard image")
            self.set_status("Pasted from the clipboard.")
        elif isinstance(data, list) and data:
            self.load_path(data[0])
        else:
            self.set_status("There is no image on the clipboard.", error=True)

    def remove_bg(self) -> None:
        if self.busy:
            return
        if self.original is None:
            self.set_status("Open an image first.", error=True)
            return
        p, img = self.p.copy(), self.original
        first = not self.engine.is_loaded(p.model)
        message = ("Downloading and loading the model — this happens once…" if first
                   else "Removing the background…")

        def work():
            return self.engine.cutout(img, p)

        def done(cut, err):
            if err:
                self.set_status("Background removal failed.", error=True)
                messagebox.showerror("Background removal failed", self.friendly(err))
                return
            if self.edits is None:
                self.edits = EditLayer(img, cut)
            else:
                # Re-running the model throws the edits away, so keep the edited state
                # on the undo stack: a stray Ctrl+R should not cost an hour of touch-up.
                self.edits.rebuild(img, cut)
            self.single.on_new_cutout()
            self.set_status("Done. Refine the edges, or use Touch up to fix leftovers.",
                            ok=True)

        self.run_task(work, done, message)

    def save_image(self) -> None:
        if self.busy:
            return
        if self.original is None:
            self.set_status("Open an image first.", error=True)
            return
        ext = FORMAT_EXT[self.p.fmt]
        stem = (Path(self.src_path).stem if self.src_path else "image")
        if self.edits is not None:
            stem += "_nobg"
        initial = str(Path(self.src_path).parent) if self.src_path else None
        path = filedialog.asksaveasfilename(
            title="Save image", initialdir=initial, initialfile=stem + ext,
            defaultextension=ext, filetypes=[(self.p.fmt, f"*{ext}"), ("All files", "*.*")])
        if path:
            self.save_to(path)

    def save_to(self, path: str) -> None:
        p = self.p.copy()
        rgb, alpha = self.output_layers()

        def work():
            return export_image(rgb, alpha, p, path, cache=self.bgcache)

        def done(size, err):
            if err:
                self.set_status("Save failed.", error=True)
                messagebox.showerror("Save failed", self.friendly(err))
            else:
                self.set_status(f"Saved {Path(path).name} — {size[0]} × {size[1]} px.", ok=True)

        self.run_task(work, done, "Saving…")

    def output_layers(self):
        """`(rgb, alpha)` at full resolution: the edited cutout, or the bare original."""
        if self.edits is not None:
            return self.edits.rgb, self.edits.alpha
        return self.original.convert("RGB"), None

    # =================================================================== editing
    def undo(self) -> None:
        if self.busy:
            return
        if self.edits is None or not self.edits.can_undo:
            self.set_status("Nothing to undo.")
            return
        self.set_status(self.edits.undo() or "")
        self.single.on_edit_committed()

    def redo(self) -> None:
        if self.busy:
            return
        if self.edits is None or not self.edits.can_redo:
            self.set_status("Nothing to redo.")
            return
        self.set_status(self.edits.redo() or "")
        self.single.on_edit_committed()

    def reset_edits(self) -> None:
        if self.edits is None:
            self.set_status("Open an image first.", error=True)
            return
        self.edits.reset()
        self.single.on_edit_committed()
        self.set_status("Edits cleared — back to the model's result.")

    # =================================================================== keys
    def _bind_keys(self) -> None:
        for mod in ("Control", "Command"):
            for key, fn in (("o", self.open_image), ("s", self.save_image),
                            ("r", self.remove_bg), ("v", self.paste_image),
                            ("z", self.undo), ("y", self.redo)):
                try:
                    self.bind(f"<{mod}-{key}>", lambda e, f=fn: f())
                except tk.TclError:
                    pass                       # this modifier does not exist here
            try:                            # Ctrl+Shift+Z is the other redo people expect
                self.bind(f"<{mod}-Shift-Z>", lambda e: self.redo())
            except tk.TclError:
                pass
        self.bind("<KeyPress>", self._bare_key)

    def _bare_key(self, event) -> None:
        """Single-key shortcuts. Skipped while typing, so they never eat a keystroke."""
        if self._typing():
            return
        if event.keysym in ("1", "2", "3"):
            self.single.view_seg.set(VIEWS[int(event.keysym) - 1])
            self.single.refresh()
        elif event.keysym in ("bracketleft", "bracketright"):
            self._nudge_brush(1 if event.keysym == "bracketright" else -1)

    def _typing(self) -> bool:
        try:
            focused = self.focus_get()
        except Exception:
            return False
        return isinstance(focused, (tk.Entry, tk.Text))

    def _nudge_brush(self, direction: int) -> None:
        page = self.single
        step = max(2, round(page.brush_size * 0.25))
        size = float(min(400, max(2, page.brush_size + direction * step)))
        page.brush_size = size
        page.size_row.set(size)
        self.set_status(f"Brush size {size:.0f} px")

    # =================================================================== drag & drop
    def _install_drop(self) -> None:
        """Optional. Everything is wrapped: without tkinterdnd2 the app still works."""
        if TkinterDnD is None:
            return
        try:
            TkinterDnD._require(self)          # loads the tkdnd package into this Tk
            self.drop_target_register(DND_FILES)
            self.dnd_bind("<<Drop>>", self._on_drop)
            self.drop_ready = True
        except Exception:
            self.drop_ready = False

    def _on_drop(self, event) -> None:
        try:
            paths = list(self.tk.splitlist(event.data))
        except Exception:
            paths = [event.data]
        if paths and not self.busy:
            self.load_path(paths[0])

    # =================================================================== settings
    def save_settings(self) -> None:
        """Best-effort: settings are a convenience and must never block closing."""
        data = dict(self._saved)
        data["dark"] = bool(self.theme_sw.get())
        data["params"] = self.p.to_dict()
        try:
            path = settings_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _on_close(self) -> None:
        if self.busy:
            self.cancel_evt.set()
        self.save_settings()
        self.destroy()
