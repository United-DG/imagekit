"""The batch page: one folder or ZIP in, one folder or ZIP out."""
from __future__ import annotations

from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk

from ..batch import list_inputs, process_batch
from .theme import C, pick
from .widgets import button, entry, label, switch


class BatchPage(ctk.CTkFrame):
    def __init__(self, app, master) -> None:
        super().__init__(master, fg_color="transparent")
        self.app = app
        self.src_var = ctk.StringVar()
        self.dst_var = ctk.StringVar()
        self._auto_dst = False          # the destination was filled in for the user
        wrap = ctk.CTkFrame(self, fg_color=C.panel, corner_radius=14)
        wrap.grid(row=0, column=0, sticky="nsew")
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        wrap.grid_columnconfigure(1, weight=1)

        label(app, wrap, "Run the same settings over a whole folder or .zip.").grid(
            row=0, column=0, columnspan=3, sticky="w", padx=24, pady=(24, 20))

        label(app, wrap, "From").grid(row=1, column=0, sticky="w", padx=(24, 10))
        entry(app, wrap, self.src_var, "folder or .zip full of images").grid(
            row=1, column=1, sticky="ew", padx=(0, 10))
        button(app, wrap, "Browse", self._pick_src, width=90).grid(row=1, column=2,
                                                                   padx=(0, 24))

        label(app, wrap, "To").grid(row=2, column=0, sticky="w", padx=(24, 10), pady=(12, 0))
        self.dst_entry = entry(app, wrap, self.dst_var, "destination folder or .zip")
        self.dst_entry.grid(row=2, column=1, sticky="ew", padx=(0, 10), pady=(12, 0))
        button(app, wrap, "Browse", self._pick_dst, width=90).grid(row=2, column=2,
                                                                   padx=(0, 24),
                                                                   pady=(12, 0))

        opts = ctk.CTkFrame(wrap, fg_color="transparent")
        opts.grid(row=3, column=0, columnspan=3, sticky="w", padx=24, pady=(18, 0))
        self.zip_sw = switch(app, opts, "Write a single .zip",
                             command=self._sync_placeholders)
        self.zip_sw.grid(row=0, column=0, padx=(0, 26))
        self.rec_sw = switch(app, opts, "Include subfolders", command=self.update_summary)
        self.rec_sw.grid(row=0, column=1)
        self.rec_sw.select()

        self.hint = label(app, wrap, "", muted=True, wrap=600)
        self.hint.grid(row=4, column=0, columnspan=3, sticky="w", padx=24, pady=(16, 0))

        self.result = label(app, wrap, "", wrap=600)
        self.result.grid(row=5, column=0, columnspan=3, sticky="w", padx=24, pady=(12, 0))

        self.start_btn = button(app, wrap, "Start batch", self._start, primary=True,
                                width=170)
        self.start_btn.grid(row=6, column=0, columnspan=3, sticky="w", padx=24,
                            pady=(22, 24))
        self.app.lockable.append(self.start_btn)
        # Only now that the hint and result labels exist can the switch sync them.
        self._sync_placeholders()

    # ------------------------------------------------------------------ paths
    def _sync_placeholders(self) -> None:
        """Destination means different things as a folder and as a .zip.

        Flipping the switch re-derives an auto-filled path instead of leaving a folder
        name in a ZIP run, but never touches a path the user chose themselves.
        """
        to_zip = bool(self.zip_sw.get())
        self.dst_entry.configure(
            placeholder_text="destination .zip" if to_zip else "destination folder")
        self.hint.configure(text="")
        self.result.configure(text="")
        if self._auto_dst and self.src_var.get().strip():
            self._default_dest(self.src_var.get().strip())

    def _pick_src(self) -> None:
        path = filedialog.askdirectory(title="Folder of images")
        if path:
            self.src_var.set(path)
            self._default_dest(path)
        else:
            path = filedialog.askopenfilename(title="Or pick a .zip",
                                              filetypes=[("Zip archives", "*.zip")])
            if path:
                self.src_var.set(path)
                self._default_dest(path)
        self.update_summary()

    def _default_dest(self, src: str) -> None:
        """Pre-fill a destination so the common case is one click, not three."""
        if self.dst_var.get() and not self._auto_dst:
            return
        p = Path(src)
        if self.zip_sw.get():
            self.dst_var.set(str(p.with_suffix(".zip")) if p.is_file()
                             else str(p.with_name(p.name + "_nobg.zip")))
        elif p.is_file():
            self.dst_var.set(str(p.with_name(p.stem + "_nobg")))
        else:
            self.dst_var.set(str(p.with_name(p.name + "_nobg")))
        self._auto_dst = True

    def _pick_dst(self) -> None:
        if self.zip_sw.get():
            path = filedialog.asksaveasfilename(title="Save batch as",
                                                defaultextension=".zip",
                                                filetypes=[("Zip archives", "*.zip")])
        else:
            path = filedialog.askdirectory(title="Destination folder")
        if path:
            self.dst_var.set(path)
            self._auto_dst = False
        self.update_summary()

    # ------------------------------------------------------------------ summary
    def update_summary(self) -> None:
        if not hasattr(self, "hint"):
            return
        src = self.src_var.get().strip()
        if not src:
            self.hint.configure(text="Pick a folder or a .zip to process.",
                                text_color=C.muted)
            return
        try:
            with list_inputs(src, recursive=bool(self.rec_sw.get())) as source:
                found = len(source)
        except Exception as e:
            self.hint.configure(text=f"Can't read that: {e}", text_color=pick(C.danger))
            return
        if not found:
            self.hint.configure(text="No images found in there.",
                                text_color=pick(C.danger))
        else:
            self.hint.configure(
                text=f"{found} image{'s' if found != 1 else ''} ready. "
                     f"Each one also gets its own model pass, so this takes a while.",
                text_color=C.muted)

    # ------------------------------------------------------------------ run
    def _start(self) -> None:
        app = self.app
        if app.busy:
            return
        src, dst = self.src_var.get().strip(), self.dst_var.get().strip()
        if not src or not dst:
            app.set_status("Choose both a source and a destination.", error=True)
            return
        to_zip = bool(self.zip_sw.get())
        recursive = bool(self.rec_sw.get())
        p = app.p.copy()
        queue = app.q
        cancel = app.cancel_evt
        self.result.configure(text="", text_color=C.text)

        def work():
            with list_inputs(src, recursive=recursive) as source:
                return process_batch(
                    source, dst, to_zip, p, app.engine,
                    on_progress=lambda i, n, text: queue.put(("progress", i, n, text)),
                    cancel=cancel)

        def done(res, err):
            if err:
                app.set_status("Batch failed.", error=True)
                messagebox.showerror("Batch failed", app.friendly(err))
                return
            self._report(res)

        app.run_task(work, done, "Starting…", determinate=True)

    def _report(self, res) -> None:
        app = self.app
        if res.cancelled:
            app.set_status(f"Stopped after {res.ok} of {res.total}.", error=True)
        else:
            app.set_status(f"Finished: {res.ok} of {res.total} written.", ok=True)
        lines = [f"Wrote {res.ok} of {res.total} image{'s' if res.total != 1 else ''} to "
                 f"{res.dest}."]
        if res.cancelled:
            lines.append("Stopped early — the files already written are complete.")
        if res.failed:
            lines.append(f"\n{len(res.failed)} could not be processed:")
            for name, why in res.failed[:8]:
                lines.append(f"  · {name} — {why}")
            if len(res.failed) > 8:
                lines.append(f"  … and {len(res.failed) - 8} more.")
        self.result.configure(text="\n".join(lines),
                              text_color=pick(C.danger) if res.failed else C.text)
        # `_finish` has already cleared the busy state; this only leaves the bar full so
        # the page still reads as finished rather than reset.
        self.app.progress.configure(progress_color=C.accent)
        self.app.progress.set(1 if res.total and not res.cancelled else 0)
