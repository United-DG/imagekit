"""rembg session management and inference."""
from __future__ import annotations

import threading
from collections import OrderedDict

from PIL import Image

from .imaging import flatten_to_rgb, is_fully_transparent
from .params import Params

# Sessions are hundreds of megabytes of ONNX weights, so keep only a couple alive.
DEFAULT_CACHE_SIZE = 2


class Engine:
    """Lazy wrapper around rembg, with an LRU cache of one session per model.

    The lock is deliberately not held while a session is being built: that call downloads
    the weights over the network and can take minutes on first use, and holding the lock
    across it would stall every other caller for the duration.
    """

    def __init__(self, cache_size: int = DEFAULT_CACHE_SIZE) -> None:
        self.cache_size = max(1, int(cache_size))
        self._sessions: OrderedDict[str, object] = OrderedDict()
        self._lock = threading.Lock()
        self._build_lock = threading.Lock()

    def is_loaded(self, model: str) -> bool:
        with self._lock:
            return model in self._sessions

    def loaded(self) -> list[str]:
        with self._lock:
            return list(self._sessions)

    def session(self, model: str):
        with self._lock:
            if model in self._sessions:
                self._sessions.move_to_end(model)
                return self._sessions[model]
        with self._build_lock:
            with self._lock:                  # another thread may have built it while we waited
                if model in self._sessions:
                    self._sessions.move_to_end(model)
                    return self._sessions[model]
            from rembg import new_session     # heavy import; may download weights
            sess = new_session(model)
            with self._lock:
                self._sessions[model] = sess
                self._sessions.move_to_end(model)
                while len(self._sessions) > self.cache_size:
                    self._sessions.popitem(last=False)
                return sess

    def cutout(self, img: Image.Image, p: Params) -> Image.Image:
        """Remove the background. Always returns RGBA at the input's size."""
        from rembg import remove
        if is_fully_transparent(img):
            return img.copy()                 # there is no subject to find; skip the model

        kwargs: dict = {"post_process_mask": bool(p.post_process)}
        if p.matting:
            kwargs.update(
                alpha_matting=True,
                alpha_matting_foreground_threshold=int(p.matting_fg),
                alpha_matting_background_threshold=int(p.matting_bg),
                alpha_matting_erode_size=int(p.matting_erode),
            )
        # flatten_to_rgb matters: handing rembg a plain RGB conversion of an RGBA image
        # discards the alpha and feeds the model whatever colour sat under the transparent
        # pixels (usually black), which mangles a PNG that already had a cut out.
        out = remove(flatten_to_rgb(img), session=self.session(p.model), **kwargs)
        return out.convert("RGBA")
