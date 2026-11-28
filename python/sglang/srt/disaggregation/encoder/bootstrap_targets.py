"""Runtime-editable list of prefill bootstrap URLs.

Prefill replicas come and go; an encoder should not need a restart to learn
about them.  :class:`BootstrapTargets` combines the static ``--encoder-register-url``
list with the optional JSON targets file named by
``SGLANG_ENCODER_REGISTER_TARGETS_FILE`` and re-reads that file whenever its
mtime changes.  Consumers get a callback only when the effective list changes.
"""

import json
import logging
import os
import threading
from typing import Callable, Iterable, List, Optional

from sglang.srt.disaggregation.encoder.bootstrap_heartbeat import _dedupe_urls

logger = logging.getLogger(__name__)

OnChangeFn = Callable[[List[str]], None]


def _parse_targets(data) -> Optional[List[str]]:
    """Accepts ``[...]`` or ``{"bootstrap_urls": [...]}``; ``None`` if invalid."""
    if isinstance(data, dict):
        data = data.get("bootstrap_urls")
    if not isinstance(data, list):
        return None
    if any(not isinstance(item, str) for item in data):
        return None
    return _dedupe_urls(data)


class BootstrapTargets:
    """Static bootstrap URLs plus an optional hot-reloaded JSON targets file.

    The file is polled on ``poll_s`` and only re-parsed when its mtime changes.
    Invalid content is logged and the previous list is kept.  ``snapshot()``
    reads the file synchronously on the first call, so callers can build an
    initial heartbeat without waiting for the first poll.
    """

    def __init__(
        self,
        static_urls: Iterable[str],
        file_path: str = "",
        poll_s: float = 10.0,
    ):
        self._static = _dedupe_urls(static_urls)
        self.file_path = file_path or ""
        self.poll_s = max(float(poll_s), 0.0)
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._on_change: Optional[OnChangeFn] = None
        self._current: List[str] = list(self._static)
        self._mtime: Optional[int] = None
        self._started = False
        self._stopped = False

    def snapshot(self) -> List[str]:
        """Current effective URL list (loads the file on first use)."""
        with self._lock:
            self._refresh_locked(force=self._mtime is None and bool(self.file_path))
            return list(self._current)

    def start(self, on_change: OnChangeFn) -> None:
        with self._lock:
            if self._started or self._stopped:
                return
            self._started = True
            self._on_change = on_change
            changed = self._refresh_locked(
                force=self._mtime is None and bool(self.file_path)
            )
        if changed is not None:
            self._notify(changed)
        if self.file_path and self.poll_s > 0:
            thread = threading.Thread(
                target=self._watch,
                daemon=True,
                name="encoder-register-targets",
            )
            with self._lock:
                if self._stopped:
                    return
                self._thread = thread
            thread.start()

    def stop(self) -> None:
        with self._lock:
            if self._stopped:
                return
            self._stopped = True
            thread = self._thread
            self._thread = None
        self._stop.set()
        if (
            thread is not None
            and thread.is_alive()
            and thread is not threading.current_thread()
        ):
            thread.join(timeout=max(self.poll_s, 1.0))

    def _watch(self) -> None:
        while not self._stop.wait(self.poll_s):
            with self._lock:
                if self._stopped:
                    return
                changed = self._refresh_locked(force=False)
            if changed is not None:
                self._notify(changed)

    def _notify(self, urls: List[str]) -> None:
        callback = self._on_change
        if callback is None:
            return
        try:
            callback(list(urls))
        except Exception:  # noqa: BLE001 - a bad consumer must not kill the watcher
            logger.exception("Bootstrap targets on_change callback failed")

    def _refresh_locked(self, force: bool) -> Optional[List[str]]:
        file_urls = self._load_file_locked(force)
        if file_urls is None:
            return None
        new_current = _dedupe_urls(list(self._static) + list(file_urls))
        if new_current == self._current:
            return None
        self._current = new_current
        return list(new_current)

    def _load_file_locked(self, force: bool) -> Optional[List[str]]:
        if not self.file_path:
            return None
        try:
            mtime = os.stat(self.file_path).st_mtime_ns
        except OSError:
            # Missing file is not an error: keep the current list.
            return None
        if not force and self._mtime == mtime:
            return None
        self._mtime = mtime
        try:
            with open(self.file_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, ValueError) as error:
            logger.warning(
                "Ignoring encoder bootstrap targets file %s (%s); keeping %s",
                self.file_path,
                error,
                self._current,
            )
            return None
        urls = _parse_targets(data)
        if urls is None:
            logger.warning(
                "Ignoring encoder bootstrap targets file %s: expected a JSON "
                "list of URLs or {\"bootstrap_urls\": [...]}; keeping %s",
                self.file_path,
                self._current,
            )
            return None
        return urls
