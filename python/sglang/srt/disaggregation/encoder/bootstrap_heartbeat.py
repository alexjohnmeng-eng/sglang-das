"""Heartbeat registration of encoder instances with prefill bootstraps.

Prefill keeps its encoder registry in process memory and only accepts push
registrations, so an encoder that registered once stays invisible to a
prefill that restarts later.  The original one-shot retry loop also gave up
after ~2.5 minutes.  Re-posting the encoder URL periodically makes the
registry self-healing: a restarted or newly added prefill converges within
one heartbeat, and a prefill that was unreachable at encoder start does not
require an encoder restart.

Targets can also be updated at runtime (:meth:`BootstrapHeartbeat.update_targets`)
so a targets-file watcher can add/remove prefill replicas without restarting
the encoder.  Each target owns its own stop event, which keeps add/remove
independent and makes shutdown deterministic.
"""

import logging
import threading
from typing import Callable, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

PostFn = Callable[[str, dict, float], bool]


def _dedupe_urls(urls: Iterable[str]) -> List[str]:
    """Strip, drop empties and deduplicate while preserving input order.

    A trailing slash is stripped so ``http://host:8997/`` cannot turn into a
    ``//register_encoder_url`` path that 404s forever.
    """
    seen = set()
    result: List[str] = []
    for url in urls:
        if not isinstance(url, str):
            continue
        url = url.strip().rstrip("/")
        if not url or url in seen:
            continue
        seen.add(url)
        result.append(url)
    return result


def _default_post(
    bootstrap_url: str, payload: dict, timeout: float, token: str = ""
) -> bool:
    import requests

    headers = {"Authorization": f"Bearer {token}"} if token else None
    try:
        resp = requests.post(
            f"{bootstrap_url}/register_encoder_url",
            json=payload,
            headers=headers,
            timeout=timeout,
        )
        if resp.status_code != 200:
            logger.warning(
                "Bootstrap %s returned %s on register: %s",
                bootstrap_url,
                resp.status_code,
                resp.text[:200],
            )
            return False
        # Policy rejections (allowlist, URL cap, retired generation) answer
        # 200 with accepted=false; counting those as success would hide a
        # prefill that silently ignores this encoder.
        try:
            body = resp.json()
        except ValueError:
            return True  # legacy prefill answers 200 with a plain "OK"
        if isinstance(body, dict) and body.get("accepted") is False:
            reason = body.get("reason") or "rejected"
            if reason == "already registered":
                return True
            logger.warning(
                "Bootstrap %s rejected encoder registration: %s",
                bootstrap_url,
                reason,
            )
            return False
        return True
    except Exception as error:  # noqa: BLE001 - any transport error means retry
        logger.debug("Register attempt to %s failed: %s", bootstrap_url, error)
    return False


class _Target:
    """One bootstrap target: a loop thread plus its private stop event."""

    __slots__ = ("url", "thread", "stop_event")

    def __init__(
        self, url: str, thread: threading.Thread, stop_event: threading.Event
    ) -> None:
        self.url = url
        self.thread = thread
        self.stop_event = stop_event


class BootstrapHeartbeat:
    """Re-post one encoder URL to every bootstrap URL on a fixed cadence.

    One daemon thread per bootstrap target: a slow or dead prefill only delays
    its own goal, never the others.  Failures back off exponentially (capped)
    and are logged on state transitions, so a long outage does not spam logs.

    Every payload carries the stable per-process ``instance_id`` so a prefill
    can tell a re-registration apart from a different encoder replica.  When
    ``token`` is set, the default transport sends it as
    ``Authorization: Bearer <token>``.
    """

    def __init__(
        self,
        encoder_url: str,
        bootstrap_urls: Iterable[str],
        *,
        instance_id: str = "",
        token: str = "",
        post_fn: Optional[PostFn] = None,
        heartbeat_s: float = 15.0,
        max_backoff_s: float = 60.0,
        request_timeout_s: float = 5.0,
    ):
        self.encoder_url = encoder_url
        self.bootstrap_urls = _dedupe_urls(bootstrap_urls)
        self.instance_id = instance_id
        self.token = token
        self.post_fn = post_fn or self._make_default_post(token)
        self.heartbeat_s = max(float(heartbeat_s), 0.0)
        self.max_backoff_s = max(float(max_backoff_s), self.heartbeat_s, 1e-3)
        self.request_timeout_s = request_timeout_s
        self._payload = {"url": self.encoder_url, "instance_id": self.instance_id}
        self._enabled = self.heartbeat_s > 0
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._targets: Dict[str, _Target] = {}
        self._threads: List[threading.Thread] = []
        self._started = False
        self._stopped = False

    @staticmethod
    def _make_default_post(token: str) -> PostFn:
        if not token:
            return _default_post

        def _post(bootstrap_url: str, payload: dict, timeout: float) -> bool:
            return _default_post(bootstrap_url, payload, timeout, token=token)

        return _post

    def start(self) -> None:
        with self._lock:
            if self._started or self._stopped:
                return
            self._started = True
            if not self._enabled:
                return
            for bootstrap_url in self.bootstrap_urls:
                self._add_target_locked(bootstrap_url)

    def update_targets(self, urls: Iterable[str]) -> None:
        """Add / remove targets without disturbing the surviving ones.

        New URLs get their own thread; removed URLs have their stop event set
        and the thread joined (bounded by ``request_timeout_s``).  Calling
        before :meth:`start` or after :meth:`stop` is a no-op.
        """
        new_urls = _dedupe_urls(urls)
        removed: List[_Target] = []
        with self._lock:
            if not self._started or self._stopped or not self._enabled:
                return
            new_set = set(new_urls)
            for url in list(self._targets):
                if url not in new_set:
                    removed.append(self._targets.pop(url))
            for url in new_urls:
                if url not in self._targets:
                    self._add_target_locked(url)
            self.bootstrap_urls = list(new_urls)

        for target in removed:
            target.stop_event.set()
        for target in removed:
            target.thread.join(self.request_timeout_s)

    def current_targets(self) -> List[str]:
        """Snapshot of the bootstrap URLs this heartbeat is posting to."""
        with self._lock:
            return list(self.bootstrap_urls)

    def stop(self, timeout: float = 2.0) -> None:
        with self._lock:
            if self._stopped:
                return
            self._stopped = True
            self._stop.set()
            targets = list(self._targets.values())
            self._targets.clear()
            self._threads.clear()

        for target in targets:
            target.stop_event.set()
        for target in targets:
            target.thread.join(timeout)

    def _add_target_locked(self, bootstrap_url: str) -> Optional[_Target]:
        if bootstrap_url in self._targets:
            return None
        stop_event = threading.Event()
        thread = threading.Thread(
            target=self._loop,
            args=(bootstrap_url, self._payload, stop_event),
            daemon=True,
            name=f"encoder-register-heartbeat-{bootstrap_url}",
        )
        target = _Target(bootstrap_url, thread, stop_event)
        self._targets[bootstrap_url] = target
        self._threads.append(thread)
        thread.start()
        return target

    def _loop(
        self, bootstrap_url: str, payload: dict, stop_event: threading.Event
    ) -> None:
        encoder_url = payload.get("url", self.encoder_url)
        registered: Optional[bool] = None
        failures = 0
        delay = self.heartbeat_s
        while not stop_event.is_set() and not self._stopped:
            ok = self.post_fn(bootstrap_url, payload, self.request_timeout_s)
            if ok:
                if registered is not True:
                    if registered is False:
                        logger.info(
                            "Encoder URL '%s' re-registered with bootstrap %s "
                            "after %d failed attempts",
                            encoder_url,
                            bootstrap_url,
                            failures,
                        )
                    else:
                        logger.info(
                            "Registered encoder URL '%s' with bootstrap at %s",
                            encoder_url,
                            bootstrap_url,
                        )
                registered = True
                failures = 0
                delay = self.heartbeat_s
            else:
                failures += 1
                if registered is not False:
                    logger.warning(
                        "Bootstrap %s has not acknowledged encoder "
                        "registration; retrying with backoff in the background",
                        bootstrap_url,
                    )
                registered = False
                delay = min(
                    self.heartbeat_s * (2 ** min(failures, 6)), self.max_backoff_s
                )
            stop_event.wait(delay)
