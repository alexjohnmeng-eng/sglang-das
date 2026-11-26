"""Out-of-band control channel for the shared embedding arena (ZMQ).

The scheduler's ``tick()`` collective is the fallback path: it needs every
rank to enter the same collective on the same scheduler iteration, so
assignment and DATA_READY notifications wait for the current iteration to
finish (measured: 0.25-1.1s for assignment, 0.24-0.9s for DATA_READY while a
long prefill batch is running).

This module moves those per-request notifications onto pyzmq
ROUTER/DEALER sockets run by a dedicated poller thread, so the owner can
push a message the moment its state changes and a peer can answer with its
ack:

    owner -> peer : ASSIGNED {rid, offset, nbytes}
                    READY    {rid}
                    FAILED   {rid, reason}
                    FREED    {rid}
    peer  -> owner: ACK      {rid, safe_reuse, drain_unknown}

Transport layout:

    owner (rank 0): zmq.ROUTER bound on ipc:///tmp/sglang_arena_ctl/<gid>/owner.ipc
    peer  (rank N): zmq.DEALER connecting to the same endpoint,
                    identity "rankN" so the owner can route replies to it
                    across reconnects.

Reliability: messages are at-least-once and per-peer ordered while the peer
is reachable. A send to a not-yet-connected peer is buffered by the
DEALER/ROUTER pipe (bounded by HWM, beyond which sends fail with EAGAIN and
are retried with backoff), so a slow or restarting peer delays delivery. A
peer that stays unreachable for _PEER_STALE_S (60s) is declared dead and
its queued messages are dropped instead of blocking the other peers; the
arena's alloc/ready/TTL backstops then fail the affected requests. Every
message is idempotent on the receiving side, so a redelivery (or the tick
fallback racing the channel) cannot corrupt state.

Liveness: an ipc endpoint has no EOF-style disconnect notification, so the
owner emits a HB message on each 30s housekeeping tick and the peer logs a
warning once the inbound stream goes quiet. Correctness never depends on
this: wait_data_ready has its own 180s backstop.
"""

import json
import logging
import os
import socket
import threading
import time
from collections import deque
from typing import Callable, Deque, Dict, Optional

import zmq

logger = logging.getLogger(__name__)

_CTL_DIR = "/tmp/sglang_arena_ctl"
_TIMER_INTERVAL_S = 30.0
_HB_STALE_S = 90.0
_POLL_MS = 100  # backstop only: send() wakes the poller through a self-pipe
_HWM = 1000
_RETRY_BACKOFF_MAX_S = 1.0
_BACKOFF_STEP_S = 0.2
_SEND_RETRY_LOG_AT = 5
_PEER_STALE_S = 60.0


class ArenaControlChannel:
    """Per-rank ZMQ poller thread for arena control messages.

    ``apply_fn`` is called with every inbound message under the arena's own
    locking policy; the channel itself holds no arena state. The zmq socket
    is owned by (created and used only in) the poller thread; ``send`` is
    the only cross-thread entry point and just appends to an internal queue.
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        group_id: str,
        apply_fn: Callable[[dict], None],
        on_timer: Optional[Callable[[], None]] = None,
    ):
        self.rank = rank
        self.world_size = world_size
        self.group_id = group_id
        self.is_owner = rank == 0
        self._apply_fn = apply_fn
        self._on_timer = on_timer
        self._ctl_dir = f"{_CTL_DIR}/{group_id}"
        self._ipc_path = f"{self._ctl_dir}/owner.ipc"
        self._endpoint = f"ipc://{self._ipc_path}"
        self._identity = f"rank{rank}".encode()
        # Per-peer outbound queues: one unreachable peer must not block the
        # messages queued for the others. The poller drains them round-robin.
        self._outq: Dict[int, Deque[dict]] = {}
        self._outq_lock = threading.Lock()
        self._rr_cursor = 0
        self._send_attempts: Dict[int, int] = {}
        self._retry_after: Dict[int, float] = {}
        self._unreachable_since: Dict[int, float] = {}
        # Self-pipe: send() writes one byte so the poller wakes immediately
        # instead of waiting out the poll timeout. Without it every control
        # message (and its ack) would sit in the queue for up to
        # _POLL_MS on each hop, adding ~100ms to peer ready_wait.
        self._wake_r, self._wake_w = socket.socketpair()
        self._wake_r.setblocking(False)
        self._wake_w.setblocking(False)
        self._stopped = threading.Event()
        self._poller = threading.Thread(
            target=self._run, daemon=True, name=f"arena-ctl-r{rank}"
        )
        self._ctx: Optional[zmq.Context] = None
        self._last_inbound = 0.0
        self._hb_warned = False

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        os.makedirs(self._ctl_dir, exist_ok=True)
        if self.is_owner:
            try:
                os.unlink(self._ipc_path)
            except FileNotFoundError:
                pass
        self._poller.start()
        logger.info(
            "Arena control channel started: rank=%d/%d role=%s endpoint=%s",
            self.rank,
            self.world_size,
            "ROUTER" if self.is_owner else "DEALER",
            self._endpoint,
        )

    def stop(self) -> None:
        self._stopped.set()
        try:
            self._wake_w.send(b"\x00")
        except OSError:
            pass
        if self._poller.is_alive():
            self._poller.join(timeout=2.0)
        for sock in (self._wake_r, self._wake_w):
            try:
                sock.close()
            except OSError:
                pass
        if self._ctx is not None:
            try:
                # destroy(linger=0) closes any leftover socket and terminates,
                # so a stuck poller cannot hang the process on exit.
                self._ctx.destroy(linger=0)
            except zmq.ZMQError:
                pass
            self._ctx = None
        if self.is_owner:
            try:
                os.unlink(self._ipc_path)
            except OSError:
                pass

    # ------------------------------------------------------------------ #
    # sending
    # ------------------------------------------------------------------ #
    def send(self, rank: int, msg: dict) -> None:
        """Queue one message for ``rank``; the poller thread delivers it.

        Delivery failures are retried with backoff, so a dropped message
        costs latency, never correctness.
        """
        if self._stopped.is_set():
            return
        with self._outq_lock:
            self._outq.setdefault(rank, deque()).append(msg)
        try:
            self._wake_w.send(b"\x00")
        except OSError:
            # The pipe is full or closed; the poll timeout is the backstop.
            pass

    def broadcast(self, msg: dict) -> None:
        for rank in range(self.world_size):
            if rank != self.rank:
                self.send(rank, msg)

    # ------------------------------------------------------------------ #
    # poller thread
    # ------------------------------------------------------------------ #
    def _run(self) -> None:
        try:
            self._ctx = zmq.Context()
            sock = self._ctx.socket(zmq.ROUTER if self.is_owner else zmq.DEALER)
            sock.setsockopt(zmq.IDENTITY, self._identity)
            sock.set_hwm(_HWM)
            if self.is_owner:
                # Fail fast (EHOSTUNREACH) on sends to peers that are not
                # connected yet, instead of silently dropping the frame.
                sock.setsockopt(zmq.ROUTER_MANDATORY, 1)
                sock.bind(self._endpoint)
            else:
                sock.connect(self._endpoint)
        except zmq.ZMQError as error:
            logger.error(
                "Arena control channel setup failed on %s: %s", self._endpoint, error
            )
            if self._ctx is not None:
                self._ctx.destroy(linger=0)
                self._ctx = None
            return

        poller = zmq.Poller()
        poller.register(sock, zmq.POLLIN)
        wake_fd = self._wake_r.fileno()
        poller.register(wake_fd, zmq.POLLIN)
        timer_deadline = time.monotonic() + _TIMER_INTERVAL_S
        try:
            while not self._stopped.is_set():
                timeout = _POLL_MS
                now = time.monotonic()
                with self._outq_lock:
                    deadlines = [
                        self._retry_after.get(peer, 0.0)
                        for peer, queue in self._outq.items()
                        if queue
                    ]
                if deadlines:
                    remaining_ms = (min(deadlines) - now) * 1000.0
                    timeout = min(timeout, max(1, int(remaining_ms) + 1))
                events = dict(poller.poll(timeout))
                if events.get(wake_fd) == zmq.POLLIN:
                    self._drain_wake()
                if events.get(sock) == zmq.POLLIN:
                    self._drain_inbound(sock)
                self._drain_outbound(sock)
                now = time.monotonic()
                if now >= timer_deadline:
                    timer_deadline = now + _TIMER_INTERVAL_S
                    if self.is_owner:
                        # Liveness probe: peers log once if this goes quiet.
                        self.broadcast({"state": "HB"})
                    if self._on_timer is not None:
                        try:
                            self._on_timer()
                        except Exception:
                            logger.exception("Arena control: timer callback failed")
                self._check_hb_stale(now)
        finally:
            sock.close(linger=0)

    def _drain_wake(self) -> None:
        while True:
            try:
                if not self._wake_r.recv(256):
                    return
            except OSError:
                return

    def _drain_outbound(self, sock: "zmq.Socket") -> None:
        now = time.monotonic()
        with self._outq_lock:
            ready = sorted(
                peer
                for peer, queue in self._outq.items()
                if queue and now >= self._retry_after.get(peer, 0.0)
            )
        if not ready:
            return
        # Round-robin so a peer with a long backlog cannot starve the others.
        start = self._rr_cursor % len(ready)
        order = ready[start:] + ready[:start]
        self._rr_cursor += 1
        for peer in order:
            with self._outq_lock:
                queue = self._outq.get(peer)
                if not queue:
                    continue
                msg = queue.popleft()
            payload = json.dumps(msg).encode()
            frames = [f"rank{peer}".encode(), payload] if self.is_owner else [payload]
            try:
                sock.send_multipart(frames, flags=zmq.DONTWAIT)
            except zmq.ZMQError as error:
                # Not connected (ROUTER_MANDATORY) or HWM full: put the
                # message back at the head of this peer's queue and back off
                # for that peer only; the others keep flowing.
                with self._outq_lock:
                    self._outq.setdefault(peer, deque()).appendleft(msg)
                attempts = self._send_attempts.get(peer, 0) + 1
                self._send_attempts[peer] = attempts
                delay = min(attempts * _BACKOFF_STEP_S, _RETRY_BACKOFF_MAX_S)
                self._retry_after[peer] = time.monotonic() + delay
                first_seen = self._unreachable_since.setdefault(peer, time.monotonic())
                if attempts == _SEND_RETRY_LOG_AT:
                    logger.warning(
                        "Arena control: send to rank=%d retrying (%s): errno=%s",
                        peer,
                        msg.get("state"),
                        error.errno,
                    )
                if time.monotonic() - first_seen > _PEER_STALE_S:
                    # A peer that has been unreachable this long is treated as
                    # dead: drop its backlog instead of retrying forever. The
                    # arena's TTL/alloc/ready backstops cover the requests.
                    with self._outq_lock:
                        dropped = len(self._outq.pop(peer, ()))
                    self._send_attempts.pop(peer, None)
                    self._retry_after.pop(peer, None)
                    self._unreachable_since.pop(peer, None)
                    logger.error(
                        "Arena control: rank=%d unreachable for %.0fs; "
                        "dropped %d queued control messages",
                        peer,
                        time.monotonic() - first_seen,
                        dropped,
                    )
                continue
            # Delivered: clear this peer's backoff state.
            self._send_attempts.pop(peer, None)
            self._retry_after.pop(peer, None)
            self._unreachable_since.pop(peer, None)

    def _drain_inbound(self, sock: "zmq.Socket") -> None:
        while True:
            try:
                frames = sock.recv_multipart(flags=zmq.DONTWAIT)
            except zmq.ZMQError:
                return
            # owner: [identity, payload]; peer: [payload]
            payload = frames[-1]
            self._last_inbound = time.monotonic()
            self._hb_warned = False
            try:
                msg = json.loads(payload.decode())
                self._apply_fn(msg)
            except Exception:
                logger.exception(
                    "Arena control: failed to apply message %s", payload[:200]
                )

    def _check_hb_stale(self, now: float) -> None:
        if self.is_owner or self._last_inbound == 0.0:
            return
        if now - self._last_inbound > _HB_STALE_S and not self._hb_warned:
            self._hb_warned = True
            logger.warning(
                "Arena control: no inbound traffic from the owner for %.0fs",
                now - self._last_inbound,
            )
