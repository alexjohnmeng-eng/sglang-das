"""Out-of-band control channel for the shared embedding arena.

The scheduler's ``tick()`` collective is the fallback path: it needs every
rank to enter the same collective on the same scheduler iteration, so
assignment and DATA_READY notifications wait for the current iteration to
finish (measured: 0.25-1.1s for assignment, 0.24-0.9s for DATA_READY while a
long prefill batch is running).

This module moves those per-request notifications onto Unix domain sockets
run by dedicated threads, so the owner can push a message the moment its
state changes and a peer can answer with its ack:

    owner -> peer : ASSIGNED {rid, offset, nbytes}
                    READY    {rid}
                    FAILED   {rid, reason}
                    FREED    {rid}
    peer  -> owner: ACK      {rid, safe_reuse, drain_unknown}

Embedding payloads stay in the shared host arena; control messages are
length-prefixed JSON. Every message is idempotent on the receiving side, so a
redelivery (or the tick fallback racing the socket) cannot corrupt state.

Sockets live under /tmp/sglang_arena_ctl/<group_id>/rank<N>.sock; the group
id is unique per prefill instance so two prefill servers cannot collide.
"""

import json
import logging
import os
import socket
import struct
import threading
import time
from collections import deque
from typing import Callable, Deque, Dict, Optional, Set, Tuple

logger = logging.getLogger(__name__)

_CTL_DIR = "/tmp/sglang_arena_ctl"
_HEADER = struct.Struct("<I")
_SEND_RETRY = 3


def _framed_send(sock: socket.socket, payload: bytes) -> None:
    sock.sendall(_HEADER.pack(len(payload)) + payload)


def _framed_recv(sock: socket.socket) -> Optional[bytes]:
    header = sock.recv(_HEADER.size)
    if len(header) < _HEADER.size:
        return None
    (length,) = _HEADER.unpack(header)
    chunks = []
    got = 0
    while got < length:
        chunk = sock.recv(length - got)
        if not chunk:
            return None
        chunks.append(chunk)
        got += len(chunk)
    return b"".join(chunks)


class ArenaControlChannel:
    """Per-rank server + per-peer sender threads for arena control messages.

    ``apply_fn`` is called with every inbound message under the arena's own
    locking policy; the channel itself holds no arena state.
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        group_id: str,
        apply_fn: Callable[[dict], None],
    ):
        self.rank = rank
        self.world_size = world_size
        self.group_id = group_id
        self._apply_fn = apply_fn
        self._ctl_dir = f"{_CTL_DIR}/{group_id}"
        self._sock_path = f"{self._ctl_dir}/rank{rank}.sock"
        self._outq: Deque[Tuple[int, dict]] = deque()
        self._outq_lock = threading.Condition()
        self._conns: Dict[int, socket.socket] = {}
        self._conns_lock = threading.Lock()
        self._stopped = threading.Event()
        self._server = threading.Thread(
            target=self._serve, daemon=True, name=f"arena-ctl-srv-r{rank}"
        )
        self._sender = threading.Thread(
            target=self._drain, daemon=True, name=f"arena-ctl-snd-r{rank}"
        )

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        os.makedirs(self._ctl_dir, exist_ok=True)
        try:
            os.unlink(self._sock_path)
        except FileNotFoundError:
            pass
        self._server.start()
        self._sender.start()
        logger.info(
            "Arena control channel started: rank=%d/%d path=%s",
            self.rank,
            self.world_size,
            self._sock_path,
        )

    def stop(self) -> None:
        self._stopped.set()
        with self._outq_lock:
            self._outq_lock.notify_all()
        for sock in list(self._conns.values()):
            try:
                sock.close()
            except OSError:
                pass
        self._conns.clear()
        try:
            os.unlink(self._sock_path)
        except OSError:
            pass

    # ------------------------------------------------------------------ #
    # sending
    # ------------------------------------------------------------------ #
    def send(self, rank: int, msg: dict) -> None:
        """Queue one message for ``rank``; the sender thread delivers it.

        Delivery failures fall back to the tick collective, so a dropped
        message costs latency, never correctness.
        """
        if self._stopped.is_set():
            return
        with self._outq_lock:
            self._outq.append((rank, msg))
            self._outq_lock.notify_all()

    def broadcast(self, msg: dict) -> None:
        for rank in range(self.world_size):
            if rank != self.rank:
                self.send(rank, msg)

    def _drain(self) -> None:
        while not self._stopped.is_set():
            with self._outq_lock:
                while not self._outq:
                    self._outq_lock.wait(0.5)
                    if self._stopped.is_set():
                        return
                rank, msg = self._outq.popleft()
            payload = json.dumps(msg).encode()
            for attempt in range(_SEND_RETRY):
                sock = self._connect(rank)
                if sock is None:
                    time.sleep(0.2 * (attempt + 1))
                    continue
                try:
                    _framed_send(sock, payload)
                    break
                except OSError:
                    self._drop_conn(rank)
                    if attempt + 1 == _SEND_RETRY:
                        logger.warning(
                            "Arena control: dropping message for rank=%d "
                            "after %d failures: %s",
                            rank,
                            _SEND_RETRY,
                            msg.get("state"),
                        )

    def _connect(self, rank: int) -> Optional[socket.socket]:
        with self._conns_lock:
            sock = self._conns.get(rank)
            if sock is not None:
                return sock
        path = f"{self._ctl_dir}/rank{rank}.sock"
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.connect(path)
        except OSError as error:
            try:
                sock.close()
            except OSError:
                pass
            if os.path.exists(path):
                logger.debug("Arena control: connect rank=%d failed: %s", rank, error)
            return None
        with self._conns_lock:
            self._conns[rank] = sock
        return sock

    def _drop_conn(self, rank: int) -> None:
        with self._conns_lock:
            sock = self._conns.pop(rank, None)
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass

    # ------------------------------------------------------------------ #
    # serving
    # ------------------------------------------------------------------ #
    def _serve(self) -> None:
        try:
            server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            server.bind(self._sock_path)
        except OSError as error:
            logger.error(
                "Arena control channel bind failed on %s: %s", self._sock_path, error
            )
            return
        server.listen(8)
        server.settimeout(0.5)
        while not self._stopped.is_set():
            try:
                conn, _ = server.accept()
            except socket.timeout:
                continue
            except OSError:
                if self._stopped.is_set():
                    break
                continue
            threading.Thread(
                target=self._serve_conn,
                args=(conn,),
                daemon=True,
                name=f"arena-ctl-msg-r{self.rank}",
            ).start()
        server.close()

    def _serve_conn(self, conn: socket.socket) -> None:
        try:
            while not self._stopped.is_set():
                try:
                    payload = _framed_recv(conn)
                except OSError:
                    break
                if payload is None:
                    break
                try:
                    msg = json.loads(payload.decode())
                    self._apply_fn(msg)
                except Exception:
                    logger.exception(
                        "Arena control: failed to apply message %s", payload[:200]
                    )
        finally:
            try:
                conn.close()
            except OSError:
                pass
