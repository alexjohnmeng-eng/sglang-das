"""Transport-level tests for the ZMQ arena control channel."""

import os
import threading
import time

import pytest
import zmq

from sglang.srt.disaggregation.encoder import arena_control
from sglang.srt.disaggregation.encoder.arena_control import ArenaControlChannel


def _make(rank, world_size, group_id, received, on_timer=None):
    channel = ArenaControlChannel(
        rank=rank,
        world_size=world_size,
        group_id=group_id,
        apply_fn=received.append,
        on_timer=on_timer,
    )
    channel.start()
    return channel


def _wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


@pytest.fixture()
def short_timer(monkeypatch):
    monkeypatch.setattr(arena_control, "_TIMER_INTERVAL_S", 0.2)
    monkeypatch.setattr(arena_control, "_HB_STALE_S", 1.0)


def test_owner_peer_roundtrip():
    gid = f"zmqrt_{os.getpid()}"
    owner_seen, peer_seen = [], []
    owner = _make(0, 2, gid, owner_seen)
    peer = _make(1, 2, gid, peer_seen)
    try:
        owner.send(1, {"state": "ASSIGNED", "rid": "r1", "offset": 8, "nbytes": 64})
        peer.send(0, {"state": "ACK", "rid": "r1", "safe_reuse": True})
        assert _wait_for(lambda: peer_seen), "peer never got ASSIGNED"
        assert _wait_for(lambda: owner_seen), "owner never got ACK"
        assert peer_seen[0]["state"] == "ASSIGNED"
        assert owner_seen[0]["state"] == "ACK"
    finally:
        owner.stop()
        peer.stop()


def test_owner_broadcast_reaches_all_peers():
    gid = f"zmqbc_{os.getpid()}"
    owner_seen, p1, p2 = [], [], []
    owner = _make(0, 3, gid, owner_seen)
    peers = [_make(1, 3, gid, p1), _make(2, 3, gid, p2)]
    try:
        owner.broadcast({"state": "READY", "rid": "r1"})
        assert _wait_for(lambda: any(m["state"] == "READY" for m in p1))
        assert _wait_for(lambda: any(m["state"] == "READY" for m in p2))
    finally:
        owner.stop()
        for peer in peers:
            peer.stop()


def test_peer_reconnect_receives_pending_message():
    """A message sent while the peer is down is delivered after it returns."""
    gid = f"zmqrc_{os.getpid()}"
    owner_seen, peer_seen = [], []
    owner = _make(0, 2, gid, owner_seen)
    peer = _make(1, 2, gid, peer_seen)
    try:
        owner.send(1, {"state": "READY", "rid": "warm"})
        assert _wait_for(lambda: peer_seen)

        peer.stop()  # peer goes away
        time.sleep(0.2)
        owner.send(1, {"state": "READY", "rid": "r2"})

        peer = _make(1, 2, gid, peer_seen)  # same identity, reconnects
        assert _wait_for(lambda: any(m.get("rid") == "r2" for m in peer_seen)), (
            "message sent while the peer was down was not delivered on reconnect"
        )
    finally:
        owner.stop()
        if peer is not None:
            peer.stop()


def test_owner_hb_and_peer_stale_warning(short_timer):
    gid = f"zmqhb_{os.getpid()}"
    owner_seen, peer_seen = [], []
    owner = _make(0, 2, gid, owner_seen)
    peer = _make(1, 2, gid, peer_seen)
    try:
        assert _wait_for(lambda: any(m["state"] == "HB" for m in peer_seen), 3.0), (
            "owner never broadcast HB"
        )
    finally:
        owner.stop()
        peer.stop()


def test_send_wakes_the_poller_immediately(monkeypatch):
    """A long poll timeout must not delay a queued send (self-pipe wakeup)."""
    monkeypatch.setattr(arena_control, "_POLL_MS", 5000)
    gid = f"zmqwk_{os.getpid()}"
    peer_seen = []
    owner = _make(0, 2, gid, [])
    peer = _make(1, 2, gid, peer_seen)
    try:
        t0 = time.monotonic()
        owner.send(1, {"state": "READY", "rid": "wake"})
        assert _wait_for(lambda: peer_seen, 3.0), "wake pipe did not deliver"
        # Without the wake pipe a successful send would sit in the queue
        # until the 5s poll timeout expired.
        assert time.monotonic() - t0 < 3.0, "send waited for the poll timeout"
    finally:
        owner.stop()
        peer.stop()


def test_stop_is_idempotent_and_unlinks_ipc():
    gid = f"zmqst_{os.getpid()}"
    channel = _make(0, 1, gid, [])
    path = channel._ipc_path
    channel.stop()
    channel.stop()  # second stop must not raise
    assert not os.path.exists(path)


def test_send_after_stop_is_dropped():
    gid = f"zmqsp_{os.getpid()}"
    channel = _make(0, 2, gid, [])
    channel.stop()
    channel.send(1, {"state": "READY", "rid": "r1"})  # must not raise


class _FakeSendSock:
    """send_multipart stand-in: fails for configured peer routing ids."""

    def __init__(self, failing_ranks):
        self.failing = {f"rank{rank}".encode() for rank in failing_ranks}
        self.sent = []

    def send_multipart(self, frames, flags=0):
        if frames[0] in self.failing:
            raise zmq.ZMQError(zmq.EHOSTUNREACH)
        self.sent.append(frames)


def _offline_owner(world_size=3):
    channel = ArenaControlChannel(
        rank=0, world_size=world_size, group_id="hol", apply_fn=lambda msg: None
    )
    return channel


def test_unreachable_peer_does_not_block_other_peers():
    channel = _offline_owner()
    try:
        channel.send(1, {"state": "READY", "rid": "to1"})
        channel.send(2, {"state": "READY", "rid": "to2"})
        sock = _FakeSendSock(failing_ranks=[1])
        channel._drain_outbound(sock)
        # rank 2 got its message even though rank 1 failed first in the loop.
        delivered = [frames for frames in sock.sent]
        assert [f[0] for f in delivered] == [b"rank2"]
        assert len(channel._outq[1]) == 1
        assert not channel._outq[2]
        assert channel._retry_after[1] > time.monotonic()
    finally:
        channel.stop()


def test_permanently_unreachable_peer_is_dropped(monkeypatch):
    monkeypatch.setattr(arena_control, "_PEER_STALE_S", 60.0)
    channel = _offline_owner()
    try:
        channel.send(1, {"state": "READY", "rid": "a"})
        channel.send(2, {"state": "READY", "rid": "b"})
        sock = _FakeSendSock(failing_ranks=[1])
        channel._drain_outbound(sock)  # records unreachable_since
        assert 1 in channel._outq and channel._outq[1]
        # Pretend the peer has been unreachable for longer than the limit and
        # let the next attempt notice it.
        channel._unreachable_since[1] = time.monotonic() - 120.0
        channel._retry_after[1] = 0.0  # skip the backoff window
        channel._drain_outbound(sock)  # stale -> drop rank 1's backlog
        assert 1 not in channel._outq or not channel._outq[1]
        assert not channel._outq[2]
        assert 1 not in channel._unreachable_since
        assert 1 not in channel._retry_after
    finally:
        channel.stop()
