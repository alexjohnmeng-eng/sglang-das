"""Unit tests for the shared host embedding arena and its staging path.

Covers the behaviours the EPD shared-arena work depends on:

* the tick gate is rank-symmetric (a wall-clock gate desynchronised the TP
  process group that the scheduler also broadcasts on);
* a normal completion frees its slot immediately, while an uncertain release
  keeps the reuse quarantine and a /send that could still be writing retires
  the range for good;
* every failing rank acknowledges (a lost ack used to strand a 143MB slot
  until the TTL abandoned it);
* the CP row predicate activates on the platform CP path, not only under
  CP-v2, so each rank stages its own shard of the embedding;
* the scatter stages only the kept rows and reports 0 when a rank keeps none;
* /send is idempotent per destination, and a duplicated completion ack is
  ignored instead of failing the request.

Run: pytest -q test/srt/test_shared_embedding_arena.py
"""

import asyncio
import threading
from collections import OrderedDict

import aiohttp
import pytest
import torch

from sglang.srt.disaggregation.encoder import shared_arena as sa
from sglang.srt.disaggregation.encoder.receiver import (
    DuplicateEmbeddingPartError,
    SharedWaitingRDMARequest,
    _aggregate_embedding_part,
    _send_may_have_written,
)
from sglang.srt.disaggregation.encoder.server import (
    MMEncoder,
    ReqState,
    SendDestination,
)
from sglang.srt.disaggregation.encoder.shared_arena import SharedEmbeddingArena
from sglang.srt.managers.mm_utils import _scatter_mm_embedding


class _FakeGroup:
    world_size = 2
    cpu_group = None


def _bare_arena(**overrides) -> SharedEmbeddingArena:
    """Arena with the state the tested methods touch, without IPC/pts."""
    arena = SharedEmbeddingArena.__new__(SharedEmbeddingArena)
    arena.group = _FakeGroup()
    arena.rank = 0
    arena.world_size = 2
    arena.is_owner = True
    arena.size_bytes = 256 * 1024 * 1024
    arena._lock = threading.Lock()
    arena._slots = {}
    arena._free = [(sa._align_up(4096, sa._ALIGN), arena.size_bytes - sa._align_up(4096, sa._ALIGN))]
    arena._quarantine = []
    arena._requests = {}
    arena._publish = {}
    arena._assigned = {}
    arena._data_ready = set()
    arena._failed = {}
    arena._ready_out = []
    arena._failed_out = {}
    arena._wait_events = {}
    arena._freed_pending = []
    arena._ack_waiters = []
    arena._acks_ready = []
    arena._acks_seen = {}
    arena._pending_since = {}
    arena._dead = set()
    arena._dead_order = []
    arena._total_inflight = 0
    arena._peak_inflight = 0
    arena._alloc_waits = 0
    arena._alloc_timeouts = 0
    arena._force_frees = 0
    arena._abandoned_bytes = 0
    arena._immediate_frees = 0
    arena._tick_calls = 0
    arena._drain_unknown = set()
    arena._unsafe_acks = []
    arena._reuse_unsafe = set()
    arena._tick_ms_sum = 0.0
    arena._tick_ms_max = 0.0
    arena._tick_ms_n = 0
    arena._last_stats = 0.0
    for key, value in overrides.items():
        setattr(arena, key, value)
    return arena


def _alloc(arena, rid, nbytes):
    slot = sa._Slot(offset=arena._try_alloc_locked(nbytes), nbytes=nbytes, rid=rid,
                    allocated_at=0.0)
    arena._slots[rid] = slot
    arena._assigned[rid] = (slot.offset, nbytes)
    arena._total_inflight += sa._align_up(nbytes, sa._ALIGN)
    arena._acks_seen[rid] = set()
    return slot


# ---------------------------------------------------------------- gate ----
def test_tick_gate_is_rank_symmetric(monkeypatch):
    monkeypatch.setattr(sa.envs.SGLANG_EMBEDDING_POOL_TICK_EVERY, "get", lambda: 3)
    a, b = _bare_arena(), _bare_arena()
    # Every rank calls tick once per scheduler iteration, so the sequence of
    # enter/skip decisions must be identical.
    assert [a._should_tick() for _ in range(9)] == [b._should_tick() for _ in range(9)]
    assert [a._should_tick() for _ in range(9)] == [False, False, True] * 3

    monkeypatch.setattr(sa.envs.SGLANG_EMBEDDING_POOL_TICK_EVERY, "get", lambda: 1)
    c = _bare_arena()
    assert all(c._should_tick() for _ in range(5))


# ------------------------------------------------------------- release ----
@pytest.fixture(autouse=True)
def _no_cuda_fence(monkeypatch):
    """release() records a CUDA event when one is not supplied; the unit tests
    run without touching the device."""
    monkeypatch.setattr(sa.torch.cuda, "is_available", lambda: False)


def test_successful_release_frees_without_quarantine(monkeypatch):
    monkeypatch.setattr(sa.envs.SGLANG_EMBEDDING_POOL_REUSE_DELAY_S, "get", lambda: 2.0)
    arena = _bare_arena()
    slot = _alloc(arena, "r1", 1024)

    arena.release("r1", ready_event=None, safe_reuse=True)
    rid, safe = arena._acks_ready[0]
    assert (rid, safe) == ("r1", True)

    acks = list(arena._acks_ready)   # every rank reports the same ack
    arena._record_peer_ack(0, acks, arena._unsafe_acks)
    arena._record_peer_ack(1, acks, [])
    arena._owner_reap()
    assert "r1" not in arena._slots
    assert arena._quarantine == []          # reused immediately
    assert arena._immediate_frees == 1
    assert arena._free                      # range is back on the free list
    assert slot.offset is not None


def test_unsafe_release_keeps_quarantine(monkeypatch):
    monkeypatch.setattr(sa.envs.SGLANG_EMBEDDING_POOL_REUSE_DELAY_S, "get", lambda: 2.0)
    arena = _bare_arena()
    _alloc(arena, "r2", 1024)
    arena.release("r2", safe_reuse=False)   # abort / GC path
    acks = list(arena._acks_ready)   # every rank reports the same ack
    arena._record_peer_ack(0, acks, arena._unsafe_acks)
    arena._record_peer_ack(1, acks, [])
    arena._owner_reap()
    assert "r2" not in arena._slots
    assert len(arena._quarantine) == 1      # still parked for the reuse delay
    assert arena._immediate_frees == 0


def test_send_failure_without_drain_proof_abandons(monkeypatch):
    monkeypatch.setattr(
        sa.envs.SGLANG_EMBEDDING_POOL_ABANDON_ON_SEND_FAILURE, "get", lambda: True
    )
    arena = _bare_arena()
    _alloc(arena, "r3", 4096)
    arena.release("r3", drain_unknown=True, safe_reuse=False)
    acks = list(arena._acks_ready)   # every rank reports the same ack
    arena._record_peer_ack(0, acks, arena._unsafe_acks)
    arena._record_peer_ack(1, acks, [])
    arena._owner_reap()
    assert "r3" not in arena._slots
    assert arena._force_frees == 1
    assert arena._abandoned_bytes == 4096
    assert arena._quarantine == []
    assert "r3" in arena._dead            # never reallocated


@pytest.mark.parametrize("safe", [True, False])
def test_one_unsafe_ack_quarantines_for_the_whole_group(safe):
    arena = _bare_arena()
    _alloc(arena, "r4", 1024)
    arena.release("r4", safe_reuse=safe)
    acks = list(arena._acks_ready)   # every rank reports the same ack
    arena._record_peer_ack(0, acks, arena._unsafe_acks)
    arena._record_peer_ack(1, acks, [])
    arena._owner_reap()
    assert ("r4" not in arena._slots)
    assert bool(arena._quarantine) is (not safe)


# ------------------------------------------------------------ cp rows ----
class _FakeStrategy:
    cp_size = 8
    cp_rank = 3

    def row_keep_mask(self, rows):
        return (rows % self.cp_size) == self.cp_rank


class _FakeForwardMode:
    def is_context_parallel_extend(self):
        return True


class _FakeForwardBatch:
    def __init__(self, cp_metadata=None):
        self.attn_cp_metadata = cp_metadata
        self.forward_mode = _FakeForwardMode()


def test_row_keep_fn_none_on_platform_cp(monkeypatch):
    """The platform CP path keeps every row.

    On HCU without SGLANG_ENABLE_CP_V2 the model consumes the full batch on
    every rank (only MoE routing is strided), so staging "this rank's share"
    would drop rows the model still reads. The predicate must stay off even
    when the batch carries attention CP metadata.
    """
    from sglang.srt.layers.cp import utils as cp_utils

    monkeypatch.setattr(cp_utils, "get_cp_strategy", lambda: _FakeStrategy())
    monkeypatch.setattr(cp_utils, "is_cp_active", lambda batch: False)
    assert cp_utils.cp_row_keep_fn(_FakeForwardBatch(cp_metadata=object())) is None
    assert cp_utils.cp_row_keep_fn(_FakeForwardBatch(cp_metadata=None)) is None


def test_row_keep_fn_returns_strategy_predicate_when_cp_active(monkeypatch):
    from sglang.srt.layers.cp import utils as cp_utils

    monkeypatch.setattr(cp_utils, "get_cp_strategy", lambda: _FakeStrategy())
    monkeypatch.setattr(cp_utils, "is_cp_active", lambda batch: True)
    keep = cp_utils.cp_row_keep_fn(_FakeForwardBatch())
    assert keep is not None
    assert keep(torch.arange(16)).tolist() == [(r % 8) == 3 for r in range(16)]


# ------------------------------------------------------------- scatter ----
def test_scatter_stages_only_kept_rows():
    hidden = 4
    src = torch.arange(16 * hidden, dtype=torch.float32).reshape(16, hidden)
    mask = torch.ones(16, dtype=torch.bool)
    dest = torch.zeros(16, hidden)
    keep = _FakeStrategy().row_keep_mask

    kept = _scatter_mm_embedding(dest, mask, src, row_index=torch.arange(16), keep_fn=keep)
    assert kept == 2                       # rows 3 and 11
    rows = [3, 11]
    for row in rows:
        assert torch.equal(dest[row], src[row])
    untouched = [r for r in range(16) if r not in rows]
    assert torch.count_nonzero(dest[untouched]) == 0


def test_scatter_returns_zero_when_nothing_kept():
    src = torch.ones(4, 2)
    dest = torch.zeros(4, 2)
    keep = lambda rows: torch.zeros_like(rows, dtype=torch.bool)
    assert _scatter_mm_embedding(dest, torch.ones(4, dtype=torch.bool), src,
                                 row_index=torch.arange(4), keep_fn=keep) == 0
    assert torch.count_nonzero(dest) == 0


def test_scatter_without_keep_copies_every_row():
    src = torch.arange(8.0).reshape(4, 2)
    dest = torch.zeros(4, 2)
    kept = _scatter_mm_embedding(dest, torch.ones(4, dtype=torch.bool), src)
    assert kept == 4
    assert torch.equal(dest, src)


# -------------------------------------------------------- send / parts ----
class _FakeDelivery:
    def __init__(self):
        self.sent = []

    async def send(self, state, destination):
        await asyncio.sleep(0.005)
        self.sent.append(destination.idempotency_key())


def test_send_to_destination_is_idempotent():
    enc = MMEncoder.__new__(MMEncoder)
    enc.req_states = {}
    enc.delivery = _FakeDelivery()
    state = ReqState(req_id="r1")
    enc.req_states["r1"] = state
    dest = SendDestination(endpoint="h:1", session_id="s", buffer_address=4096)

    async def run():
        await enc.send_to_destination(state, dest)
        await enc.send_to_destination(state, dest)                     # retry
        await asyncio.gather(                                           # racing retry
            enc.send_to_destination(state, dest),
            enc.send_to_destination(state, dest),
        )
        assert len(enc.delivery.sent) == 1
        other = SendDestination(endpoint="h:1", session_id="s", buffer_address=8192)
        await enc.send_to_destination(state, other)                     # another part
        assert len(enc.delivery.sent) == 2
        assert state.active_sends == 0

    asyncio.run(run())


class _FakeAggregate:
    req_id = "r2"

    def __init__(self):
        self.added = 0

    def add(self, part):
        if part.duplicate:
            raise DuplicateEmbeddingPartError("duplicate embedding part 0")
        self.added += 1


class _FakePart:
    part_idx = 0

    def __init__(self, duplicate):
        self.duplicate = duplicate


def test_duplicate_part_is_ignored():
    current = _FakeAggregate()
    assert _aggregate_embedding_part(current, _FakePart(False), None) is current
    assert _aggregate_embedding_part(current, _FakePart(True), None) is current
    assert current.added == 1


class _FakeArena:
    def __init__(self):
        self.releases = []

    def release(self, rid, drain_unknown=False, **kwargs):
        self.releases.append((rid, drain_unknown))


def _bare_shared_request(**attrs):
    """A SharedWaitingRDMARequest with only the fields release paths touch."""
    import threading

    req = object.__new__(SharedWaitingRDMARequest)
    req.rid = "rid-1"
    req._pool_slot_id = "rid-1"
    req.embeddings_buffer = None
    req._receive_running = False
    req._failed_acked = False
    req._drain_unknown = False
    req._lease_bound = False
    req.registration_future = None
    req._mm_finalizer = None
    req._buffer_lock = threading.Lock()
    req.embedding_pool = None
    req.arena = _FakeArena()
    for key, value in attrs.items():
        setattr(req, key, value)
    return req


def test_bound_lease_blocks_the_receive_thread_ack():
    """The consumer, not the receive thread, owns a bound slot's release.

    The scheduler may bind (and detach the buffer) before the receive thread
    reaches its finally block; acking there frees the range while prefill is
    still reading it through the lease.
    """
    req = _bare_shared_request(_pool_slot_id=None, _lease_bound=True)
    assert req._should_ack_on_thread_exit() is False


def test_unbound_request_still_acks_on_thread_exit():
    req = _bare_shared_request(_pool_slot_id=None, _lease_bound=False)
    assert req._should_ack_on_thread_exit() is True


def test_bound_lease_release_resources_does_not_ack_twice():
    req = _bare_shared_request(_pool_slot_id=None, _lease_bound=True)
    req.release_resources()
    assert req.arena.releases == []


def test_failed_slot_acks_once_with_drain_unknown():
    """Cleanup must not emit a plain ACK before the unsafe one."""
    req = _bare_shared_request(_drain_unknown=True)
    req._cleanup_gpu_buffer()
    req.release_resources()
    assert req.arena.releases == [("rid-1", True)]
    assert req._pool_slot_id is None


def test_send_may_have_written_classification():
    class _Resp:
        def __init__(self, status):
            self.status = status

    assert _send_may_have_written(asyncio.TimeoutError()) is True
    assert _send_may_have_written(aiohttp.ClientConnectionError("reset")) is True
    # The DP dispatcher reports a worker timeout as an HTTP 504 envelope.
    assert _send_may_have_written(_Resp(504)) is True
    assert _send_may_have_written(_Resp(500)) is True
    assert _send_may_have_written(_Resp(400)) is False
    assert _send_may_have_written(_Resp(200)) is False


def test_send_tombstone_survives_request_release():
    server = object.__new__(MMEncoder)
    server._completed_sends = OrderedDict()
    server._remember_completed_send("req-1", "ep|session=1|addr=4096|meta_only=0")
    assert server._send_already_completed("req-1", "ep|session=1|addr=4096|meta_only=0")
    assert not server._send_already_completed("req-1", "ep|session=2|addr=4096|meta_only=0")
    assert not server._send_already_completed("req-2", "ep|session=1|addr=4096|meta_only=0")


def test_cp_row_filter_only_for_shared_host_items():
    """GPU pool and ordinary CPU embeddings must stage every row.

    ``src.is_cpu`` is not a valid discriminator: a ZMQ CPU fallback is also a
    CPU tensor, but every rank needs its complete embedding there.
    """
    from sglang.srt.managers.mm_utils import _cp_keep_fn_for_items

    class _Item:
        def __init__(self, shared):
            self.is_shared_host_embedding = shared

    keep = lambda rows: rows  # noqa: E731 - identity predicate

    assert _cp_keep_fn_for_items(keep, [_Item(True), _Item(True)]) is keep
    assert _cp_keep_fn_for_items(keep, [_Item(True), _Item(False)]) is None
    assert _cp_keep_fn_for_items(keep, [_Item(False)]) is None
    assert _cp_keep_fn_for_items(None, [_Item(True)]) is None
    assert _cp_keep_fn_for_items(keep, []) is None


def test_shared_arena_is_marked_as_shared_host_pool():
    from sglang.srt.disaggregation.encoder.shared_arena import SharedEmbeddingArena

    assert SharedEmbeddingArena.is_shared_host is True


def test_wait_data_ready_is_woken_by_the_owner_signal():
    """The receive thread blocks on an Event, not a 2ms poll."""
    import time as _time

    arena = _bare_arena()
    result = {}

    def waiter():
        t0 = _time.monotonic()
        result["ok"] = arena.wait_data_ready("rid-evt", timeout=5)
        result["elapsed"] = _time.monotonic() - t0

    thread = threading.Thread(target=waiter)
    thread.start()
    _time.sleep(0.05)
    arena.mark_data_ready("rid-evt")
    thread.join(timeout=2)
    assert result["ok"] is True
    assert result["elapsed"] < 1.0  # woke on the signal, not on the timeout


def test_wait_data_ready_failure_wakes_and_returns_false():
    arena = _bare_arena()
    result = {}

    def waiter():
        result["ok"] = arena.wait_data_ready("rid-fail", timeout=5)

    thread = threading.Thread(target=waiter)
    thread.start()
    import time as _time

    _time.sleep(0.05)
    arena.mark_failed("rid-fail", "encoder refused")
    thread.join(timeout=2)
    assert result["ok"] is False


def test_ready_and_failed_deltas_are_published_once():
    arena = _bare_arena()
    arena.mark_data_ready("rid-d")
    arena.mark_data_ready("rid-d")
    arena.mark_failed("rid-f", "x")
    arena.mark_failed("rid-f", "x")
    assert arena._ready_out == ["rid-d"]
    assert arena._failed_out == {"rid-f": "x"}


def test_peer_adopts_owner_assignment_and_resolves_waiter():
    """Peers no longer publish pending work; the owner's assignment wakes them."""
    from sglang.srt.disaggregation.encoder.shared_arena import SharedEmbeddingArena

    arena = _bare_arena(is_owner=False)
    arena._requests = {}
    got = {}

    def peer_request():
        got["slot"] = arena.request_slot("rid-peer", 4096, timeout=5)

    thread = threading.Thread(target=peer_request)
    thread.start()
    import time as _time

    _time.sleep(0.05)
    arena._adopt("rid-peer", (8192, 4096))
    thread.join(timeout=2)
    assert got["slot"] == (8192, 4096)


def test_release_waits_for_receive_thread_before_acking():
    """A failed request must not ack while /send may still be in flight.

    The base cleanup latches _terminal and defers the release to the thread's
    exit hook; the arena ack therefore sees the final drain_unknown value.
    """
    req = _bare_shared_request(_receive_running=True)
    req._cleanup_gpu_buffer()
    assert req._terminal is True
    assert req.arena.releases == []  # still blocked on the receive thread

    # The thread finishes: its /send timed out, so the write is unproven.
    with req._buffer_lock:
        req._receive_running = False
        req._drain_unknown = True
        req._release_buffer_locked()
    assert req.arena.releases == [("rid-1", True)]


def test_public_send_answers_released_retry_from_tombstone():
    from sglang.srt.disaggregation.encoder.server import MMEncoder, SendDestination

    enc = object.__new__(MMEncoder)
    enc.req_states = {}
    enc._completed_sends = OrderedDict()
    dest = SendDestination.from_host_port(
        "10.0.0.1", 30005, session_id="s1", buffer_address=4096, meta_only=False
    )
    args = dict(
        req_id="req-send",
        prefill_host="10.0.0.1",
        embedding_port=30005,
        session_id="s1",
        buffer_address=4096,
        meta_only=False,
    )
    from sglang.srt.disaggregation.encoder.server import SEND_ALREADY_COMPLETED

    assert asyncio.run(enc.send(**args)) is False  # unknown request
    enc._remember_completed_send("req-send", dest.idempotency_key())
    got = asyncio.run(enc.send(**args))
    assert got is SEND_ALREADY_COMPLETED  # exact retry after release
    args2 = dict(args, buffer_address=8192)
    assert asyncio.run(enc.send(**args2)) is False  # different destination


def test_allocation_timeout_publishes_failed_delta():
    import time as _time

    arena = _bare_arena()
    arena._pending_since = {"rid-to": _time.monotonic() - 3600}
    arena._alloc_timeouts = 0
    failed = []

    class _Req:
        def fail(self, reason):
            failed.append(reason)

    arena._requests = {"rid-to": _Req()}
    arena._expire_pending_locked("rid-to", _time.monotonic())
    assert "rid-to" in arena._failed_out  # peers learn about the refusal
    assert failed and "full" in failed[0]


def test_prune_keeps_events_for_rids_still_waiting():
    arena = _bare_arena()
    arena._wait_events = {
        "rid-ready": threading.Event(),
        "rid-waiting": threading.Event(),
    }
    arena._data_ready = {"rid-ready"}
    arena._requests = {}  # assignment waiter already resolved
    old_keep = sa._STATE_KEEP
    sa._STATE_KEEP = 1
    try:
        arena._prune_state()
    finally:
        sa._STATE_KEEP = old_keep
    assert "rid-waiting" in arena._wait_events  # still blocking on DATA_READY
    assert "rid-ready" not in arena._wait_events  # already signalled


def test_owner_prunes_acks_for_slotless_failed_requests():
    """An ACK from a request that never got a slot must not leak on the owner.

    Early ACKs (rid still pending allocation) must be kept.
    """
    arena = _bare_arena()
    arena._acks_seen = {"rid-fail": {0, 1}, "rid-early": {1}}
    arena._failed = {"rid-fail": "shared arena full"}
    arena._requests = {"rid-early": object()}
    arena._slots = {}
    old_keep = sa._STATE_KEEP
    sa._STATE_KEEP = 1
    try:
        arena._prune_state()
    finally:
        sa._STATE_KEEP = old_keep
    assert "rid-fail" not in arena._acks_seen
    assert "rid-early" in arena._acks_seen
