"""Before/after demonstrations for the four shared-arena lifetime bugs.

Each test shows the legacy predicate/path exhibiting the bug and the shipped
code rejecting it, so a future regression is visible as a semantic change
rather than only as a missing call.
"""

import asyncio
import threading
from collections import OrderedDict

import aiohttp
import torch

from sglang.srt.disaggregation.encoder.receiver import (
    SharedWaitingRDMARequest,
    _send_may_have_written,
)
from sglang.srt.disaggregation.encoder.server import MMEncoder
from sglang.srt.managers.mm_schedule import _concat_or_view


class _FakeArena:
    def __init__(self):
        self.releases = []

    def release(self, rid, drain_unknown=False, **kwargs):
        self.releases.append((rid, drain_unknown))


def _bare_shared_request(**attrs):
    req = object.__new__(SharedWaitingRDMARequest)
    req.rid = "rid-1"
    req._pool_slot_id = None
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


def test_ack_predicate_before_and_after():
    """Legacy: "no buffer == done reading". Fixed: a bound lease keeps it."""
    req = _bare_shared_request(_lease_bound=True)
    legacy_acks = req._pool_slot_id is None and req.embeddings_buffer is None
    assert legacy_acks is True  # the bug: the consumer may still be reading
    assert req._should_ack_on_thread_exit() is False  # fixed


def test_failed_release_single_unsafe_ack_before_and_after():
    """Legacy emitted a plain ACK first, which the owner frees on."""
    legacy_acks = []

    def legacy_release(slot):
        legacy_acks.append(("plain", False))  # _cleanup_gpu_buffer(slot)
        legacy_acks.append(("unsafe", True))  # later _ack_once()

    legacy_release("rid-1")
    assert legacy_acks[0] == ("plain", False)  # owner would already free

    req = _bare_shared_request(_pool_slot_id="rid-1", _drain_unknown=True)
    req._cleanup_gpu_buffer()
    req.release_resources()
    assert req.arena.releases == [("rid-1", True)]  # fixed: one unsafe ACK


def test_http_5xx_classification_before_and_after():
    class _Resp:
        def __init__(self, status):
            self.status = status

    legacy_504 = isinstance(_Resp(504), (asyncio.TimeoutError, aiohttp.ClientError))
    assert legacy_504 is False  # the bug: dispatched worker timeout looked safe
    assert _send_may_have_written(_Resp(504)) is True  # fixed
    assert _send_may_have_written(_Resp(400)) is False


def test_send_retry_after_release_before_and_after():
    """Legacy dedup lived in ReqState; the fixed one keeps a tombstone."""
    enc = object.__new__(MMEncoder)
    enc.req_states = {}
    key = "ep|session=1|addr=4096|meta_only=0"

    legacy_ok = "req-1" in enc.req_states  # state already popped
    assert legacy_ok is False  # the bug: a delivered /send looked failed

    enc._completed_sends = OrderedDict()
    enc._remember_completed_send("req-1", key)
    assert enc._send_already_completed("req-1", key) is True  # fixed
    assert enc._send_already_completed("req-1", "other") is False


def test_sorted_view_would_reorder_rows_before_and_after():
    """Legacy sorted by storage offset; the fix keeps the caller's order."""
    base = torch.arange(16.0).reshape(4, 4)
    chunks = [base[2:4], base[0:2]]
    legacy = base.new_empty(0).set_(
        base.untyped_storage(), 0, (4, 4), (4, 1)
    )
    assert not torch.equal(legacy, torch.cat(chunks))  # the bug
    assert torch.equal(_concat_or_view(chunks), torch.cat(chunks))  # fixed
