"""One pinned host embedding arena shared by every TP/CP rank of a prefill node.

The EPD encoder pushes ViT embeddings over RDMA. Without sharing, every rank
receives its own copy (the encoder writes ``tp_size`` times) and keeps it in
device memory. This module keeps a single pinned host allocation instead:

* the arena lives in one ``hipHostMalloc`` region created by rank 0 and mapped
  into every rank through HSA IPC (``HostSharedMemoryManager(use_hsa_ipc=True)``,
  the same primitive the Engram host table uses);
* only the owner (rank 0) registers the arena with mooncake and receives the
  RDMA write -- one transfer per request instead of ``tp_size``;
* every rank reads its own virtual address of those physical pages, so slot
  contents need no further coordination.

Slot bookkeeping is request-scoped and single-writer: the owner allocates and
frees, the other ranks adopt the owner's decision. Both directions ride the
collective the scheduler already performs every iteration (``tick``), so there
is no cross-process lock, no shared free list and no cross-process CUDA event.

Lifecycle, in one place:

    require(rid)          receive thread asks for a slot
    tick()                owner allocates -> publishes (offset, nbytes)
    mark_data_ready(rid)  owner finished the RDMA write
    ... consumer stages its chunk rows to the GPU ...
    release(rid, event)   ack once the reader's stream event completes
    tick()                owner frees after every rank acknowledges

Anything that can be allocated must be released exactly once: the normal path,
every failure path and the lease GC finalizer all funnel into ``release``, and
the owner force-reclaims a slot whose acknowledgements stop arriving.
"""

from __future__ import annotations

import threading
import time
import weakref
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.utils.host_shared_memory import HostSharedMemoryManager
from sglang.utils import logger

_ALIGN = 256
# Request-state dicts keep the recent past for diagnostics; older entries are
# dropped so a long-running server does not grow without bound.
_STATE_KEEP = 4096


@dataclass
class _Slot:
    offset: int
    nbytes: int
    rid: str
    allocated_at: float


class _SlotRequest:
    """One rank's blocking wait for the owner's assignment."""

    __slots__ = ("rid", "nbytes", "deadline", "event", "result", "error")

    def __init__(self, rid: str, nbytes: int, deadline: float):
        self.rid = rid
        self.nbytes = nbytes
        self.deadline = deadline
        self.event = threading.Event()
        self.result: Optional[Tuple[int, int]] = None
        self.error: Optional[str] = None

    def resolve(self, offset: int, nbytes: int) -> None:
        self.result = (offset, nbytes)
        self.event.set()

    def fail(self, error: str) -> None:
        self.error = error
        self.event.set()


class SharedEmbeddingArena:
    """Pinned host slots shared by all ranks; see the module docstring."""

    # Tells the consumers that a view over this pool is rank-shared host
    # memory (CP row staging applies only to this kind of embedding).
    is_shared_host = True

    def __init__(
        self,
        group,
        rank_in_group: int,
        size_bytes: int,
        engine=None,
        name: str = "sglang_emb_pool",
    ):
        self.group = group
        self.rank = rank_in_group
        self.world_size = group.world_size
        self.is_owner = rank_in_group == 0
        self.size_bytes = size_bytes
        self.engine = engine

        self._manager = HostSharedMemoryManager(
            f"{name}_r{self.world_size}", group=group, use_hsa_ipc=True
        )
        self.buffer = self._manager.malloc(shape=(size_bytes,), dtype=torch.uint8)
        self.base = self.buffer.data_ptr()
        self._host_async = True
        if self.is_owner and engine is not None:
            engine.register(self.base, self.buffer.nbytes)

        self._lock = threading.Lock()
        # receive threads waiting for an assignment
        self._requests: Dict[str, _SlotRequest] = {}
        # owner: assignments to publish on the next tick
        self._publish: Dict[str, Tuple[int, int]] = {}
        # every rank: assignments adopted from the owner
        self._assigned: Dict[str, Tuple[int, int]] = {}
        # owner: live slots
        self._slots: Dict[str, _Slot] = {}
        # owner: free segments ready to hand out
        self._free: List[Tuple[int, int]] = [
            (_align_up(4096, _ALIGN), size_bytes - _align_up(4096, _ALIGN))
        ]
        # owner: (reusable_at, offset, length) freed but still quarantine
        self._quarantine: List[Tuple[float, int, int]] = []
        # shared state propagated by the owner
        self._data_ready: Set[str] = set()
        self._failed: Dict[str, str] = {}
        # tick publishes deltas, not snapshots: `ready`/`failed` grow on the
        # owner and are cleared after every publish.
        self._ready_out: List[str] = []
        self._failed_out: Dict[str, str] = {}
        # One waiter per rid instead of a 2ms poll in the receive thread.
        self._wait_events: Dict[str, threading.Event] = {}
        # owner: slots freed since the last tick, so peers can prune their
        # copies of the request state instead of growing without bound
        self._freed_pending: List[str] = []
        # released but not yet acknowledged upward (event-gated)
        self._ack_waiters: List[Tuple[str, torch.cuda.Event]] = []
        self._acks_ready: List[str] = []
        # ack sets the owner has seen so far, rid -> {rank}
        self._acks_seen: Dict[str, Set[int]] = {}
        # owner: request deadlines so a stuck alloc cannot hang a receive thread
        self._pending_since: Dict[str, float] = {}
        self._dead: Set[str] = set()
        self._dead_order: List[str] = []

        self._total_inflight = 0
        self._peak_inflight = 0
        self._alloc_waits = 0
        self._alloc_timeouts = 0
        self._force_frees = 0
        self._abandoned_bytes = 0
        # slots returned without the reuse delay (normal completions)
        self._immediate_frees = 0
        # tick collective cost, sampled when SGLANG_EMBEDDING_POOL_PROFILE=1
        self._tick_ms_sum = 0.0
        self._tick_ms_max = 0.0
        self._tick_ms_n = 0
        self._last_stats = time.monotonic()
        # Monotonic count of tick() calls; identical on every rank because the
        # scheduler loop calls it once per iteration everywhere.
        self._tick_calls = 0
        # rids whose /send died without proof that the encoder stopped writing
        # (owner frees those slots permanently instead of recycling them)
        self._drain_unknown: Set[str] = set()
        # rids at least one rank released without a safe-reuse proof; those
        # still go through the reuse-delay quarantine
        self._reuse_unsafe: Set[str] = set()
        # this rank's report of such rids, shipped on the next tick
        self._unsafe_acks: List[str] = []

        logger.info(
            "SharedEmbeddingArena ready: rank=%d/%d owner=%s size=%dMB base=0x%x "
            "rdma_registered=%s",
            self.rank,
            self.world_size,
            self.is_owner,
            size_bytes // (1024 * 1024),
            self.base,
            bool(self.is_owner and engine is not None),
        )

    # ------------------------------------------------------------------ #
    # allocation
    # ------------------------------------------------------------------ #
    def request_slot(
        self, rid: str, nbytes: int, timeout: Optional[float] = None
    ) -> Optional[Tuple[int, int]]:
        """Wait for the owner's slot assignment; returns ``(offset, nbytes)``.

        Only the owner runs the allocator (it is the single RDMA writer and
        knows every rid/size); a peer calls this to register its local waiter
        and is resolved by the assignment the owner publishes on the next
        tick. It blocks the receive thread for at most one scheduler
        iteration plus whatever the arena is full for.
        """
        if timeout is None:
            timeout = envs.SGLANG_EMBEDDING_POOL_ALLOC_TIMEOUT_S.get()
        if nbytes > self.size_bytes:
            logger.error(
                "SharedEmbeddingArena: request %dMB exceeds arena %dMB",
                nbytes // (1024 * 1024),
                self.size_bytes // (1024 * 1024),
            )
            return None
        req = _SlotRequest(rid, nbytes, time.monotonic() + timeout)
        with self._lock:
            if rid in self._dead:
                return None
            # The owner may have allocated this request before this rank's
            # receive thread got here (it allocates from the union of every
            # rank's pending list). Adopt the published offset now: `tick`
            # never republishes an assigned rid, so waiting would hang.
            assigned = self._assigned.get(rid)
            if assigned is not None:
                return assigned
            self._requests[rid] = req
        if not req.event.wait(timeout):
            with self._lock:
                self._requests.pop(rid, None)
                self._alloc_timeouts += 1
            logger.error(
                "SharedEmbeddingArena: slot for rid=%s (%dMB) timed out after %.1fs",
                rid,
                nbytes // (1024 * 1024),
                timeout,
            )
            return None
        with self._lock:
            self._requests.pop(rid, None)
        if req.error is not None:
            logger.error("SharedEmbeddingArena: slot for rid=%s failed: %s", rid, req.error)
            return None
        return req.result

    def view(self, offset: int, nbytes: int) -> torch.Tensor:
        return self.buffer[offset : offset + nbytes]

    # ------------------------------------------------------------------ #
    # readiness / failure
    # ------------------------------------------------------------------ #
    def _signal_waiters_locked(self, rid: str) -> None:
        event = self._wait_events.get(rid)
        if event is not None:
            event.set()

    def mark_data_ready(self, rid: str) -> None:
        with self._lock:
            if rid not in self._data_ready:
                self._data_ready.add(rid)
                self._ready_out.append(rid)
            self._signal_waiters_locked(rid)

    def _mark_failed_locked(self, rid: str, reason: str) -> None:
        """Record FAILED once, in the state *and* in the tick delta.

        Every failure entry point (receiver ``mark_failed``, allocation
        timeout) must go through here: writing ``_failed`` directly used to
        leave ``_failed_out`` empty, so peers never learned that the owner
        had already refused the slot.
        """
        if rid in self._failed:
            return
        self._failed[rid] = reason or "shared arena failure"
        self._failed_out[rid] = self._failed[rid]
        self._signal_waiters_locked(rid)

    def mark_failed(self, rid: str, reason: str = "") -> None:
        with self._lock:
            self._mark_failed_locked(rid, reason or "shared arena failure")

    def is_data_ready(self, rid: str) -> bool:
        with self._lock:
            return rid in self._data_ready

    def wait_data_ready(self, rid: str, timeout: float) -> bool:
        """Block until the owner publishes READY/FAILED for ``rid``."""
        with self._lock:
            if rid in self._data_ready:
                return True
            if rid in self._failed:
                return False
            event = self._wait_events.setdefault(rid, threading.Event())
        if not event.wait(timeout):
            with self._lock:
                # Keep the event if the answer arrived just after the wait:
                # the next call returns from the state check above.
                if rid not in self._data_ready and rid not in self._failed:
                    self._wait_events.pop(rid, None)
            return False
        with self._lock:
            return rid in self._data_ready

    def failure_reason(self, rid: str) -> str:
        with self._lock:
            return self._failed.get(rid, "shared arena failure")

    # ------------------------------------------------------------------ #
    # release (duck-typed EmbeddingPool API, so leases work unchanged)
    # ------------------------------------------------------------------ #
    def release(
        self,
        rid: str,
        ready_event: Optional[torch.cuda.Event] = None,
        drain_unknown: bool = False,
        safe_reuse: bool = False,
    ) -> None:
        """Acknowledge a slot once this rank no longer reads it.

        ``drain_unknown`` says the encoder's write could not be proven
        finished (the /send died with a timeout or a dropped connection). The
        ack only covers the local reader, so the owner must retire the range
        permanently instead of putting it back in the free list.

        ``safe_reuse`` marks the normal path: the request completed, its RDMA
        write was proven by DATA_READY, and every reader is fenced by
        ``ready_event``. The owner can then hand the range out again right
        away instead of parking it in the reuse-delay quarantine, which would
        otherwise tax every request 2s of arena capacity.
        """
        if ready_event is None and torch.cuda.is_available():
            # GC / abort path: fence conservatively on the current stream so a
            # copy queued before the release cannot be overwritten.
            try:
                ready_event = torch.cuda.Event()
                ready_event.record(torch.cuda.current_stream())
            except Exception:
                logger.warning(
                    "SharedEmbeddingArena: could not record a release fence",
                    exc_info=True,
                )
                ready_event = None
        with self._lock:
            if drain_unknown:
                self._unsafe_acks.append(rid)
            if ready_event is not None and not ready_event.query():
                self._ack_waiters.append((rid, ready_event, safe_reuse))
            else:
                self._acks_ready.append((rid, safe_reuse))

    def release_on_gc(self, obj, rid: str):
        return weakref.finalize(obj, self.release, rid)

    # ------------------------------------------------------------------ #
    # the per-iteration collective
    # ------------------------------------------------------------------ #
    def _should_tick(self) -> bool:
        """Advance the call counter and decide whether to enter the collective.

        The answer depends only on this rank's call index, which is identical
        on every rank because the scheduler loop calls ``tick`` once per
        iteration everywhere. Any wall-clock or workload-dependent gate would
        let ranks issue different collectives in the same slot.
        """
        self._tick_calls += 1
        every = envs.SGLANG_EMBEDDING_POOL_TICK_EVERY.get()
        return every <= 1 or (self._tick_calls % every) == 0

    def tick(self) -> None:
        """Exchange allocations / acknowledgements with the other ranks.

        Called from every rank exactly once per scheduler iteration. The
        collective uses the TP CPU process group, which the scheduler also
        uses for its own broadcasts, so the decision to enter must be
        identical on every rank: a rank-local wall-clock gate would let one
        rank issue all_gather while another issues broadcast and desynchronise
        the group. The call counter below is such a gate because every rank
        runs the same number of scheduler iterations.

        The owner's assignments are published in the payload, so a peer learns
        them one tick later -- one scheduler iteration, against the hundreds
        of milliseconds an embedding transfer takes.
        """
        if not self._should_tick():
            return
        self._drain_acks()
        with self._lock:
            # Only the owner allocates (it already knows the rid and size and
            # is the single RDMA writer); peers just wait for the assignment
            # it publishes.
            pending = (
                [
                    (rid, req.nbytes)
                    for rid, req in self._requests.items()
                    if rid not in self._assigned and rid not in self._dead
                ]
                if self.is_owner
                else []
            )
            ack_payload = self._acks_ready
            self._acks_ready = []
            unsafe_payload = self._unsafe_acks
            self._unsafe_acks = []
            assign_payload = self._publish
            self._publish = {}
            # Deltas, not snapshots: the ready/failed sets only grow, so
            # republishing them every iteration burns serialization time.
            state_payload = {
                "ready": tuple(self._ready_out),
                "failed": dict(self._failed_out),
                "freed": tuple(self._freed_pending),
            }
            self._ready_out = []
            self._failed_out = {}
            self._freed_pending = []
            inflight = len(self._slots) if self.is_owner else 0

        payload = {
            "rank": self.rank,
            "req": pending,
            "ack": ack_payload,
            "unsafe": unsafe_payload,
            "assign": assign_payload,
            "state": state_payload if self.is_owner else None,
            "inflight": inflight,
        }
        gathered = [None] * self.world_size
        t_gather = time.perf_counter()
        torch.distributed.all_gather_object(
            gathered, payload, group=self.group.cpu_group
        )
        if envs.SGLANG_EMBEDDING_POOL_PROFILE.get():
            gather_ms = (time.perf_counter() - t_gather) * 1000.0
            self._tick_ms_sum += gather_ms
            self._tick_ms_max = max(self._tick_ms_max, gather_ms)
            self._tick_ms_n += 1
            if self._tick_ms_n % 200 == 0:
                logger.info(
                    "[pool-profile] tick avg_ms=%.2f max_ms=%.2f n=%d "
                    "pending=%d acks=%d",
                    self._tick_ms_sum / self._tick_ms_n,
                    self._tick_ms_max,
                    self._tick_ms_n,
                    len(pending),
                    len(ack_payload),
                )

        # ---- merge ---------------------------------------------------- #
        all_reqs: List[Tuple[str, int]] = []
        states = None
        for item in gathered:
            if not item:
                continue
            all_reqs.extend(item["req"])
            if self.is_owner:
                # Ack tally is owner-side state; peers would only grow it.
                self._record_peer_ack(item["rank"], item["ack"], item.get("unsafe"))
            for rid, assignment in (item["assign"] or {}).items():
                self._adopt(rid, assignment)
            if item["state"] is not None:
                states = item["state"]

        if states is not None and not self.is_owner:
            with self._lock:
                for rid in states["ready"]:
                    if rid not in self._data_ready:
                        self._data_ready.add(rid)
                    self._signal_waiters_locked(rid)
                for rid, reason in states["failed"].items():
                    if rid not in self._failed:
                        self._failed[rid] = reason
                    self._signal_waiters_locked(rid)
                for rid in states.get("freed", ()):
                    self._data_ready.discard(rid)
                    self._failed.pop(rid, None)
                    self._assigned.pop(rid, None)
                    self._wait_events.pop(rid, None)

        if self.is_owner:
            self._owner_allocate(all_reqs)
            self._owner_reap()
            self._prune_state()
        else:
            self._fail_dead_requests()
            self._prune_state()
        self._maybe_log_stats()

    def _prune_state(self) -> None:
        """Bound the per-rid state a long-running server accumulates.

        ``_dead`` only guards against a re-allocation that can never happen;
        ``_failed`` only serves receive threads polling right now. Both are
        capped so days of traffic cannot bloat the tick payload.
        """
        with self._lock:
            while len(self._dead_order) > _STATE_KEEP:
                self._dead.discard(self._dead_order.pop(0))
            if len(self._failed) > _STATE_KEEP:
                excess = len(self._failed) - _STATE_KEEP
                for rid in list(self._failed)[:excess]:
                    self._failed.pop(rid, None)
            # ACKs from requests that never received a slot (metadata pull
            # failed, oversize, allocation timeout) have no slot to reap, so
            # the owner would keep their _acks_seen entries forever. Keep the
            # early-ACK case: a rid that may still be allocated (pending,
            # published, assigned or live slot) retains its tally.
            if self.is_owner:
                for rid in list(self._acks_seen):
                    if rid in self._slots:
                        continue
                    if rid in self._dead:
                        self._acks_seen.pop(rid, None)
                if len(self._acks_seen) > _STATE_KEEP:
                    for rid in list(self._acks_seen):
                        if (
                            rid in self._slots
                            or rid in self._publish
                            or rid in self._assigned
                            or rid in self._requests
                        ):
                            continue
                        if rid in self._dead or rid in self._failed:
                            self._acks_seen.pop(rid, None)
                        if len(self._acks_seen) <= _STATE_KEEP:
                            break
            # Marks whose slot is already gone (TTL, abort) would otherwise
            # accumulate for the life of the process.
            if len(self._drain_unknown) > _STATE_KEEP:
                for rid in list(self._drain_unknown):
                    if rid not in self._slots:
                        self._drain_unknown.discard(rid)
                    if len(self._drain_unknown) <= _STATE_KEEP:
                        break
            if len(self._reuse_unsafe) > _STATE_KEEP:
                for rid in list(self._reuse_unsafe):
                    if rid not in self._slots:
                        self._reuse_unsafe.discard(rid)
                    if len(self._reuse_unsafe) <= _STATE_KEEP:
                        break
            # One Event per rid replaces the old 2ms poll. ``_requests`` only
            # tracks slot-assignment waiters, so "not in _requests" does not
            # mean nobody waits for DATA_READY. Only drop events whose rid is
            # already terminal: those were signalled (Event.set persists, a
            # blocked thread still wakes) and future waiters return from the
            # state check without the event.
            if len(self._wait_events) > _STATE_KEEP:
                for rid in list(self._wait_events):
                    if (
                        rid in self._data_ready
                        or rid in self._failed
                        or rid in self._dead
                    ):
                        self._wait_events.pop(rid, None)
                    if len(self._wait_events) <= _STATE_KEEP:
                        break

    def _record_peer_ack(self, rank: int, acks, unsafe) -> None:
        """Fold one rank's tick payload into the owner's ack bookkeeping.

        ``acks`` carries ``(rid, safe_reuse)`` pairs: a rank that could not
        prove its release is safe (abort, GC, receive failure) makes the whole
        group's range go through the reuse quarantine. ``unsafe`` lists rids
        whose /send died without drain proof; those ranges are retired.
        """
        with self._lock:
            for rid, safe_reuse in acks or ():
                self._acks_seen.setdefault(rid, set()).add(rank)
                if not safe_reuse:
                    self._reuse_unsafe.add(rid)
            for rid in unsafe or ():
                self._drain_unknown.add(rid)

    def _adopt(self, rid: str, assignment: Tuple[int, int]) -> None:
        with self._lock:
            self._assigned[rid] = assignment
            req = self._requests.get(rid)
            if req is not None:
                req.resolve(*assignment)

    def _owner_allocate(self, reqs: List[Tuple[str, int]]) -> None:
        max_inflight = envs.SGLANG_EMBEDDING_POOL_MAX_INFLIGHT.get()
        now = time.monotonic()
        # Big first: fewer holes when 6MB and 318MB requests interleave.
        for rid, nbytes in sorted(reqs, key=lambda item: (-item[1], item[0])):
            with self._lock:
                if (
                    rid in self._slots
                    or rid in self._dead
                    or rid in self._publish
                    or rid in self._assigned
                ):
                    continue
                if max_inflight and len(self._slots) >= max_inflight:
                    self._pending_since.setdefault(rid, now)
                    self._expire_pending_locked(rid, now)
                    continue
                offset = self._try_alloc_locked(nbytes)
                if offset is None:
                    if rid not in self._pending_since:
                        self._pending_since[rid] = now
                        self._alloc_waits += 1
                        logger.warning(
                            "SharedEmbeddingArena full: %dMB in %d slots; "
                            "queueing %dMB (rid=%s)",
                            self._total_inflight // (1024 * 1024),
                            len(self._slots),
                            nbytes // (1024 * 1024),
                            rid,
                        )
                    self._expire_pending_locked(rid, now)
                    continue
                self._pending_since.pop(rid, None)
                slot = _Slot(offset=offset, nbytes=nbytes, rid=rid, allocated_at=now)
                self._slots[rid] = slot
                self._assigned[rid] = (offset, nbytes)
                self._publish[rid] = (offset, nbytes)
                self._acks_seen.setdefault(rid, set())
                self._total_inflight += _align_up(nbytes, _ALIGN)
                if self._total_inflight > self._peak_inflight:
                    bucket = self._total_inflight // (512 * 1024 * 1024)
                    if bucket > self._peak_inflight // (512 * 1024 * 1024):
                        logger.info(
                            "SharedEmbeddingArena high-watermark: %dMB of %dMB "
                            "(%d slots)",
                            self._total_inflight // (1024 * 1024),
                            self.size_bytes // (1024 * 1024),
                            len(self._slots),
                        )
                    self._peak_inflight = self._total_inflight
            self._resolve_local_request(rid, nbytes)

    def _expire_pending_locked(self, rid: str, now: float) -> None:
        started = self._pending_since.get(rid)
        if started is None:
            self._pending_since[rid] = now
            return
        timeout = envs.SGLANG_EMBEDDING_POOL_ALLOC_TIMEOUT_S.get()
        if now - started > timeout:
            self._pending_since.pop(rid, None)
            reason = (
                f"shared embedding arena full for {timeout:.0f}s "
                f"(inflight={len(self._slots)}, cap={self.size_bytes // (1024 * 1024)}MB)"
            )
            self._mark_failed_locked(rid, reason)
            self._alloc_timeouts += 1
            req = self._requests.get(rid)
            if req is not None:
                req.fail(reason)

    def _resolve_local_request(self, rid: str, nbytes: int) -> None:
        with self._lock:
            req = self._requests.get(rid)
        if req is not None and not req.event.is_set():
            req.resolve(self._assigned[rid][0], nbytes)

    def _fail_dead_requests(self) -> None:
        with self._lock:
            failed = dict(self._failed)
        for rid, reason in failed.items():
            req = self._requests.get(rid)
            if req is not None and not req.event.is_set():
                req.fail(reason)

    def _owner_reap(self) -> None:
        """Free slots every rank acknowledged, and force-free stale ones."""
        ttl = envs.SGLANG_EMBEDDING_POOL_SLOT_TTL_S.get()
        abandon_unsafe = envs.SGLANG_EMBEDDING_POOL_ABANDON_ON_SEND_FAILURE.get()
        now = time.monotonic()
        with self._lock:
            for rid, slot in list(self._slots.items()):
                acked = self._acks_seen.get(rid, set())
                if len(acked) >= self.world_size:
                    if abandon_unsafe and rid in self._drain_unknown:
                        # Every rank acked, but a /send died without proof the
                        # encoder stopped writing: the ack only covers local
                        # reads. A late RDMA write would overwrite whichever
                        # request reuses this range, so retire it for good.
                        self._force_frees += 1
                        self._abandoned_bytes += _align_up(slot.nbytes, _ALIGN)
                        logger.error(
                            "SharedEmbeddingArena: abandoning rid=%s (%dMB): "
                            "a /send failed without drain proof; %dMB lost "
                            "until restart",
                            rid,
                            slot.nbytes // (1024 * 1024),
                            self._abandoned_bytes // (1024 * 1024),
                        )
                        self._free_slot_locked(slot, reusable=False)
                        self._dead.add(rid)
                        self._dead_order.append(rid)
                    elif rid in self._reuse_unsafe:
                        # Some rank released without the safe-reuse proof
                        # (abort, GC, receive failure): keep the short
                        # quarantine so a copy still in flight cannot be
                        # overwritten.
                        self._free_slot_locked(slot)
                    else:
                        # Normal completion: the write was proven by
                        # DATA_READY and every reader is fenced by its release
                        # event, so the range is free immediately.
                        self._free_slot_locked(slot, quarantine=False)
                        self._immediate_frees += 1
                    self._acks_seen.pop(rid, None)
                    self._drain_unknown.discard(rid)
                    self._reuse_unsafe.discard(rid)
                elif ttl > 0 and now - slot.allocated_at > ttl:
                    # A rank that never acked is a bug, not a crash (a crash
                    # takes the whole TP group down). The missing rank may
                    # still be blocked inside /send, so its RDMA write could
                    # land at any time: abandon the range for good instead of
                    # handing it to the next request.
                    self._force_frees += 1
                    self._abandoned_bytes += _align_up(slot.nbytes, _ALIGN)
                    logger.error(
                        "SharedEmbeddingArena: abandoning rid=%s after %.0fs "
                        "(acks from %s of %d ranks); %dMB lost until restart",
                        rid,
                        now - slot.allocated_at,
                        sorted(acked),
                        self.world_size,
                        self._abandoned_bytes // (1024 * 1024),
                    )
                    self._free_slot_locked(slot, reusable=False)
                    self._acks_seen.pop(rid, None)
                    self._drain_unknown.discard(rid)
                    self._dead.add(rid)
                    self._dead_order.append(rid)

    def _free_slot_locked(
        self, slot: _Slot, reusable: bool = True, quarantine: bool = True
    ) -> None:
        self._slots.pop(slot.rid, None)
        self._data_ready.discard(slot.rid)
        self._assigned.pop(slot.rid, None)
        self._total_inflight -= _align_up(slot.nbytes, _ALIGN)
        aligned = _align_up(slot.nbytes, _ALIGN)
        if reusable:
            delay = envs.SGLANG_EMBEDDING_POOL_REUSE_DELAY_S.get() if quarantine else 0
            if delay > 0:
                # A failed /send or a late ack can leave the NIC still writing
                # into this range; quarantine keeps the next request out of it.
                self._quarantine.append(
                    (time.monotonic() + delay, slot.offset, aligned)
                )
            else:
                self._insert_free_locked(slot.offset, aligned)
        self._freed_pending.append(slot.rid)

    # ------------------------------------------------------------------ #
    # free list (owner only)
    # ------------------------------------------------------------------ #
    def _try_alloc_locked(self, nbytes: int) -> Optional[int]:
        self._reap_quarantine_locked()
        aligned = _align_up(nbytes, _ALIGN)
        for index, (offset, length) in enumerate(self._free):
            if length >= aligned:
                if length == aligned:
                    self._free.pop(index)
                else:
                    self._free[index] = (offset + aligned, length - aligned)
                return offset
        return None

    def _reap_quarantine_locked(self) -> None:
        if not self._quarantine:
            return
        now = time.monotonic()
        still: List[Tuple[float, int, int]] = []
        for ready_at, offset, length in self._quarantine:
            if now >= ready_at:
                self._insert_free_locked(offset, length)
            else:
                still.append((ready_at, offset, length))
        self._quarantine = still

    def _insert_free_locked(self, offset: int, length: int) -> None:
        self._free.append((offset, length))
        self._free.sort()
        merged: List[Tuple[int, int]] = []
        for seg_offset, seg_length in self._free:
            if merged and merged[-1][0] + merged[-1][1] == seg_offset:
                merged[-1] = (merged[-1][0], merged[-1][1] + seg_length)
            else:
                merged.append((seg_offset, seg_length))
        self._free = merged

    # ------------------------------------------------------------------ #
    # ack bookkeeping
    # ------------------------------------------------------------------ #
    def _drain_acks(self) -> None:
        with self._lock:
            if self._ack_waiters:
                still: List[Tuple[str, torch.cuda.Event, bool]] = []
                for rid, event, safe_reuse in self._ack_waiters:
                    if event.query():
                        self._acks_ready.append((rid, safe_reuse))
                    else:
                        still.append((rid, event, safe_reuse))
                self._ack_waiters = still

    def _maybe_log_stats(self) -> None:
        interval = envs.SGLANG_EMBEDDING_POOL_STATS_INTERVAL_S.get()
        if interval <= 0:
            return
        now = time.monotonic()
        if now - self._last_stats < interval:
            return
        self._last_stats = now
        with self._lock:
            logger.info(
                "SharedEmbeddingArena stats: inflight=%dMB/%dMB slots=%d "
                "free_segs=%d reqs=%d ack_waiters=%d peak=%dMB waits=%d "
                "timeouts=%d force_frees=%d abandoned=%dMB "
                "quarantine=%dMB immediate_frees=%d owner=%s",
                self._total_inflight // (1024 * 1024),
                self.size_bytes // (1024 * 1024),
                len(self._slots),
                len(self._free),
                len(self._requests),
                len(self._ack_waiters),
                self._peak_inflight // (1024 * 1024),
                self._alloc_waits,
                self._alloc_timeouts,
                self._force_frees,
                self._abandoned_bytes // (1024 * 1024),
                sum(length for _, _, length in self._quarantine) // (1024 * 1024),
                self._immediate_frees,
                self.is_owner,
            )

    # ------------------------------------------------------------------ #
    # misc
    # ------------------------------------------------------------------ #
    @property
    def on_host(self) -> bool:
        return True

    @property
    def host_async(self) -> bool:
        """Arena views are pinned, even though ``Tensor.is_pinned()`` says no."""
        return self._host_async


def _align_up(value: int, align: int) -> int:
    return (value + align - 1) & ~(align - 1)
