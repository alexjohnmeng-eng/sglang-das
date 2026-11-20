"""Regression tests for the /health probe cleanup path.

Under load the DP-attention health probe drops its in-flight requests. The
unpatched handler cancelled the tasks, popped rid_to_state and returned
without awaiting the probe's gather, which produced, on every probe:

* ``asyncio: _GatheringFuture exception was never retrieved``;
* ``Decode handshake failed ... Fake KVReceiver Exception`` from probes that
  were torn down mid-handshake;
* lingering decode-side preallocations (#prealloc-req climbing), because the
  cancelled task's generic failure path could no longer find the state to
  abort.

Run: pytest -q test/srt/test_health_probe_cleanup.py
"""

import asyncio
import gc

import pytest

from sglang.srt.entrypoints.http_server import _drop_health_requests


class _FakeState:
    pass


class _FakeTokenizerManager:
    def __init__(self, rids):
        self.rid_to_state = {rid: _FakeState() for rid in rids}
        self.aborted = []

    def abort_request(self, rid):
        self.aborted.append(rid)


class _FakeGri:
    def __init__(self, rids):
        self.rid = rids


@pytest.mark.parametrize("use_gather", [True, False])
def test_drop_consumes_futures_and_aborts(use_gather):
    async def scenario():
        loop = asyncio.get_running_loop()
        reported = []
        loop.set_exception_handler(lambda _l, ctx: reported.append(ctx.get("message", "")))
        tm = _FakeTokenizerManager(["h0", "h1"])
        gri = _FakeGri(["h0", "h1"])
        tasks = [asyncio.create_task(asyncio.sleep(30)) for _ in range(2)]
        gather = asyncio.gather(*tasks) if use_gather else None
        await asyncio.sleep(0)  # let the tasks start

        await _drop_health_requests(tm, tasks, [gri], gather)

        assert all(t.cancelled() for t in tasks)
        if gather is not None:
            assert gather.cancelled() or gather.done()
        # Both probe requests were aborted before their state was dropped.
        assert sorted(tm.aborted) == ["h0", "h1"]
        assert tm.rid_to_state == {}
        return reported

    reported = asyncio.run(scenario())
    gc.collect()
    assert not any("never retrieved" in m for m in reported), reported


def test_drop_tolerates_probe_that_already_cleaned_up():
    """A probe whose own cleanup removed its state must not be aborted twice."""

    async def scenario():
        tm = _FakeTokenizerManager([])
        gri = _FakeGri("h2")
        task = asyncio.create_task(asyncio.sleep(30))
        await asyncio.sleep(0)
        await _drop_health_requests(tm, [task], [gri], None)
        assert tm.aborted == []
        assert tm.rid_to_state == {}

    asyncio.run(scenario())


def test_drop_abort_failure_does_not_escape():
    """A scheduler-side abort error must not turn /health into a 500."""

    class _BrokenTM(_FakeTokenizerManager):
        def abort_request(self, rid):
            raise RuntimeError("scheduler gone")

    async def scenario():
        tm = _BrokenTM(["h3"])
        task = asyncio.create_task(asyncio.sleep(30))
        await asyncio.sleep(0)
        await _drop_health_requests(tm, [task], [_FakeGri("h3")], None)
        assert tm.rid_to_state == {}

    asyncio.run(scenario())


def test_unpatched_pattern_leaks_the_cancelled_gather():
    """Shows the failure the fix removes, without a live server.

    The old handler cancelled the probe tasks and returned without awaiting
    the gather; the cancelled _GatheringFuture is then collected with its
    exception never retrieved, which the event loop reports exactly as the
    production log did.
    """

    async def scenario():
        loop = asyncio.get_running_loop()
        reported = []
        loop.set_exception_handler(lambda _l, ctx: reported.append(ctx.get("message", "")))
        tasks = [asyncio.create_task(asyncio.sleep(30)) for _ in range(2)]
        gather = asyncio.gather(*tasks)
        await asyncio.sleep(0)
        for task in tasks:
            task.cancel()
        gather.cancel()
        return reported

    reported = asyncio.run(scenario())
    for _ in range(3):
        gc.collect()
    assert any("never retrieved" in message for message in reported), reported


def test_health_routes_point_at_the_handler():
    """The /health routes must not be captured by a module-level helper.

    Moving code around previously left the route decorators attached to the
    drop helper, so FastAPI answered /health with a 400 listing 'tokenizer_manager'
    and 'tasks' as missing query parameters.
    """
    from sglang.srt.entrypoints import http_server as hs

    routes = {r.path: r.endpoint for r in hs.app.routes if hasattr(r, "path")}
    assert routes["/health"] is hs.health_generate
    assert routes["/health_generate"] is hs.health_generate
    assert hs._drop_health_requests not in set(routes.values())
