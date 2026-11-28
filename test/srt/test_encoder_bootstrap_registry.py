"""Unit tests for the prefill-side encoder bootstrap registry.

Covers the dynamic-registration contract without starting uvicorn:

* instance-generation tombstones reject a retired encoder's heartbeat;
* a new generation (or a legacy payload with no instance id) is accepted;
* repeating the same generation is idempotent and must not reset the health
  failure count, otherwise a heartbeat could mask an unhealthy encoder;
* the URL allowlist and max-URL cap reject policy violations;
* the health probe falls back to /health on a 404 from a custom path;
* bearer-token validation for the bootstrap routes.

Run: pytest -q test/srt/test_encoder_bootstrap_registry.py
"""

import asyncio

import pytest

from sglang.srt.disaggregation.encoder import receiver as receiver_mod
from sglang.srt.disaggregation.encoder.receiver import (
    EncoderBootstrapServer,
    EncoderRegistry,
    is_authorized,
    parse_allowed_prefixes,
    probe_paths,
    probe_with_fallback,
)

URL = "http://10.0.0.1:8000"


# --------------------------------------------------------------------------- #
# Generation / tombstone semantics                                           #
# --------------------------------------------------------------------------- #
def test_tombstone_rejects_same_generation_heartbeat():
    registry = EncoderRegistry()
    assert registry.register(URL, "abc").accepted is True
    assert registry.unregister(URL) is True

    result = registry.register(URL, "abc")
    assert result.accepted is False
    assert "stale" in result.reason
    assert registry.list_urls() == []


def test_tombstone_accepts_new_generation():
    registry = EncoderRegistry()
    registry.register(URL, "abc")
    registry.unregister(URL)

    result = registry.register(URL, "def")
    assert result.accepted is True
    assert registry.list_urls() == [URL]
    assert registry.instance_id(URL) == "def"
    # The tombstone was superseded by the fresh generation.
    assert registry.tombstone(URL) is None


def test_tombstone_none_or_empty_is_new_generation():
    registry = EncoderRegistry()
    registry.register(URL, "abc")
    registry.unregister(URL)

    # Legacy / empty instance id means "no generation" and may re-register.
    assert registry.register(URL, None).accepted is True
    registry.unregister(URL)
    assert registry.register(URL, "").accepted is True


def test_legacy_payload_without_instance_id():
    registry = EncoderRegistry()
    assert registry.register(URL).accepted is True
    assert registry.instance_id(URL) is None
    # A repeated legacy heartbeat is idempotent (accepted, not re-added).
    assert registry.register(URL).accepted is True
    assert registry.list_urls() == [URL]
    assert registry.list_urls() == [URL]


def test_unregister_unknown_url_is_a_noop():
    # Unknown URLs are not tombstoned: otherwise every rollout and every
    # unauthenticated DELETE with a fresh URL would grow the registry forever.
    registry = EncoderRegistry()
    assert registry.unregister(URL) is False
    assert registry.tombstone(URL) is None
    assert registry.register(URL, "abc").accepted is True


def test_upgrade_from_legacy_to_instance_id_is_new_generation():
    registry = EncoderRegistry()
    registry.register(URL)
    assert registry.register(URL, "abc").accepted is True
    assert registry.instance_id(URL) == "abc"


# --------------------------------------------------------------------------- #
# Health state preservation                                                  #
# --------------------------------------------------------------------------- #
def test_repeat_same_generation_keeps_fail_count():
    registry = EncoderRegistry()
    registry.register(URL, "abc")
    registry.apply_health_results([(URL, False)], now=0.0, fail_threshold=99)
    assert registry.fail_count(URL) == 1

    result = registry.register(URL, "abc")
    assert result.accepted is True  # idempotent repeat: it is registered
    assert result.reason == "already registered"
    # A heartbeat must not hide an unhealthy encoder.
    assert registry.fail_count(URL) == 1


def test_new_generation_clears_fail_count():
    registry = EncoderRegistry()
    registry.register(URL, "abc")
    registry.apply_health_results([(URL, False)], now=0.0, fail_threshold=99)
    assert registry.fail_count(URL) == 1

    assert registry.register(URL, "new").accepted is True
    assert registry.fail_count(URL) == 0


def test_revive_from_evicted_clears_fail_count():
    registry = EncoderRegistry()
    registry.register(URL, "abc")
    registry.apply_health_results([(URL, False)], now=0.0, fail_threshold=1)
    assert registry.is_evicted(URL) is True
    assert registry.list_urls() == []

    # Same-generation heartbeat must not override health eviction.
    assert registry.register(URL, "abc").accepted is False
    assert registry.is_evicted(URL) is True
    # A new instance generation revives and clears the probe state.
    assert registry.register(URL, "new").accepted is True
    assert registry.is_evicted(URL) is False
    assert registry.fail_count(URL) == 0


# --------------------------------------------------------------------------- #
# Allowlist / max URL cap                                                    #
# --------------------------------------------------------------------------- #
def test_allowlist_rejects_unlisted_prefix():
    registry = EncoderRegistry(allowed_url_prefixes=["http://10.0.0.1:8000", "http://10.0.0.2:8000"])
    result = registry.register("http://evil.example:8000", "abc")
    assert result.accepted is False
    assert "allowlist" in result.reason
    assert registry.list_urls() == []


def test_allowlist_accepts_matching_prefix():
    registry = EncoderRegistry(allowed_url_prefixes=["http://10.0.0.1:8000", "http://10.0.0.2:8000"])
    assert registry.register(URL, "abc").accepted is True


def test_max_urls_rejects_new_url_but_not_existing():
    registry = EncoderRegistry(max_urls=1)
    assert registry.register(URL, "abc").accepted is True

    result = registry.register("http://10.0.0.2:8000", "abc")
    assert result.accepted is False
    assert "max" in result.reason

    # Existing URL: idempotent repeat and a new generation both bypass the cap.
    assert registry.register(URL, "abc").accepted is True
    assert registry.register(URL, "def").accepted is True
    assert registry.list_urls() == [URL]


def test_parse_allowed_prefixes():
    assert parse_allowed_prefixes("") == []
    assert parse_allowed_prefixes(None) == []
    assert parse_allowed_prefixes(" http://a , http://b ,") == [
        "http://a",
        "http://b",
    ]


def test_shared_url_list_mutated_in_place():
    shared = [URL]
    registry = EncoderRegistry(urls=shared)
    registry.unregister(URL)
    assert shared == []
    registry.register(URL, "new")
    assert shared == [URL]


# --------------------------------------------------------------------------- #
# Probe path fallback                                                        #
# --------------------------------------------------------------------------- #
def test_probe_paths():
    assert probe_paths(URL, "/live") == [f"{URL}/live", f"{URL}/health"]
    assert probe_paths(URL, "live") == [f"{URL}/live", f"{URL}/health"]
    assert probe_paths(URL, "/health") == [f"{URL}/health"]
    assert probe_paths(URL, "") == [f"{URL}/health"]
    assert probe_paths(URL + "/", "/live") == [
        f"{URL}/live",
        f"{URL}/health",
    ]


class _FakeResponse:
    def __init__(self, status):
        self.status = status

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False


class _FakeSession:
    """Serve the given statuses in order and record requested URLs."""

    def __init__(self, statuses):
        self._statuses = list(statuses)
        self.requested = []

    def get(self, url):
        self.requested.append(url)
        return _FakeResponse(self._statuses.pop(0))


def test_probe_falls_back_to_health_on_404():
    session = _FakeSession([404, 200])
    assert asyncio.run(probe_with_fallback(session, URL, "/live")) is True
    assert session.requested == [f"{URL}/live", f"{URL}/health"]


def test_probe_does_not_fall_back_on_other_status():
    session = _FakeSession([503])
    assert asyncio.run(probe_with_fallback(session, URL, "/live")) is False
    assert session.requested == [f"{URL}/live"]


def test_probe_default_path_404_has_no_duplicate_request():
    session = _FakeSession([404])
    assert asyncio.run(probe_with_fallback(session, URL, "/health")) is False
    assert session.requested == [f"{URL}/health"]


def test_probe_fallback_invokes_callback_per_fallback():
    calls = []
    session = _FakeSession([404, 404, 404, 200])
    assert (
        asyncio.run(
            probe_with_fallback(session, URL, "/live", on_fallback=calls.append)
        )
        is False
    )
    assert calls == [URL]
    assert (
        asyncio.run(
            probe_with_fallback(session, URL, "/live", on_fallback=calls.append)
        )
        is True
    )
    assert calls == [URL, URL]


def test_server_warns_fallback_only_once(monkeypatch):
    # Build the server object without __init__ so uvicorn is never started.
    server = EncoderBootstrapServer.__new__(EncoderBootstrapServer)
    server._probe_fallback_warned = set()
    warnings = []
    monkeypatch.setattr(
        receiver_mod.logger,
        "warning",
        lambda *args, **kwargs: warnings.append(args),
    )

    server._warn_probe_fallback(URL)
    server._warn_probe_fallback(URL)

    assert len(warnings) == 1


# --------------------------------------------------------------------------- #
# Bearer token                                                               #
# --------------------------------------------------------------------------- #
def test_is_authorized():
    assert is_authorized(None, "") is True  # auth disabled
    assert is_authorized("Bearer secret", "secret") is True
    assert is_authorized("Bearer wrong", "secret") is False
    assert is_authorized(None, "secret") is False
    assert is_authorized("bearer secret", "secret") is False
    assert is_authorized("", "secret") is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


def test_evicted_encoder_is_not_revived_by_same_generation_heartbeat():
    registry = EncoderRegistry()
    registry.register(URL, "abc")
    registry.apply_health_results([(URL, False)] * 3, now=0.0, fail_threshold=3)
    assert URL not in registry.list_urls()
    assert registry.is_evicted(URL)

    # The encoder process is alive but unhealthy: its heartbeat must not
    # override health eviction.
    assert registry.register(URL, "abc").accepted is False
    assert URL not in registry.list_urls()
    assert registry.is_evicted(URL)

    # A new instance generation is a different process and may register; the
    # probe loop still decides whether it stays.
    assert registry.register(URL, "new").accepted is True
    assert URL in registry.list_urls()
    assert not registry.is_evicted(URL)


def test_allowlist_uses_origin_boundaries():
    registry = EncoderRegistry(allowed_url_prefixes=["http://10.0.0.1:8000"])
    # Suffix trick that a raw startswith check would have accepted.
    assert registry.register("http://10.0.0.1.attacker.example:8000/", "x").accepted is False
    assert registry.register("http://10.0.0.1:8000", "x").accepted is True


def test_tombstones_are_bounded():
    registry = EncoderRegistry()
    for index in range(1200):
        url = f"http://10.9.0.{index}:8000"
        registry.register(url, "g")
        registry.unregister(url)
    assert len(registry._tombstones) <= registry._max_tombstones
    assert len(registry._tombstone_order) <= registry._max_tombstones


def test_wait_bound_raises_when_port_is_taken():
    import socket

    from sglang.srt.disaggregation.encoder.receiver import EncoderBootstrapServer

    blocker = socket.socket()
    blocker.bind(("127.0.0.1", 0))
    blocker.listen(1)
    port = blocker.getsockname()[1]
    server = EncoderBootstrapServer(
        host="127.0.0.1", port=port, urls=[], health_check_interval=0
    )
    try:
        with pytest.raises(RuntimeError):
            server.wait_bound(timeout=1.5)
    finally:
        server.close()
        blocker.close()
