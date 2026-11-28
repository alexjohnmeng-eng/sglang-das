"""Tests for encoder -> prefill bootstrap registration helpers.

Covers the heartbeat re-registration loop, backoff/recovery, runtime target
updates, the instance_id/token contract, and the hot-reloaded targets file.
All timings are tiny (0.02-0.2s) so the whole module runs in a few seconds.
"""

import json
import logging
import os
import threading
import time

from sglang.srt.disaggregation.encoder.bootstrap_heartbeat import (
    BootstrapHeartbeat,
)
from sglang.srt.disaggregation.encoder.bootstrap_targets import BootstrapTargets

HEARTBEAT = 0.03


def _wait_until(predicate, timeout=3.0, interval=0.005):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def test_heartbeat_periodically_resends():
    calls = []
    lock = threading.Lock()

    def post(url, payload, timeout):
        with lock:
            calls.append((url, payload.get("url"), time.monotonic()))
        return True

    heartbeat = BootstrapHeartbeat(
        "http://encoder:8000",
        ["http://prefill:30000"],
        post_fn=post,
        heartbeat_s=HEARTBEAT,
        max_backoff_s=0.2,
    )
    heartbeat.start()
    try:
        assert _wait_until(lambda: len(calls) >= 4)
    finally:
        heartbeat.stop(timeout=1.0)

    assert len(calls) >= 4
    assert all(url == "http://prefill:30000" for url, _, _ in calls)
    assert all(enc == "http://encoder:8000" for _, enc, _ in calls)
    # Consecutive posts are ~one heartbeat apart, clearly more than one shot.
    gap = calls[-1][2] - calls[-2][2]
    assert gap >= HEARTBEAT * 0.5


def test_payload_contains_stable_instance_id():
    payloads = []
    lock = threading.Lock()

    def post(url, payload, timeout):
        with lock:
            payloads.append(dict(payload))
        return True

    heartbeat = BootstrapHeartbeat(
        "http://encoder:8000",
        ["http://prefill:30000"],
        instance_id="deadbeef" * 4,
        post_fn=post,
        heartbeat_s=HEARTBEAT,
    )
    heartbeat.start()
    try:
        assert _wait_until(lambda: len(payloads) >= 2)
    finally:
        heartbeat.stop(timeout=1.0)

    expected = {"url": "http://encoder:8000", "instance_id": "deadbeef" * 4}
    assert all(payload == expected for payload in payloads)


def test_failure_backs_off_then_recovers():
    attempts = []
    lock = threading.Lock()

    def post(url, payload, timeout):
        with lock:
            attempts.append(time.monotonic())
            return len(attempts) >= 4  # first three fail, then recover
        # unreachable

    heartbeat = BootstrapHeartbeat(
        "http://encoder:8000",
        ["http://prefill:30000"],
        post_fn=post,
        heartbeat_s=0.02,
        max_backoff_s=0.2,
    )
    heartbeat.start()
    try:
        assert _wait_until(lambda: len(attempts) >= 5, timeout=3.0)
    finally:
        heartbeat.stop(timeout=1.0)

    gaps = [later - earlier for earlier, later in zip(attempts, attempts[1:])]
    # Delays grow: 2x, 4x, 8x the heartbeat before the success resets it.
    assert gaps[0] >= 0.02
    # Bounded, tolerance-friendly checks: OS jitter can inflate gaps[0], so
    # assert growth only against the configured backoff bounds instead of a
    # strict ratio between individual samples.
    assert gaps[2] >= 0.15  # capped backoff neighbourhood
    assert gaps[2] >= gaps[0] * 0.8
    # Recovery then resumes the fixed cadence instead of the capped backoff.
    assert gaps[-1] < gaps[2]


def test_stop_halts_further_calls():
    counter = {"n": 0}
    lock = threading.Lock()

    def post(url, payload, timeout):
        with lock:
            counter["n"] += 1
        return True

    heartbeat = BootstrapHeartbeat(
        "http://encoder:8000",
        ["http://prefill:30000"],
        post_fn=post,
        heartbeat_s=HEARTBEAT,
    )
    heartbeat.start()
    assert _wait_until(lambda: counter["n"] >= 2)
    heartbeat.stop(timeout=1.0)
    stopped_at = counter["n"]
    time.sleep(HEARTBEAT * 4)
    assert counter["n"] == stopped_at


def test_update_targets_adds_and_removes():
    counts = {}
    lock = threading.Lock()

    def post(url, payload, timeout):
        with lock:
            counts[url] = counts.get(url, 0) + 1
        return True

    heartbeat = BootstrapHeartbeat(
        "http://encoder:8000",
        ["http://prefill-a"],
        post_fn=post,
        heartbeat_s=HEARTBEAT,
    )
    heartbeat.start()
    try:
        assert _wait_until(lambda: counts.get("http://prefill-a", 0) >= 2)

        # Duplicates in the update are deduped; existing targets survive.
        heartbeat.update_targets(
            ["http://prefill-b", "http://prefill-b", "http://prefill-a"]
        )
        assert _wait_until(lambda: counts.get("http://prefill-b", 0) >= 2)

        heartbeat.update_targets(["http://prefill-b"])
        frozen = counts.get("http://prefill-a", 0)
        time.sleep(HEARTBEAT * 4)
        assert counts.get("http://prefill-a", 0) == frozen
        assert counts.get("http://prefill-b", 0) > 2
    finally:
        heartbeat.stop(timeout=1.0)


def test_token_sent_as_bearer_header(monkeypatch):
    captured = {}

    class _Resp:
        status_code = 200
        text = ""

    import requests

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        return _Resp()

    monkeypatch.setattr(requests, "post", fake_post)

    heartbeat = BootstrapHeartbeat(
        "http://encoder:8000",
        ["http://prefill:30000"],
        instance_id="iid",
        token="s3cret",
        heartbeat_s=HEARTBEAT,
    )
    heartbeat.start()
    try:
        assert _wait_until(lambda: "headers" in captured)
    finally:
        heartbeat.stop(timeout=1.0)

    assert captured["url"] == "http://prefill:30000/register_encoder_url"
    assert captured["headers"] == {"Authorization": "Bearer s3cret"}
    assert captured["json"]["instance_id"] == "iid"


def test_no_token_omits_auth_header(monkeypatch):
    captured = {}

    class _Resp:
        status_code = 200
        text = ""

    import requests

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["headers"] = headers
        return _Resp()

    monkeypatch.setattr(requests, "post", fake_post)

    heartbeat = BootstrapHeartbeat(
        "http://encoder:8000", ["http://prefill:30000"], heartbeat_s=HEARTBEAT
    )
    heartbeat.start()
    try:
        assert _wait_until(lambda: "headers" in captured)
    finally:
        heartbeat.stop(timeout=1.0)

    assert captured["headers"] is None


def test_targets_file_hot_reload(tmp_path):
    path = tmp_path / "targets.json"
    base_ns = time.time_ns()
    path.write_text(json.dumps(["http://prefill-1"]))
    os.utime(path, ns=(base_ns, base_ns))

    seen = []
    targets = BootstrapTargets(
        ["http://static"], file_path=str(path), poll_s=0.05
    )
    # First snapshot loads the file synchronously (static first).
    assert targets.snapshot() == ["http://static", "http://prefill-1"]

    targets.start(lambda urls: seen.append(urls))
    try:
        # No change yet: the watcher must not emit a spurious callback.
        time.sleep(0.15)
        assert seen == []

        next_ns = base_ns + 1_000_000_000
        path.write_text(
            json.dumps({"bootstrap_urls": ["http://prefill-2", "http://static"]})
        )
        os.utime(path, ns=(next_ns, next_ns))
        assert _wait_until(
            lambda: seen and seen[-1] == ["http://static", "http://prefill-2"],
            timeout=2.0,
        )
        assert targets.snapshot() == ["http://static", "http://prefill-2"]
    finally:
        targets.stop()


def test_targets_file_invalid_content_keeps_previous(tmp_path, caplog):
    path = tmp_path / "targets.json"
    base_ns = time.time_ns()
    path.write_text(json.dumps(["http://prefill-1"]))
    os.utime(path, ns=(base_ns, base_ns))

    targets = BootstrapTargets([], file_path=str(path), poll_s=0.05)
    assert targets.snapshot() == ["http://prefill-1"]

    targets.start(lambda urls: None)
    try:
        next_ns = base_ns + 1_000_000_000
        with caplog.at_level(logging.WARNING):
            path.write_text("{not valid json")
            os.utime(path, ns=(next_ns, next_ns))
            assert _wait_until(
                lambda: any(
                    "Ignoring encoder bootstrap targets file" in rec.message
                    for rec in caplog.records
                ),
                timeout=2.0,
            )
        # Bad content is dropped; the last good list survives.
        assert targets.snapshot() == ["http://prefill-1"]
    finally:
        targets.stop()


def test_targets_file_absent_falls_back_to_static(tmp_path):
    path = tmp_path / "missing.json"
    targets = BootstrapTargets(["http://static"], file_path=str(path), poll_s=0.05)
    assert targets.snapshot() == ["http://static"]
    targets.start(lambda urls: None)
    targets.stop()
    assert targets.snapshot() == ["http://static"]


class _JsonResp:
    def __init__(self, body=None, status=200, text=""):
        self.status_code = status
        self.text = text
        self._body = body

    def json(self):
        if self._body is None:
            raise ValueError("not json")
        return self._body


def test_policy_rejection_is_a_failure(monkeypatch):
    from sglang.srt.disaggregation.encoder import bootstrap_heartbeat as bh

    import requests

    def fake_post(url, json=None, headers=None, timeout=None):
        return _JsonResp({"accepted": False, "reason": "max encoder URLs reached"})

    monkeypatch.setattr(requests, "post", fake_post)
    assert bh._default_post("http://prefill:30000", {"url": "http://enc"}, 1.0) is False

    monkeypatch.setattr(
        requests, "post",
        lambda *a, **k: _JsonResp({"accepted": True, "reason": ""}),
    )
    assert bh._default_post("http://prefill:30000", {"url": "http://enc"}, 1.0) is True

    # Idempotent repeat is reported as a rejection reason but is a success.
    monkeypatch.setattr(
        requests, "post",
        lambda *a, **k: _JsonResp({"accepted": False, "reason": "already registered"}),
    )
    assert bh._default_post("http://prefill:30000", {"url": "http://enc"}, 1.0) is True

    # Legacy plain-text "OK" (no JSON body) still counts as success.
    monkeypatch.setattr(requests, "post", lambda *a, **k: _JsonResp(None, text="OK"))
    assert bh._default_post("http://prefill:30000", {"url": "http://enc"}, 1.0) is True


def test_trailing_slashes_are_normalized(monkeypatch):
    captured = {}

    import requests

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["url"] = url
        return _JsonResp({"accepted": True})

    monkeypatch.setattr(requests, "post", fake_post)
    heartbeat = BootstrapHeartbeat(
        "http://encoder:8000/",
        ["http://prefill:30000/"],
        heartbeat_s=HEARTBEAT,
    )
    heartbeat.start()
    try:
        assert _wait_until(lambda: "url" in captured)
    finally:
        heartbeat.stop(timeout=1.0)
    assert captured["url"] == "http://prefill:30000/register_encoder_url"


def test_current_targets_tracks_updates():
    heartbeat = BootstrapHeartbeat(
        "http://encoder:8000", ["http://prefill-a"], heartbeat_s=HEARTBEAT
    )
    heartbeat.start()
    try:
        assert heartbeat.current_targets() == ["http://prefill-a"]
        heartbeat.update_targets(["http://prefill-b/"])
        assert heartbeat.current_targets() == ["http://prefill-b"]
    finally:
        heartbeat.stop(timeout=1.0)
