"""The renamed send HTTP timeout still honours its deprecated name."""

import os
import warnings

import pytest

from sglang.srt import environ as envmod


def test_send_http_timeout_accepts_deprecated_name(monkeypatch):
    new = "SGLANG_ENCODER_SEND_HTTP_TIMEOUT_S"
    old = "SGLANG_ENCODER_SEND_TIMEOUT_S"
    monkeypatch.delenv(new, raising=False)
    monkeypatch.setenv(old, "12.5")
    try:
        with pytest.warns(DeprecationWarning):
            assert envmod.envs.SGLANG_ENCODER_SEND_HTTP_TIMEOUT_S.get() == 12.5
    finally:
        os.environ.pop(new, None)


def test_send_http_timeout_new_name_wins(monkeypatch):
    new = "SGLANG_ENCODER_SEND_HTTP_TIMEOUT_S"
    old = "SGLANG_ENCODER_SEND_TIMEOUT_S"
    monkeypatch.setenv(new, "7.5")
    monkeypatch.setenv(old, "12.5")
    try:
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            assert envmod.envs.SGLANG_ENCODER_SEND_HTTP_TIMEOUT_S.get() == 7.5
        assert not [w for w in record if issubclass(w.category, DeprecationWarning)]
    finally:
        os.environ.pop(new, None)
