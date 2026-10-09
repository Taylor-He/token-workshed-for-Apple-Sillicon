# SPDX-License-Identifier: Apache-2.0
"""Security-focused regressions for token-workshed UI/server helpers."""

from __future__ import annotations

import pytest

from vllm_mlx.css_svg_ui import _normalize_server_url
from vllm_mlx.desktop_ui import _is_loopback_host


def test_normalize_server_url_allows_loopback_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TOKEN_WORKSHED_ALLOWED_SERVER_HOSTS", raising=False)
    assert _normalize_server_url("127.0.0.1:8000") == "http://127.0.0.1:8000"
    assert _normalize_server_url("http://localhost:7862/") == "http://localhost:7862"
    assert _normalize_server_url("http://[::1]:8000/v1/models") == "http://[::1]:8000"


def test_normalize_server_url_rejects_non_allowlisted_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TOKEN_WORKSHED_ALLOWED_SERVER_HOSTS", raising=False)
    with pytest.raises(ValueError):
        _normalize_server_url("https://example.com")


def test_normalize_server_url_accepts_env_allowlist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TOKEN_WORKSHED_ALLOWED_SERVER_HOSTS", "example.com, model.internal")
    assert _normalize_server_url("https://example.com/path?q=1") == "https://example.com"
    assert _normalize_server_url("http://model.internal:9000/v1") == "http://model.internal:9000"


def test_normalize_server_url_rejects_embedded_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TOKEN_WORKSHED_ALLOWED_SERVER_HOSTS", raising=False)
    with pytest.raises(ValueError):
        _normalize_server_url("http://user:pass@localhost:8000")


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("localhost", True),
        ("127.0.0.1", True),
        ("::1", True),
        ("0.0.0.0", False),
        ("example.com", False),
    ],
)
def test_is_loopback_host(host: str, expected: bool) -> None:
    assert _is_loopback_host(host) is expected
