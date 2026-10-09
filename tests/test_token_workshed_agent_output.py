# SPDX-License-Identifier: Apache-2.0
"""Regressions for agent-mode output normalization in token-workshed UI."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

import vllm_mlx.css_svg_ui as css_svg_ui
from vllm_mlx.css_svg_ui import _extract_openclaw_text_and_meta


def test_openclaw_extracts_message_text_from_tool_envelope_json() -> None:
    payload = {
        "result": {
            "payloads": [
                {
                    "text": (
                        '{"name":"message","parameters":{"action":"send",'
                        '"channel":"token-workshed-state","target":"user",'
                        '"message":"I am an OpenClaw personal assistant."}}'
                    )
                }
            ],
            "meta": {},
        }
    }

    text, _, _ = _extract_openclaw_text_and_meta(payload)
    assert text == "I am an OpenClaw personal assistant."


def test_openclaw_keeps_plain_json_when_not_message_envelope() -> None:
    payload = {
        "result": {
            "payloads": [{"text": '{"foo":"bar","answer":"ok"}'}],
            "meta": {},
        }
    }

    text, _, _ = _extract_openclaw_text_and_meta(payload)
    assert text == '{"foo":"bar","answer":"ok"}'


def test_openclaw_timeout_response_does_not_retry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    popen_calls = 0
    timeout_payload = {
        "result": {
            "payloads": [
                {"text": "Request timed out before a response was generated."}
            ],
            "meta": {"aborted": True, "stopReason": "timeout"},
        }
    }

    class FakeProcess:
        returncode = 0

        def poll(self) -> int:
            return 0

        def communicate(self, timeout: float | None = None) -> tuple[str, str]:
            return json.dumps(timeout_payload), ""

    def fake_popen(*args: object, **kwargs: object) -> FakeProcess:
        nonlocal popen_calls
        popen_calls += 1
        return FakeProcess()

    monkeypatch.setattr(css_svg_ui, "OPENCLAW_DIR", tmp_path)
    monkeypatch.setattr(
        css_svg_ui,
        "_resolve_openclaw_runtime_target",
        lambda **kwargs: {
            "backend": "ollama",
            "provider_id": "ollama",
            "provider_api": "ollama",
            "model_id": "model",
            "provider_model": "ollama/model",
            "base_url": "http://127.0.0.1:8000",
            "runtime_key": "runtime",
        },
    )
    monkeypatch.setattr(
        css_svg_ui,
        "_ensure_openclaw_runtime_config",
        lambda **kwargs: "cfg",
    )
    monkeypatch.setattr(
        css_svg_ui,
        "_openclaw_runtime_cache_key",
        lambda **kwargs: "runtime|cfg",
    )
    monkeypatch.setattr(css_svg_ui, "_is_openclaw_runtime_warm", lambda key: False)
    monkeypatch.setattr(css_svg_ui, "_ensure_openclaw_gateway_process", lambda **kwargs: 19123)
    monkeypatch.setattr(css_svg_ui, "_resolve_openclaw_timeout_seconds", lambda: 120)
    monkeypatch.setattr(
        css_svg_ui,
        "_resolve_openclaw_launcher",
        lambda: (["node", "entry.js"], "fake"),
    )
    monkeypatch.setattr(
        css_svg_ui,
        "_resolve_openclaw_agent_capabilities",
        lambda **kwargs: frozenset(),
    )
    monkeypatch.setattr(css_svg_ui, "_build_openclaw_subprocess_env", lambda: {})
    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    with pytest.raises(RuntimeError, match="timed out before a response"):
        css_svg_ui._run_openclaw_turn(
            server_url="http://127.0.0.1:8000",
            selected_model="model",
            system_prompt="",
            user_message="hello",
            user_content=None,
            conversation_id="diag",
        )

    assert popen_calls == 1
