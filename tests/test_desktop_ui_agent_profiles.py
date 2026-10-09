# SPDX-License-Identifier: Apache-2.0
"""Profile parser precedence in the local model server manager."""

from __future__ import annotations

from vllm_mlx.agent_profiles import save_agent_profile
from vllm_mlx.desktop_ui import LocalModelServerManager


def _profile() -> dict[str, object]:
    return {
        "profile_version": 1,
        "model_id": "owner/model",
        "configured_at": 1,
        "family": "qwen3",
        "context_window": 32768,
        "chat_template": None,
        "generation_defaults": {},
        "tool_policy": {
            "always": ["read", "write", "edit", "process"],
            "terminal_on_demand": ["exec"],
            "web_on_demand": ["web_search", "web_fetch", "browser"],
        },
        "openclaw": {
            "ready": True,
            "tool_calling": "passed",
            "tool_call_parser": "qwen",
            "reasoning_parser": "qwen3",
            "timeout_seconds": 240,
            "probe": {},
        },
        "hermes": {
            "ready": True,
            "tool_calling": "passed",
            "tool_call_parser": "qwen",
            "reasoning_parser": "qwen3",
            "timeout_seconds": 240,
            "max_turns": 10,
            "probe": {},
        },
        "warnings": [],
    }


def _flag_value(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def test_profile_parsers_are_used_for_future_server_starts(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("TOKEN_WORKSHED_DESKTOP_STATE_PATH", str(tmp_path / "state.json"))
    monkeypatch.delenv("TOKEN_WORKSHED_ENABLE_AUTO_TOOL_CHOICE", raising=False)
    save_agent_profile(_profile())
    manager = LocalModelServerManager("python", "127.0.0.1", 8000)

    command = manager._build_serve_cmd("owner/model")

    assert _flag_value(command, "--reasoning-parser") == "qwen3"
    assert _flag_value(command, "--tool-call-parser") == "qwen"
    assert manager._effective_reasoning_parser("owner/model") == "qwen3"
    assert manager._effective_tool_call_parser("owner/model") == "qwen"


def test_calibrated_profile_replaces_an_older_adaptive_parser_cache(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("TOKEN_WORKSHED_DESKTOP_STATE_PATH", str(tmp_path / "state.json"))
    monkeypatch.delenv("TOKEN_WORKSHED_ENABLE_AUTO_TOOL_CHOICE", raising=False)
    save_agent_profile(_profile())
    manager = LocalModelServerManager("python", "127.0.0.1", 8000)
    manager._tool_call_parser_overrides["owner/model"] = "mistral"

    command = manager._build_serve_cmd("owner/model")

    assert _flag_value(command, "--tool-call-parser") == "qwen"
