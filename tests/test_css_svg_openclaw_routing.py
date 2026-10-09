# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from typing import Any

from fastapi.testclient import TestClient

from vllm_mlx import css_svg_ui


def _make_openclaw_app(
    monkeypatch,
    *,
    reply_text: str = "Final Answer: routed",
) -> tuple[TestClient, list[dict[str, Any]]]:
    captured_calls: list[dict[str, Any]] = []

    def _fake_run_openclaw_turn(**kwargs: Any) -> dict[str, Any]:
        captured_calls.append(kwargs)
        return {
            "reply": reply_text,
            "metrics": {},
            "session_id": "session-test",
            "stop_reason": "stop",
            "model": str(kwargs.get("selected_model") or "default"),
        }

    monkeypatch.setattr(css_svg_ui, "_run_openclaw_turn", _fake_run_openclaw_turn)
    monkeypatch.setattr(
        css_svg_ui, "_should_run_agent_probe_on_chat", lambda _force: False
    )
    monkeypatch.setattr(
        css_svg_ui,
        "_configured_agent_profile",
        lambda _model: {
            "profile_version": 1,
            "model_id": "test-model",
            "context_window": 32768,
            "tool_policy": {
                "always": ["read", "write", "edit", "process"],
                "terminal_on_demand": ["exec"],
                "web_on_demand": ["web_search", "web_fetch", "browser"],
            },
            "openclaw": {"ready": True, "timeout_seconds": 240},
            "hermes": {"ready": True, "timeout_seconds": 240, "max_turns": 10},
        },
    )
    app = css_svg_ui.create_app("http://127.0.0.1:8000", 1024, 0.7)
    return TestClient(app), captured_calls


def _post_stream_payload(
    client: TestClient,
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    response = client.post("/api/chat/stream", json=payload)
    assert response.status_code == 200
    events = [json.loads(line) for line in response.text.splitlines() if line.strip()]
    assert any(event.get("type") == "done" for event in events)
    return events


def test_chat_stream_uses_profile_base_tools_for_normal_chat(monkeypatch) -> None:
    client, captured = _make_openclaw_app(monkeypatch)
    try:
        _post_stream_payload(
            client,
            {
                "message": "Explain what vector databases are in one paragraph.",
                "openclaw_enabled": True,
                "agent_runtime": "openclaw",
                "server_url": "http://127.0.0.1:8000",
                "model": "mlx-community/Llama-3.2-3B-Instruct-4bit",
            },
        )
    finally:
        client.close()

    assert captured
    call = captured[-1]
    assert call.get("tool_mode") == "small"
    assert call.get("allowed_tools") == ["read", "write", "edit", "process"]
    assert call.get("model_first_chunk_timeout_ms") == 30000


def test_chat_stream_routes_shell_and_file_to_small_allowlist(monkeypatch) -> None:
    client, captured = _make_openclaw_app(monkeypatch)
    try:
        _post_stream_payload(
            client,
            {
                "message": "Run a shell command to list files and read README.md.",
                "openclaw_enabled": True,
                "agent_runtime": "openclaw",
                "server_url": "http://127.0.0.1:8000",
                "model": "mlx-community/Llama-3.2-3B-Instruct-4bit",
            },
        )
    finally:
        client.close()

    assert captured
    call = captured[-1]
    assert call.get("tool_mode") == "small"
    assert call.get("allowed_tools") == ["read", "write", "edit", "process", "exec"]
    assert call.get("model_first_chunk_timeout_ms") == 30000


def test_chat_stream_routes_web_requests_to_small_allowlist(monkeypatch) -> None:
    client, captured = _make_openclaw_app(monkeypatch)
    try:
        _post_stream_payload(
            client,
            {
                "message": "Search the web for today's AI news and cite links.",
                "openclaw_enabled": True,
                "agent_runtime": "openclaw",
                "server_url": "http://127.0.0.1:8000",
                "model": "mlx-community/Llama-3.2-3B-Instruct-4bit",
            },
        )
    finally:
        client.close()

    assert captured
    call = captured[-1]
    assert call.get("tool_mode") == "small"
    assert call.get("allowed_tools") == [
        "read",
        "write",
        "edit",
        "process",
        "web_search",
        "web_fetch",
        "browser",
    ]
    assert call.get("model_first_chunk_timeout_ms") == 30000


def test_chat_stream_preserves_explicit_full_mode(monkeypatch) -> None:
    client, captured = _make_openclaw_app(monkeypatch)
    try:
        _post_stream_payload(
            client,
            {
                "message": "Summarize this quickly.",
                "openclaw_enabled": True,
                "agent_runtime": "openclaw",
                "tool_mode": "full",
                "server_url": "http://127.0.0.1:8000",
                "model": "mlx-community/Llama-3.2-3B-Instruct-4bit",
            },
        )
    finally:
        client.close()

    assert captured
    call = captured[-1]
    assert call.get("tool_mode") == "full"
    assert call.get("model_first_chunk_timeout_ms") == 30000


def test_chat_non_stream_uses_same_profile_base_defaults(monkeypatch) -> None:
    client, captured = _make_openclaw_app(monkeypatch)
    try:
        response = client.post(
            "/api/chat",
            json={
                "message": "Give me a short summary of retrieval augmented generation.",
                "openclaw_enabled": True,
                "agent_runtime": "openclaw",
                "server_url": "http://127.0.0.1:8000",
                "model": "mlx-community/Llama-3.2-3B-Instruct-4bit",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data.get("ok") is True
    finally:
        client.close()

    assert captured
    call = captured[-1]
    assert call.get("tool_mode") == "small"
    assert call.get("allowed_tools") == ["read", "write", "edit", "process"]
    assert call.get("model_first_chunk_timeout_ms") == 30000


def test_agent_chat_rejects_an_unconfigured_model() -> None:
    app = css_svg_ui.create_app("http://127.0.0.1:8000", 1024, 0.7)
    with TestClient(app) as client:
        response = client.post(
            "/api/chat",
            json={
                "message": "Use tools to inspect this project.",
                "openclaw_enabled": True,
                "agent_runtime": "openclaw",
                "server_url": "http://127.0.0.1:8000",
                "model": "unconfigured/model",
            },
        )

    assert response.status_code == 409
    assert response.json()["detail"]["error_type"] == "agent_profile_required"


def test_openclaw_runtime_config_sets_sampling_guards_for_openai_compat() -> None:
    runtime_target = css_svg_ui._resolve_openclaw_runtime_target(
        server_url="http://127.0.0.1:8000",
        selected_model="mlx-community/Llama-3.2-3B-Instruct-4bit",
        backend_override="openai-compat",
    )
    config = css_svg_ui._build_openclaw_runtime_config(runtime_target=runtime_target)
    provider_model = runtime_target["provider_model"]
    params = config["agents"]["defaults"]["models"][provider_model]["params"]

    assert params.get("repetition_penalty") == 1.08
    assert "My core truths are:" in params.get("stop", [])


def test_openclaw_runtime_config_uses_native_context_and_tool_profile(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        css_svg_ui,
        "read_model_metadata",
        lambda *_args, **_kwargs: {"max_context_length": 131072},
    )
    runtime_target = css_svg_ui._resolve_openclaw_runtime_target(
        server_url="http://127.0.0.1:8000",
        selected_model="ibm-granite/granite-4.0-1b",
        backend_override="openai-compat",
    )

    config = css_svg_ui._build_openclaw_runtime_config(
        runtime_target=runtime_target,
        tool_mode="small",
        allowed_tools=["read", "exec", "READ"],
    )
    model = config["models"]["providers"]["openai"]["models"][0]

    assert model["contextWindow"] == 131072
    assert config["tools"]["profile"] == "minimal"
    assert config["tools"]["alsoAllow"] == ["read", "exec"]


def test_agent_profile_is_runtime_default_and_user_generation_wins() -> None:
    profile = {
        "context_window": 32768,
        "generation_defaults": {"temperature": 0.35, "max_tokens": 2048},
        "openclaw": {
            "ready": True,
            "tool_calling": "passed",
            "reasoning_parser": "qwen3",
            "timeout_seconds": 240,
        },
        "hermes": {"ready": True, "timeout_seconds": 240, "max_turns": 10},
    }
    runtime_target = css_svg_ui._resolve_openclaw_runtime_target(
        server_url="http://127.0.0.1:8000",
        selected_model="owner/model",
        backend_override="openai-compat",
    )

    config = css_svg_ui._build_openclaw_runtime_config(
        runtime_target=runtime_target,
        tool_mode="small",
        allowed_tools=["read"],
        agent_profile=profile,
        generation_overrides={"temperature": 0.82, "max_tokens": 640},
    )
    model = config["models"]["providers"]["openai"]["models"][0]
    params = config["agents"]["defaults"]["models"][runtime_target["provider_model"]][
        "params"
    ]

    assert model["contextWindow"] == 32768
    assert model["maxTokens"] == 640
    assert model["compat"] == {"supportsTools": True, "thinkingFormat": "qwen"}
    assert params["temperature"] == 0.82


def test_openclaw_full_mode_uses_full_tool_profile(monkeypatch) -> None:
    monkeypatch.setattr(
        css_svg_ui,
        "read_model_metadata",
        lambda *_args, **_kwargs: {"max_context_length": 32768},
    )
    runtime_target = css_svg_ui._resolve_openclaw_runtime_target(
        server_url="http://127.0.0.1:8000",
        selected_model="local/model",
        backend_override="openai-compat",
    )

    config = css_svg_ui._build_openclaw_runtime_config(
        runtime_target=runtime_target,
        tool_mode="full",
        allowed_tools=["read"],
    )

    assert config["tools"]["profile"] == "full"
    assert "alsoAllow" not in config["tools"]


def test_openclaw_none_mode_disables_model_tool_schema(monkeypatch) -> None:
    monkeypatch.setattr(
        css_svg_ui,
        "read_model_metadata",
        lambda *_args, **_kwargs: {"max_context_length": 131072},
    )
    runtime_target = css_svg_ui._resolve_openclaw_runtime_target(
        server_url="http://127.0.0.1:8000",
        selected_model="local/model",
        backend_override="openai-compat",
    )

    config = css_svg_ui._build_openclaw_runtime_config(
        runtime_target=runtime_target,
        tool_mode="none",
    )
    model = config["models"]["providers"]["openai"]["models"][0]

    assert model["compat"] == {"supportsTools": False}
    assert config["skills"]["limits"]["maxSkillsInPrompt"] == 0
    assert config["skills"]["limits"]["maxSkillsPromptChars"] == 0


def test_hermes_runtime_config_tracks_selected_local_model(
    monkeypatch, tmp_path
) -> None:
    hermes_home = tmp_path / "home"
    config_path = hermes_home / "config.yaml"
    monkeypatch.setattr(css_svg_ui, "HERMES_HOME_DIR", hermes_home)
    monkeypatch.setattr(css_svg_ui, "HERMES_CONFIG_PATH", config_path)
    monkeypatch.setattr(
        css_svg_ui,
        "read_model_metadata",
        lambda *_args, **_kwargs: {"max_context_length": 262144},
    )

    css_svg_ui._sync_hermes_runtime_config(
        server_url="http://127.0.0.1:8000",
        selected_model="Qwen/Qwen3.5-2B",
    )
    config = css_svg_ui.yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["model"] == {
        "provider": "custom",
        "default": "Qwen/Qwen3.5-2B",
        "base_url": "http://127.0.0.1:8000/v1",
        "api_mode": "chat_completions",
        "api_key": "token-workshed-local",
        "context_length": 262144,
        "ollama_num_ctx": 262144,
    }
    assert config["display"]["show_reasoning"] is False
    assert config["security"]["tirith_enabled"] is False
    assert config["platform_toolsets"]["cli"] == []


def test_hermes_profile_context_and_user_max_tokens_override(
    monkeypatch, tmp_path
) -> None:
    hermes_home = tmp_path / "home"
    config_path = hermes_home / "config.yaml"
    monkeypatch.setattr(css_svg_ui, "HERMES_HOME_DIR", hermes_home)
    monkeypatch.setattr(css_svg_ui, "HERMES_CONFIG_PATH", config_path)
    profile = {
        "context_window": 65536,
        "generation_defaults": {"max_tokens": 4096},
        "openclaw": {"ready": True},
        "hermes": {"ready": True, "timeout_seconds": 240, "max_turns": 10},
    }

    css_svg_ui._sync_hermes_runtime_config(
        server_url="http://127.0.0.1:8000",
        selected_model="owner/model",
        agent_profile=profile,
        generation_overrides={"max_tokens": 512, "temperature": 0.8},
    )
    config = css_svg_ui.yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["model"]["context_length"] == 65536
    assert config["model"]["ollama_num_ctx"] == 65536
    assert config["model"]["max_tokens"] == 512
    assert config["model"]["temperature"] == 0.8


def test_hermes_toolsets_are_narrowed_by_request_intent() -> None:
    assert (
        css_svg_ui._resolve_hermes_toolsets_for_message("Run pwd in the shell.")
        == "terminal"
    )
    assert (
        css_svg_ui._resolve_hermes_toolsets_for_message(
            "Run a command, then read file README.md."
        )
        == "terminal,file"
    )
    assert (
        css_svg_ui._resolve_hermes_toolsets_for_message(
            "Search the web for release notes."
        )
        == "web"
    )
    assert (
        css_svg_ui._resolve_hermes_toolsets_for_message(
            "Explain retrieval in one line."
        )
        is None
    )


def test_openclaw_turn_message_includes_persona_manifesto_guard() -> None:
    message = css_svg_ui._compose_openclaw_turn_message(
        system_prompt="",
        user_message="Who are you?",
        user_content=None,
    )
    assert "Never recite internal identity/persona manifesto text" in message


def test_sanitize_assistant_output_strips_manifesto_and_dedupes() -> None:
    raw = (
        "I'm an evolving entity, designed to assist and learn alongside you. My core truths are:\n\n"
        "1. Be genuinely helpful, not performatively helpful.\n"
        "2. Have opinions.\n"
        "3. Be resourceful before asking.\n"
        "4. Earn trust through competence.\n\n"
        "结论：缓存目录占用了主要空间。\n\n"
        "结论：缓存目录占用了主要空间。"
    )
    cleaned = css_svg_ui._sanitize_assistant_output_text(raw)
    assert "core truths" not in cleaned.lower()
    assert "be genuinely helpful" not in cleaned.lower()
    assert cleaned.count("结论：缓存目录占用了主要空间。") == 1


def test_chat_stream_strips_manifesto_and_duplicate_answer(monkeypatch) -> None:
    client, _captured = _make_openclaw_app(
        monkeypatch,
        reply_text=(
            "Final Answer: I'm an evolving entity, designed to assist and learn alongside you. "
            "My core truths are:\n\n"
            "1. Be genuinely helpful, not performatively helpful.\n"
            "2. Have opinions.\n"
            "3. Be resourceful before asking.\n"
            "4. Earn trust through competence.\n\n"
            "结论：缓存目录占用了主要空间。\n\n"
            "结论：缓存目录占用了主要空间。"
        ),
    )
    try:
        events = _post_stream_payload(
            client,
            {
                "message": "Please diagnose disk usage.",
                "openclaw_enabled": True,
                "agent_runtime": "openclaw",
                "server_url": "http://127.0.0.1:8000",
                "model": "mlx-community/Llama-3.2-3B-Instruct-4bit",
            },
        )
    finally:
        client.close()

    answer_text = "".join(
        str(event.get("text") or "")
        for event in events
        if str(event.get("type") or "") == "answer_delta"
    )
    assert "core truths" not in answer_text.lower()
    assert "be genuinely helpful" not in answer_text.lower()
    assert answer_text.count("结论：缓存目录占用了主要空间。") == 1
