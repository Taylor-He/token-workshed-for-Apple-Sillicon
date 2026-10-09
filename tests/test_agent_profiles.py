# SPDX-License-Identifier: Apache-2.0
"""Coverage for model-level Configure AI Agents profiles."""

from __future__ import annotations

import json
import time
from typing import Any

import pytest
from fastapi.testclient import TestClient

from vllm_mlx import css_svg_ui
from vllm_mlx.agent_profiles import (
    delete_agent_profile,
    load_agent_profile,
    save_agent_profile,
)


def _profile(model: str = "owner/model") -> dict[str, Any]:
    return {
        "profile_version": 1,
        "model_id": model,
        "configured_at": 1,
        "family": "qwen3",
        "context_window": 32768,
        "chat_template": "{{ messages }}",
        "generation_defaults": {"temperature": 0.6},
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


def _success_probe(*, runtime: str, **_kwargs: Any) -> dict[str, Any]:
    return {
        "ok": True,
        "runtime": runtime,
        "checked_at": 1.0,
        "status": 200,
        "agent_ready": True,
        "tool_calling": "passed",
        "parser_used": "qwen",
        "reasoning_parser_used": "qwen3",
        "chat_template_used": "{{ messages }}",
        "needs_manual_review": False,
        "failures": [],
        "fix_suggestions": [],
    }


def _wait_for_job(
    client: TestClient,
    job_id: str,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    for _ in range(80):
        response = client.get(f"/api/agent/profile/configure/{job_id}", headers=headers)
        assert response.status_code == 200
        job = response.json()["job"]
        if job["state"] in {"configured", "failed"}:
            return job
        time.sleep(0.01)
    raise AssertionError("Agent Profile job did not complete in time")


def test_profile_storage_is_secret_free_and_legacy_state_safe(
    monkeypatch, tmp_path
) -> None:
    state_path = tmp_path / "desktop_state.json"
    monkeypatch.setenv("TOKEN_WORKSHED_DESKTOP_STATE_PATH", str(state_path))
    state_path.write_text(
        json.dumps({"last_model": "owner/model", "legacy": True}), encoding="utf-8"
    )

    profile = _profile()
    profile["api_key"] = "must-not-persist"
    profile["openclaw"]["probe"] = {
        "token": "must-not-persist",
        "reason": "Authorization: Bearer sk-must-not-persist-12345678",
    }
    profile["warnings"] = [
        "https://example.invalid/check?access_token=must-not-persist"
    ]
    saved = save_agent_profile(profile)

    assert saved["model_id"] == "owner/model"
    raw = state_path.read_text(encoding="utf-8")
    assert "must-not-persist" not in raw
    assert load_agent_profile("owner/model") is not None
    assert delete_agent_profile("owner/model") is True
    assert load_agent_profile("owner/model") is None
    assert json.loads(state_path.read_text(encoding="utf-8"))["legacy"] is True


def test_profile_storage_rejects_runtime_that_did_not_pass_tool_probe(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv(
        "TOKEN_WORKSHED_DESKTOP_STATE_PATH", str(tmp_path / "desktop_state.json")
    )
    profile = _profile()
    profile["hermes"]["tool_calling"] = "failed"

    with pytest.raises(ValueError, match="both runtimes must be ready"):
        save_agent_profile(profile)

    assert load_agent_profile("owner/model") is None


def test_profile_storage_accepts_healthy_runtime_with_degraded_tools(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv(
        "TOKEN_WORKSHED_DESKTOP_STATE_PATH", str(tmp_path / "desktop_state.json")
    )
    profile = _profile()
    profile["openclaw"]["tool_calling"] = "degraded"
    profile["hermes"]["tool_calling"] = "degraded"
    profile["warnings"] = ["Structured tool calls were not confirmed."]

    saved = save_agent_profile(profile)

    assert saved["openclaw"]["tool_calling"] == "degraded"
    assert saved["hermes"]["tool_calling"] == "degraded"
    assert load_agent_profile("owner/model") is not None


def test_profile_storage_keeps_legacy_ready_profile_usable(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv(
        "TOKEN_WORKSHED_DESKTOP_STATE_PATH", str(tmp_path / "desktop_state.json")
    )
    profile = _profile()
    profile["openclaw"].pop("tool_calling")
    profile["hermes"].pop("tool_calling")

    saved = save_agent_profile(profile)

    assert "tool_calling" not in saved["openclaw"]
    assert "tool_calling" not in saved["hermes"]
    assert load_agent_profile("owner/model") is not None


def test_configure_job_persists_degraded_runtime_diagnostics(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv(
        "TOKEN_WORKSHED_DESKTOP_STATE_PATH", str(tmp_path / "state.json")
    )
    monkeypatch.setattr(
        css_svg_ui,
        "read_model_metadata",
        lambda *_args, **_kwargs: {"generation_config": {}, "warnings": []},
    )

    def degraded_probe(*, runtime: str, **kwargs: Any) -> dict[str, Any]:
        result = _success_probe(runtime=runtime, **kwargs)
        result.update(
            tool_calling="degraded",
            degraded=True,
            warnings=["Structured tool calls were not confirmed."],
        )
        return result

    monkeypatch.setattr(css_svg_ui, "_run_agent_runtime_probe", degraded_probe)
    app = css_svg_ui.create_app("http://127.0.0.1:8000", 1024, 0.7)
    css_svg_ui._run_agent_profile_configure_job(
        app,
        job_id="degraded-tools",
        model_id="owner/model",
        server_url="http://127.0.0.1:8000",
    )

    job = css_svg_ui._get_agent_profile_job(app, "degraded-tools")
    assert job is not None
    assert job["state"] == "configured"
    assert job["profile"]["openclaw"]["tool_calling"] == "degraded"
    assert job["profile"]["hermes"]["tool_calling"] == "degraded"
    assert any("limited" in item for item in job["profile"]["warnings"])


def test_configure_job_uses_metadata_fallback_and_ordered_stages(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv(
        "TOKEN_WORKSHED_DESKTOP_STATE_PATH", str(tmp_path / "state.json")
    )
    monkeypatch.setattr(
        css_svg_ui,
        "read_model_metadata",
        lambda *_args, **_kwargs: {
            "max_context_length": None,
            "chat_template": None,
            "generation_config": {},
            "warnings": ["metadata was not cached"],
        },
    )
    monkeypatch.setattr(
        css_svg_ui, "_query_local_model_context_window", lambda **_kwargs: None
    )
    monkeypatch.setattr(css_svg_ui, "_run_agent_runtime_probe", _success_probe)
    app = css_svg_ui.create_app("http://127.0.0.1:8000", 1024, 0.7)
    history: list[tuple[str, int]] = []
    original_set = css_svg_ui._set_agent_profile_job

    def record_stage(*args: Any, **kwargs: Any) -> dict[str, Any]:
        result = original_set(*args, **kwargs)
        history.append(
            (str(result.get("phase") or ""), int(result.get("progress") or 0))
        )
        return result

    monkeypatch.setattr(css_svg_ui, "_set_agent_profile_job", record_stage)
    css_svg_ui._run_agent_profile_configure_job(
        app,
        job_id="ordered-stages",
        model_id="owner/model",
        server_url="http://127.0.0.1:8000",
    )

    job = css_svg_ui._get_agent_profile_job(app, "ordered-stages")
    assert job is not None
    assert job["state"] == "configured"
    assert job["profile"]["context_window"] == 16384
    stages: list[str] = []
    for phase, _progress in history:
        if not stages or stages[-1] != phase:
            stages.append(phase)
    assert stages == ["metadata", "openclaw probe", "hermes probe", "persist"]
    assert [progress for _phase, progress in history] == sorted(
        progress for _phase, progress in history
    )


def test_configure_job_requires_both_runtime_probes(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(
        "TOKEN_WORKSHED_DESKTOP_STATE_PATH", str(tmp_path / "state.json")
    )
    monkeypatch.setattr(
        css_svg_ui,
        "read_model_metadata",
        lambda *_args, **_kwargs: {
            "max_context_length": 32768,
            "chat_template": "{{ messages }}",
            "generation_config": {},
            "warnings": [],
        },
    )

    def failed_hermes(*, runtime: str, **kwargs: Any) -> dict[str, Any]:
        result = _success_probe(runtime=runtime, **kwargs)
        if runtime == "hermes":
            result.update(
                ok=False,
                agent_ready=False,
                tool_calling="failed",
                reason="Hermes smoke failed",
            )
        return result

    monkeypatch.setattr(css_svg_ui, "_run_agent_runtime_probe", failed_hermes)
    app = css_svg_ui.create_app("http://127.0.0.1:8000", 1024, 0.7)
    with TestClient(app) as client:
        started = client.post(
            "/api/agent/profile/configure",
            json={"model": "owner/model", "server_url": "http://127.0.0.1:8000"},
        ).json()
        job = _wait_for_job(client, started["job_id"])

    assert job["state"] == "failed"
    assert job["progress"] == 90
    assert "Hermes" in job["error"]
    assert load_agent_profile("owner/model") is None


def test_recalibration_failure_keeps_previous_profile(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(
        "TOKEN_WORKSHED_DESKTOP_STATE_PATH", str(tmp_path / "state.json")
    )
    monkeypatch.setattr(
        css_svg_ui,
        "read_model_metadata",
        lambda *_args, **_kwargs: {"generation_config": {}, "warnings": []},
    )
    previous = _profile()
    previous["configured_at"] = 123
    save_agent_profile(previous)

    def failed_recalibration(*, runtime: str, **kwargs: Any) -> dict[str, Any]:
        result = _success_probe(runtime=runtime, **kwargs)
        result.update(
            ok=False,
            agent_ready=False,
            tool_calling="failed",
            status=422,
            reason="parser mismatch",
        )
        return result

    monkeypatch.setattr(css_svg_ui, "_run_agent_runtime_probe", failed_recalibration)
    app = css_svg_ui.create_app("http://127.0.0.1:8000", 1024, 0.7)
    css_svg_ui._run_agent_profile_configure_job(
        app,
        job_id="failed-recalibration",
        model_id="owner/model",
        server_url="http://127.0.0.1:8000",
    )

    job = css_svg_ui._get_agent_profile_job(app, "failed-recalibration")
    assert job is not None
    assert job["state"] == "failed"
    stored = load_agent_profile("owner/model")
    assert stored is not None
    assert stored["configured_at"] == 123


def test_configure_endpoint_persists_profile_force_and_auth(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv(
        "TOKEN_WORKSHED_DESKTOP_STATE_PATH", str(tmp_path / "state.json")
    )
    monkeypatch.setattr(
        css_svg_ui,
        "read_model_metadata",
        lambda *_args, **_kwargs: {
            "max_context_length": None,
            "chat_template": None,
            "generation_config": {"temperature": 0.4},
            "warnings": ["metadata fallback"],
        },
    )
    monkeypatch.setattr(
        css_svg_ui, "_query_local_model_context_window", lambda **_kwargs: 65536
    )
    monkeypatch.setattr(css_svg_ui, "_run_agent_runtime_probe", _success_probe)
    app = css_svg_ui.create_app("http://127.0.0.1:8000", 1024, 0.7)
    app.state.manager_api_token = "manager-test-token"

    with TestClient(app) as client:
        unauthorized = client.get("/api/agent/profile?model=owner/model")
        assert unauthorized.status_code == 401
        headers = {"x-token-workshed-manager-token": "manager-test-token"}
        started = client.post(
            "/api/agent/profile/configure",
            headers=headers,
            json={"model": "owner/model", "server_url": "http://127.0.0.1:8000"},
        ).json()
        job = _wait_for_job(client, started["job_id"], headers=headers)
        assert job["state"] == "configured"
        assert job["progress"] == 100
        configured = client.get(
            "/api/agent/profile?model=owner/model", headers=headers
        ).json()
        assert configured["configured"] is True
        assert configured["profile"]["context_window"] == 65536
        assert configured["profile"]["openclaw"]["tool_call_parser"] == "qwen"
        assert configured["profile"]["hermes"]["max_turns"] == 10

        reused = client.post(
            "/api/agent/profile/configure",
            headers=headers,
            json={"model": "owner/model", "server_url": "http://127.0.0.1:8000"},
        ).json()
        assert reused["reused"] is True

        forced = client.post(
            "/api/agent/profile/configure",
            headers=headers,
            json={
                "model": "owner/model",
                "server_url": "http://127.0.0.1:8000",
                "force": True,
            },
        ).json()
        forced_job = _wait_for_job(client, forced["job_id"], headers=headers)
        assert forced_job["state"] == "configured"


def test_configure_retries_transient_bridge_failures(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(
        "TOKEN_WORKSHED_DESKTOP_STATE_PATH", str(tmp_path / "state.json")
    )
    monkeypatch.setattr(
        css_svg_ui,
        "read_model_metadata",
        lambda *_args, **_kwargs: {"generation_config": {}, "warnings": []},
    )
    calls: dict[str, int] = {}

    def flaky_probe(*, runtime: str, **kwargs: Any) -> dict[str, Any]:
        calls[runtime] = calls.get(runtime, 0) + 1
        if runtime == "openclaw" and calls[runtime] == 1:
            return {
                "ok": False,
                "runtime": runtime,
                "status": 500,
                "reason": "bridge warming up",
                "agent_ready": False,
                "tool_calling": "failed",
            }
        return _success_probe(runtime=runtime, **kwargs)

    monkeypatch.setattr(css_svg_ui, "_run_agent_runtime_probe", flaky_probe)
    app = css_svg_ui.create_app("http://127.0.0.1:8000", 1024, 0.7)
    with TestClient(app) as client:
        started = client.post(
            "/api/agent/profile/configure",
            json={"model": "owner/model", "server_url": "http://127.0.0.1:8000"},
        ).json()
        job = _wait_for_job(client, started["job_id"])

    assert job["state"] == "configured"
    assert calls == {"openclaw": 2, "hermes": 1}


def test_profile_tool_policy_routes_safe_base_and_on_demand_tools() -> None:
    profile = _profile()
    normal = css_svg_ui._resolve_openclaw_turn_request_options(
        user_message="Explain this repository.",
        tool_mode=None,
        allowed_tools=None,
        model_first_chunk_timeout_ms=None,
        server_url="http://127.0.0.1:8000",
        selected_model="owner/model",
        agent_runtime="openclaw",
        agent_profile=profile,
    )
    assert normal["allowed_tools"] == ["read", "write", "edit", "process"]

    terminal = css_svg_ui._resolve_openclaw_turn_request_options(
        user_message="Run pwd in the terminal.",
        tool_mode=None,
        allowed_tools=None,
        model_first_chunk_timeout_ms=None,
        server_url="http://127.0.0.1:8000",
        selected_model="owner/model",
        agent_runtime="openclaw",
        agent_profile=profile,
    )
    assert terminal["allowed_tools"][-1] == "exec"

    web = css_svg_ui._resolve_openclaw_turn_request_options(
        user_message="Search the web for release notes.",
        tool_mode=None,
        allowed_tools=None,
        model_first_chunk_timeout_ms=None,
        server_url="http://127.0.0.1:8000",
        selected_model="owner/model",
        agent_runtime="openclaw",
        agent_profile=profile,
    )
    assert web["allowed_tools"][-3:] == ["web_search", "web_fetch", "browser"]
    assert (
        css_svg_ui._resolve_hermes_toolsets_for_message(
            "Explain this repository.", agent_profile=profile
        )
        == "file"
    )

    degraded = _profile()
    degraded["openclaw"]["tool_calling"] = "degraded"
    degraded["hermes"]["tool_calling"] = "degraded"
    assert (
        css_svg_ui._resolve_openclaw_turn_request_options(
            user_message="Explain this repository.",
            tool_mode=None,
            allowed_tools=None,
            model_first_chunk_timeout_ms=None,
            server_url="http://127.0.0.1:8000",
            selected_model="owner/model",
            agent_runtime="openclaw",
            agent_profile=degraded,
        )["tool_mode"]
        == "none"
    )
    assert (
        css_svg_ui._resolve_hermes_toolsets_for_message(
            "Explain this repository.", agent_profile=degraded
        )
        is None
    )
    assert (
        css_svg_ui._resolve_hermes_toolsets_for_message(
            "Run pwd in the terminal.", agent_profile=degraded
        )
        is None
    )
