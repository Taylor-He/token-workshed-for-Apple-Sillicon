from __future__ import annotations

import json
import threading
import time

from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm_mlx import jetbrains_integration as integration_module
from vllm_mlx.ide_bridge import IDEBridge
from vllm_mlx.jetbrains_integration import (
    ensure_acp_token,
    ensure_provider_key,
    read_discovery_record,
    register_jetbrains_integration_routes,
)


class FakeManager:
    def active_model(self):
        return "local-model"

    def discover_models(self):
        return ["local-model"]

    def active_server_url(self):
        return "http://127.0.0.1:8000"


def _bridge(tmp_path, monkeypatch):
    monkeypatch.setenv("TOKEN_WORKSHED_DESKTOP_STATE_PATH", str(tmp_path / "state.json"))
    return IDEBridge(
        get_models=lambda: ["local-model"],
        get_active_model=lambda: "local-model",
        switch_model=lambda model: (True, "switched"),
        get_server_url=lambda: "http://127.0.0.1:8000",
        default_server_url="http://127.0.0.1:8000",
        default_max_tokens=512,
        default_temperature=0.2,
        agent_runner=lambda **_: {"reply": "ok"},
    )


def test_discovery_is_loopback_only_and_provider_key_is_not_in_plugin_config(
    tmp_path, monkeypatch
):
    bridge = _bridge(tmp_path, monkeypatch)
    app = FastAPI()
    register_jetbrains_integration_routes(
        app,
        manager=FakeManager(),
        ui_url="http://127.0.0.1:7862",
        manager_api_token=None,
        bridge=bridge,
    )
    record = read_discovery_record()
    assert record["ui_url"] == "http://127.0.0.1:7862"
    assert read_discovery_record()["provider_url"].startswith("http://127.0.0.1")

    with TestClient(app) as client:
        config = client.get("/api/jetbrains/integration")
        assert config.status_code == 200
        assert config.json()["provider"]["api_key"]
        assert config.json()["intellij_mcp_return"]["default_enabled"] is False
        plugin_config = client.get(
            "/api/ide/v2/integration",
            headers={"Authorization": "Bearer not-authorized"},
        )
        assert plugin_config.status_code == 401


def test_provider_route_rejects_unscoped_fields_but_preserves_empty_tools(
    tmp_path, monkeypatch
):
    bridge = _bridge(tmp_path, monkeypatch)
    app = FastAPI()
    register_jetbrains_integration_routes(
        app,
        manager=FakeManager(),
        ui_url="http://127.0.0.1:7862",
        manager_api_token=None,
        bridge=bridge,
    )
    provider_key = ensure_provider_key()
    monkeypatch.setattr(
        "vllm_mlx.jetbrains_integration._forward_completion",
        lambda **kwargs: {"ok": True, "tools": kwargs["body"].get("tools")},
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {provider_key}"},
            json={
                "model": "local-model",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [],
                "stream": False,
            },
        )
        assert response.status_code == 200
        assert response.json()["tools"] == []
        rejected = client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {provider_key}"},
            json={
                "model": "local-model",
                "messages": [],
                "manager_api_token": "must-not-pass",
            },
        )
        assert rejected.status_code == 400


def test_mcp_gateway_keeps_read_only_automatic_and_requires_app_decision_for_writes(
    tmp_path, monkeypatch
):
    bridge = _bridge(tmp_path, monkeypatch)
    app = FastAPI()
    register_jetbrains_integration_routes(
        app,
        manager=FakeManager(),
        ui_url="http://127.0.0.1:7862",
        manager_api_token=None,
        bridge=bridge,
    )
    monkeypatch.setattr(
        integration_module,
        "_mcp_tools",
        lambda _upstream: [
            {"name": "read_file", "description": "read", "inputSchema": {"type": "object"}},
            {"name": "write_file", "description": "write", "inputSchema": {"type": "object"}},
        ],
    )
    calls: list[str] = []
    monkeypatch.setattr(
        integration_module,
        "_mcp_call",
        lambda _upstream, name, _arguments: calls.append(name)
        or {"isError": False, "content": [{"type": "text", "text": "ok"}]},
    )
    token = ensure_acp_token()
    headers = {"Authorization": f"Bearer {token}"}

    with TestClient(app) as client:
        listed = client.get("/api/jetbrains/mcp/tools", headers=headers)
        assert listed.status_code == 200
        assert {item["name"] for item in listed.json()["tools"]} == {"read_file", "write_file"}

        read = client.post(
            "/api/jetbrains/mcp/call",
            headers=headers,
            json={
                "name": "read_file",
                "arguments": {"path": "src/app.py"},
                "session_id": "s-read",
            },
        )
        assert read.status_code == 200
        assert read.json()["isError"] is False
        assert calls == ["read_file"]

        result: dict[str, object] = {}

        def request_write() -> None:
            response = client.post(
                "/api/jetbrains/mcp/call",
                headers=headers,
                json={
                    "name": "write_file",
                    "arguments": {"path": "src/app.py"},
                    "session_id": "s-write",
                },
            )
            result.update({"status": response.status_code, "body": response.json()})

        worker = threading.Thread(target=request_write, daemon=True)
        worker.start()
        deadline = time.time() + 2
        while time.time() < deadline and not app.state.jetbrains_mcp_gate.public():
            time.sleep(0.01)
        pending = app.state.jetbrains_mcp_gate.public()
        assert len(pending) == 1
        call_id = pending[0]["id"]
        decision = client.post(
            f"/api/jetbrains/mcp/decision/{call_id}",
            json={"approved": False},
        )
        assert decision.status_code == 200
        worker.join(2)
        assert not worker.is_alive()
        assert result["status"] == 200
        assert result["body"]["isError"] is True
        assert calls == ["read_file"]

        audit = client.get("/api/jetbrains/mcp/audit")
        assert audit.status_code == 200
        assert "s-write" in json.dumps(audit.json())
