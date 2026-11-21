from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from token_workshed_nextcore.ide_bridge import IDEBridge
from token_workshed_nextcore.jetbrains_integration import (
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
    monkeypatch.setenv(
        "TOKEN_WORKSHED_DESKTOP_STATE_PATH", str(tmp_path / "state.json")
    )
    return IDEBridge(
        get_models=lambda: ["local-model"],
        get_active_model=lambda: "local-model",
        switch_model=lambda model: (True, "switched"),
        get_server_url=lambda: "http://127.0.0.1:8000",
        default_server_url="http://127.0.0.1:8000",
        default_max_tokens=512,
        default_temperature=0.2,
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
        "token_workshed_nextcore.jetbrains_integration._forward_completion",
        lambda **kwargs: {
            "ok": True,
            "tools": kwargs["body"].get("tools"),
            "chat_template_kwargs": kwargs["body"].get("chat_template_kwargs"),
        },
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {provider_key}"},
            json={
                "model": "local-model",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [],
                "chat_template_kwargs": {"enable_thinking": False},
                "stream": False,
            },
        )
        assert response.status_code == 200
        assert response.json()["tools"] == []
        assert response.json()["chat_template_kwargs"] == {"enable_thinking": False}
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


def test_mcp_gateway_does_not_expose_agent_tools_to_external_clients(
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
    token = ensure_acp_token()
    headers = {"Authorization": f"Bearer {token}"}

    with TestClient(app) as client:
        listed = client.get("/api/jetbrains/mcp/tools", headers=headers)
        assert listed.status_code == 200
        assert listed.json()["tools"] == []
        assert listed.json()["count"] == 0

        call = client.post(
            "/api/jetbrains/mcp/call",
            headers=headers,
            json={
                "name": "terminal",
                "arguments": {"command": "pwd"},
                "session_id": "external-mcp",
            },
        )
        assert call.status_code == 403
        assert "only from NextCore chat" in call.json()["detail"]
