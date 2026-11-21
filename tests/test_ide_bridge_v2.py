from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from token_workshed_nextcore.ide_bridge import (
    IDE_AGENT_PROTOCOL_VERSION,
    IDEBridge,
    IDEBridgeError,
    IDEPairingRequest,
    IDESessionChatRequest,
    register_ide_bridge_routes,
)


def _bridge(tmp_path, monkeypatch) -> IDEBridge:
    monkeypatch.setenv("TOKEN_WORKSHED_DESKTOP_STATE_PATH", str(tmp_path / "state.json"))
    return IDEBridge(
        get_models=lambda: ["one"],
        get_active_model=lambda: "one",
        switch_model=lambda model: (True, "switched"),
        get_server_url=lambda: "http://127.0.0.1:8010",
        default_server_url="http://127.0.0.1:8010",
        default_max_tokens=1024,
        default_temperature=0.2,
    )


def _v2_pairing() -> IDEPairingRequest:
    return IDEPairingRequest(
        client_id="external-agent-client",
        client_name="External IDE",
        ide_name="IDE",
        plugin_version="0.2.1",
        protocol_version=IDE_AGENT_PROTOCOL_VERSION,
        requested_scopes=["ide.sessions.chat"],
    )


def test_ide_status_advertises_no_agent_or_v2_access(tmp_path, monkeypatch):
    bridge = _bridge(tmp_path, monkeypatch)

    assert bridge.status()["agent_access"] is False
    assert bridge.status()["supported_protocol_versions"] == ["1"]
    assert bridge.agent_status()["agent_access"] is False
    assert bridge.agent_status()["capabilities"] == []
    assert "runtime" not in bridge.agent_status()


def test_v2_pairing_and_direct_runner_injection_are_rejected(tmp_path, monkeypatch):
    bridge = _bridge(tmp_path, monkeypatch)

    with pytest.raises(IDEBridgeError) as error:
        bridge.create_pairing(_v2_pairing())
    assert error.value.status_code == 403
    assert "NextCore chat" in error.value.message

    with pytest.raises(TypeError):
        IDEBridge(
            get_models=lambda: [],
            get_active_model=lambda: None,
            switch_model=lambda model: (True, "switched"),
            get_server_url=lambda: "http://127.0.0.1:8010",
            default_server_url="http://127.0.0.1:8010",
            default_max_tokens=256,
            default_temperature=0.2,
            agent_runner=lambda **_: {"reply": "should never run"},
        )


def test_ide_agent_endpoints_are_denied_even_with_app_owned_token(tmp_path, monkeypatch):
    bridge = _bridge(tmp_path, monkeypatch)
    bridge.register_internal_authorization(
        client_id="nextcore-acp-test", token="app-owned-test-token"
    )
    app = FastAPI()
    register_ide_bridge_routes(app, bridge)

    with TestClient(app) as client:
        status = client.get("/api/ide/v2/status")
        pairing = client.post(
            "/api/ide/v2/pairing/requests",
            json=_v2_pairing().model_dump(),
        )
        sessions = client.post(
            "/api/ide/v2/sessions",
            headers={"Authorization": "Bearer app-owned-test-token"},
            json={"title": "External Agent", "model": "one"},
        )
        skills = client.get(
            "/api/ide/v2/skills",
            headers={"Authorization": "Bearer app-owned-test-token"},
        )

    assert status.status_code == 200
    assert status.json()["agent_access"] is False
    assert pairing.status_code == 403
    assert sessions.status_code == 403
    assert skills.status_code == 403


def test_ide_agent_request_has_no_runtime_selector():
    with pytest.raises(ValidationError):
        IDESessionChatRequest(
            request_id="external-request-1",
            message="Run a command",
            runtime="hermes",
        )
