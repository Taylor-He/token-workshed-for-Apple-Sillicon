from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm_mlx.ide_bridge import (
    IDEAssistRequest,
    IDEBridge,
    IDEBridgeError,
    IDEContextItem,
    IDEPairingRequest,
    IDE_PROTOCOL_VERSION,
    register_ide_bridge_routes,
)


def _bridge(*, launch_companion=None) -> IDEBridge:
    return IDEBridge(
        get_models=lambda: ["demo-model"],
        get_active_model=lambda: "demo-model",
        switch_model=lambda model: (True, f"switched to {model}"),
        get_server_url=lambda: "http://127.0.0.1:8000",
        default_server_url="http://127.0.0.1:8000",
        default_max_tokens=2048,
        default_temperature=0.2,
        launch_companion=launch_companion,
    )


def test_pairing_creates_scoped_credential_and_revokes(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TOKEN_WORKSHED_DESKTOP_STATE_PATH", str(tmp_path / "state.json"))
    bridge = _bridge()
    pairing = bridge.create_pairing(
        IDEPairingRequest(
            client_id="jetbrains-client-123",
            client_name="Token Workshed for JetBrains",
            ide_name="PyCharm",
            plugin_version="0.2.0",
            protocol_version=IDE_PROTOCOL_VERSION,
        )
    )
    pairing_id = pairing["pairing_id"]
    assert bridge.pending_pairings()[0]["client_id"] == "jetbrains-client-123"

    bridge.decide_pairing(pairing_id, approved=True)
    approved = bridge.pairing_status(pairing_id)
    credential = approved["credential"]
    record = bridge.authorize(
        SimpleNamespace(headers={"authorization": f"Bearer {credential}"}),
        "ide.assist.stream",
    )
    assert record["client_id"] == "jetbrains-client-123"
    assert bridge.revoke_authorization("jetbrains-client-123") is True
    with pytest.raises(IDEBridgeError, match="not authorized"):
        bridge.authorize(
            SimpleNamespace(headers={"authorization": f"Bearer {credential}"}),
            "ide.assist.stream",
        )


def test_sensitive_context_needs_explicit_confirmation() -> None:
    bridge = _bridge()
    payload = IDEAssistRequest(
        message="Explain this config",
        context=[IDEContextItem(path=".env", language="dotenv", content="KEY=secret")],
    )
    with pytest.raises(IDEBridgeError, match="Confirm this attachment"):
        bridge._prepare_context(payload.context)


def test_pairing_rejects_incompatible_protocol() -> None:
    bridge = _bridge()
    with pytest.raises(IDEBridgeError, match="Incompatible IDE bridge protocol"):
        bridge.create_pairing(
            IDEPairingRequest(
                client_id="jetbrains-client-123",
                client_name="Token Workshed for JetBrains",
                ide_name="PyCharm",
                plugin_version="0.2.0",
                protocol_version="999",
            )
        )


def test_rejected_or_expired_pairing_never_issues_a_credential() -> None:
    bridge = _bridge()
    rejected = bridge.create_pairing(
        IDEPairingRequest(
            client_id="jetbrains-client-123",
            client_name="Token Workshed for JetBrains",
            ide_name="PyCharm",
            plugin_version="0.2.0",
            protocol_version=IDE_PROTOCOL_VERSION,
        )
    )
    bridge.decide_pairing(rejected["pairing_id"], approved=False)
    assert bridge.pairing_status(rejected["pairing_id"])["status"] == "rejected"

    expired = bridge.create_pairing(
        IDEPairingRequest(
            client_id="jetbrains-client-456",
            client_name="Token Workshed for JetBrains",
            ide_name="PyCharm",
            plugin_version="0.2.0",
            protocol_version=IDE_PROTOCOL_VERSION,
        )
    )
    bridge._pending[expired["pairing_id"]]["created_at"] = 0
    with pytest.raises(IDEBridgeError, match="not found or expired"):
        bridge.pairing_status(expired["pairing_id"])


def test_companion_open_uses_a_separate_scoped_capability(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TOKEN_WORKSHED_DESKTOP_STATE_PATH", str(tmp_path / "state.json"))
    opened = []
    bridge = _bridge(launch_companion=lambda: opened.append(True) or True)
    pairing = bridge.create_pairing(
        IDEPairingRequest(
            client_id="jetbrains-client-123",
            client_name="Token Workshed for JetBrains",
            ide_name="PyCharm",
            plugin_version="0.2.0",
            protocol_version=IDE_PROTOCOL_VERSION,
        )
    )
    bridge.decide_pairing(pairing["pairing_id"], approved=True)
    credential = bridge.pairing_status(pairing["pairing_id"])["credential"]
    bridge.authorize(
        SimpleNamespace(headers={"authorization": f"Bearer {credential}"}),
        "ide.companion.open",
    )
    assert bridge.open_companion() == {"ok": True, "opened": True}
    assert opened == [True]


def test_routes_reject_agent_fields_before_the_bridge_can_handle_them() -> None:
    app = FastAPI()
    register_ide_bridge_routes(app, _bridge())
    response = TestClient(app).post(
        "/api/ide/v1/pairing/requests",
        json={
            "client_id": "jetbrains-client-123",
            "client_name": "Token Workshed for JetBrains",
            "ide_name": "PyCharm",
            "plugin_version": "0.2.0",
            "protocol_version": IDE_PROTOCOL_VERSION,
            "agent": True,
        },
    )
    assert response.status_code == 422


def test_artifacts_cannot_escape_context_or_use_shell() -> None:
    bridge = _bridge()
    prepared = bridge._prepare_context(
        [
            IDEContextItem(
                path="src/app.py",
                language="python",
                content="print('before')",
                selection_start_offset=0,
                selection_end_offset=15,
            )
        ]
    )
    allowed_hash = prepared[0]["base_sha256"]
    parsed = bridge._parse_artifacts(
        {
            "patches": [
                {
                    "path": "src/app.py",
                    "base_sha256": allowed_hash,
                    "content": "print('after')",
                    "create": False,
                },
                {
                    "path": "../outside.py",
                    "base_sha256": allowed_hash,
                    "content": "bad",
                    "create": False,
                },
            ],
            "commands": [
                {"argv": ["pytest", "-q"], "cwd": ".", "reason": "test"},
                {"argv": ["sh", "-c", "pytest -q"], "cwd": ".", "reason": "bad"},
            ],
        },
        "fix",
        prepared,
    )
    assert len(parsed["patches"]) == 1
    assert parsed["patches"][0]["selection_start_offset"] == 0
    assert len(parsed["commands"]) == 1
