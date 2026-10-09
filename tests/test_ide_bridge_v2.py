from __future__ import annotations

import hashlib
import json
import threading
import time
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm_mlx.ide_bridge import (
    IDE_AGENT_PROTOCOL_VERSION,
    IDE_AGENT_SCOPES,
    IDEBridge,
    IDEContextItem,
    IDEPairingRequest,
    IDESessionChatRequest,
    IDESessionCreateRequest,
    register_ide_bridge_routes,
)


def _runner(**kwargs):
    return {"reply": f"App reply: {kwargs['message']}", "model": kwargs["model"], "metrics": {"latency_ms": 1}}


def _bridge(tmp_path, monkeypatch):
    monkeypatch.setenv("TOKEN_WORKSHED_DESKTOP_STATE_PATH", str(tmp_path / "state.json"))
    return IDEBridge(
        get_models=lambda: ["one", "two"],
        get_active_model=lambda: "one",
        switch_model=lambda model: (True, "switched"),
        get_server_url=lambda: "http://127.0.0.1:8000",
        default_server_url="http://127.0.0.1:8000",
        default_max_tokens=1024,
        default_temperature=0.2,
        agent_runner=_runner,
        agent_capabilities=lambda: {"runtimes": ["hermes"], "skills": ["read"], "toolsets": ["read"]},
    )


def _credential(bridge: IDEBridge) -> str:
    pairing = bridge.create_pairing(
        IDEPairingRequest(
            client_id="jetbrains-v2-client",
            client_name="Token Workshed for JetBrains",
            ide_name="PyCharm",
            plugin_version="0.2.0",
            protocol_version=IDE_AGENT_PROTOCOL_VERSION,
            requested_scopes=list(IDE_AGENT_SCOPES),
        )
    )
    bridge.decide_pairing(pairing["pairing_id"], approved=True, grant_all_requested=True)
    return bridge.pairing_status(pairing["pairing_id"])["credential"]


def test_v2_pairing_and_session_crud(tmp_path, monkeypatch):
    bridge = _bridge(tmp_path, monkeypatch)
    token = _credential(bridge)
    app = FastAPI()
    register_ide_bridge_routes(app, bridge)
    headers = {"Authorization": f"Bearer {token}"}

    with TestClient(app) as client:
        status = client.get("/api/ide/v2/status")
        assert status.status_code == 200
        assert status.json()["agent_access"] is True

        created = client.post(
            "/api/ide/v2/sessions",
            json={"title": "IDE work", "model": "one"},
            headers=headers,
        )
        assert created.status_code == 200
        session_id = created.json()["session"]["id"]
        assert session_id.startswith("ide_")

        listed = client.get("/api/ide/v2/sessions", headers=headers)
        assert listed.status_code == 200
        assert listed.json()["data"][0]["id"] == session_id

        renamed = client.patch(
            f"/api/ide/v2/sessions/{session_id}",
            json={"title": "Renamed"},
            headers=headers,
        )
        assert renamed.status_code == 200
        assert renamed.json()["session"]["title"] == "Renamed"

        forked = client.post(
            f"/api/ide/v2/sessions/{session_id}/fork",
            json={"title": "Branch"},
            headers=headers,
        )
        assert forked.status_code == 200
        fork_id = forked.json()["session"]["id"]
        assert fork_id != session_id

        deleted = client.delete(f"/api/ide/v2/sessions/{fork_id}", headers=headers)
        assert deleted.status_code == 200
        assert deleted.json()["deleted"] is True


def test_v2_stream_waits_for_approval_and_persists_reply(tmp_path, monkeypatch):
    bridge = _bridge(tmp_path, monkeypatch)
    token = _credential(bridge)
    record = bridge.authorize(
        SimpleNamespace(headers={"authorization": f"Bearer {token}"}),
        "ide.sessions.write",
    )
    created = bridge.create_session(record, type("Payload", (), {"title": "Chat", "model": "one"})())
    session_id = created["session"]["id"]
    payload = type(
        "ChatPayload",
        (),
        {"request_id": "request-123", "message": "hello", "model": "one", "runtime": "hermes", "mode": "ask", "context": []},
    )()
    events: list[str] = []

    def consume() -> None:
        events.extend(bridge.stream_session_chat(record, session_id, payload))

    worker = threading.Thread(target=consume, daemon=True)
    worker.start()
    deadline = time.time() + 2
    while time.time() < deadline and not bridge._active_runs:
        time.sleep(0.01)
    run_id = next(iter(bridge._active_runs))
    bridge.approve_run(record, run_id, type("Approval", (), {"choice": "once"})())
    worker.join(2)
    assert not worker.is_alive()
    assert any('"type":"approval_request"' in event for event in events)
    assert any('"type":"done"' in event and '"ok":true' in event for event in events)
    messages = bridge.get_session_messages(record, session_id)["data"]
    assert [item["role"] for item in messages] == ["user", "assistant"]


def test_app_owned_agent_skips_ide_approval_and_returns_non_persisting_patch(
    tmp_path, monkeypatch
):
    bridge = _bridge(tmp_path, monkeypatch)
    app_token = "app-owned-acp-token"
    bridge.register_internal_authorization(client_id="token-workshed-acp", token=app_token)
    record = bridge.authorize(
        SimpleNamespace(headers={"authorization": f"Bearer {app_token}"}),
        "ide.sessions.write",
    )
    created = bridge.create_session(
        record,
        IDESessionCreateRequest(title="App Agent", model="one", cwd=str(tmp_path)),
    )
    session_id = created["session"]["id"]
    original = "print('old')\n"
    base_hash = hashlib.sha256(original.encode("utf-8")).hexdigest()
    bridge._agent_runner = lambda **_: {
        "reply": (
            "Here is a proposal.\n"
            "<tokenworkshed-artifacts>\n"
            + json.dumps(
                {
                    "patches": [
                        {
                            "path": "src/app.py",
                            "base_sha256": base_hash,
                            "content": "print('new')\n",
                            "create": False,
                        }
                    ],
                    "commands": [],
                }
            )
            + "\n</tokenworkshed-artifacts>"
        ),
        "model": "one",
    }
    payload = IDESessionChatRequest(
        request_id="app-owned-request",
        message="suggest a change",
        model="one",
        runtime="hermes",
        mode="fix",
        context=[IDEContextItem(path="src/app.py", language="python", content=original)],
    )

    events = list(bridge.stream_session_chat(record, session_id, payload))
    decoded = [json.loads(event) for event in events]
    assert any(item["type"] == "approval_policy" for item in decoded)
    assert not any(item["type"] == "approval_request" for item in decoded)
    patch_events = [item for item in decoded if item["type"] == "patch"]
    assert patch_events and patch_events[0]["apply"] is False
    assert patch_events[0]["proposal_only"] is True
    saved = bridge.get_session(record, session_id)["session"]
    assert saved["last_patches"][0]["content"] == "print('new')\n"
    assert not (tmp_path / "src" / "app.py").exists()


def test_sensitive_scopes_require_explicit_all_grant(tmp_path, monkeypatch):
    bridge = _bridge(tmp_path, monkeypatch)
    pairing = bridge.create_pairing(
        IDEPairingRequest(
            client_id="jetbrains-safe-client",
            client_name="JetBrains",
            ide_name="IDE",
            plugin_version="0.2.0",
            protocol_version=IDE_AGENT_PROTOCOL_VERSION,
            requested_scopes=list(IDE_AGENT_SCOPES),
        )
    )
    approved = bridge.decide_pairing(pairing["pairing_id"], approved=True)
    assert "ide.sessions.read_all" not in approved["granted_scopes"]
    assert "ide.runs.approve" not in approved["granted_scopes"]
    assert "ide.sessions.read" in approved["granted_scopes"]


def test_run_approval_is_owned_by_the_pairing_client(tmp_path, monkeypatch):
    bridge = _bridge(tmp_path, monkeypatch)
    first = _credential(bridge)
    second_pairing = bridge.create_pairing(
        IDEPairingRequest(
            client_id="jetbrains-second-client",
            client_name="JetBrains 2",
            ide_name="IDE",
            plugin_version="0.2.0",
            protocol_version=IDE_AGENT_PROTOCOL_VERSION,
            requested_scopes=list(IDE_AGENT_SCOPES),
        )
    )
    bridge.decide_pairing(second_pairing["pairing_id"], approved=True, grant_all_requested=True)
    second = bridge.pairing_status(second_pairing["pairing_id"])["credential"]
    first_record = bridge.authorize(SimpleNamespace(headers={"authorization": f"Bearer {first}"}), "ide.sessions.write")
    second_record = bridge.authorize(SimpleNamespace(headers={"authorization": f"Bearer {second}"}), "ide.runs.approve")
    created = bridge.create_session(first_record, type("Payload", (), {"title": "Chat", "model": "one"})())
    payload = type(
        "ChatPayload",
        (),
        {"request_id": "request-456", "message": "hello", "model": "one", "runtime": "hermes", "mode": "ask", "context": []},
    )()
    worker = threading.Thread(target=lambda: list(bridge.stream_session_chat(first_record, created["session"]["id"], payload)), daemon=True)
    worker.start()
    deadline = time.time() + 2
    while time.time() < deadline and not bridge._active_runs:
        time.sleep(0.01)
    run_id = next(iter(bridge._active_runs))
    try:
        bridge.approve_run(second_record, run_id, type("Approval", (), {"choice": "once"})())
    except Exception as error:
        assert getattr(error, "status_code", None) == 404
    else:
        raise AssertionError("a different IDE client approved the run")
    bridge.stop_run(first_record, run_id)
    worker.join(2)
