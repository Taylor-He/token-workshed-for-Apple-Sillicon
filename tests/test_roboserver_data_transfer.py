from __future__ import annotations

import json
import sys
from types import SimpleNamespace

from fastapi.testclient import TestClient

from token_workshed_nextcore import css_svg_ui
from token_workshed_nextcore import desktop_ui
from token_workshed_nextcore.agent.store import AgentStateStore
from token_workshed_nextcore import server_control
from token_workshed_nextcore.server_control import RemoteTarget, ServerConnectionController


def test_performance_snapshot_reports_measured_cpu_and_unified_memory(monkeypatch):
    total_bytes = 24 * (1024**3)
    used_bytes = 12 * (1024**3)
    monkeypatch.setitem(
        sys.modules,
        "psutil",
        SimpleNamespace(
            cpu_percent=lambda interval: 37.4,
            cpu_count=lambda logical=True: 10,
            virtual_memory=lambda: SimpleNamespace(
                total=total_bytes, used=used_bytes, percent=50.0
            ),
        ),
    )

    metrics = desktop_ui._performance_snapshot()

    assert metrics == {
        "ok": True,
        "cpu_utilization_percent": 37.4,
        "logical_cpu_count": 10,
        "unified_memory_used_gb": 12.0,
        "unified_memory_total_gb": 24.0,
        "unified_memory_percent": 50.0,
    }


def test_transfer_only_exports_selected_user_data_and_scrubs_credentials(tmp_path):
    source = AgentStateStore(tmp_path / "source")
    source.save_memory("Project preference", "Keep the UI compact.", ["design"])
    source.save_skill("Release checklist", "Packaging", "Check the app bundle.")
    source.index_conversation(
        "conversation-1",
        [
            {"role": "user", "content": "Deploy with api_key=super-secret-value"},
            {"role": "assistant", "content": "Use token sk-1234567890abcdefghijklmnop."},
        ],
        "Release discussion",
    )
    with source._connect() as db:
        db.execute(
            "INSERT INTO mcp_servers(id,name,transport,updated_at) VALUES(?,?,?,?)",
            ("mcp-1", "Private MCP", "stdio", 1.0),
        )
        db.execute(
            "INSERT INTO browser_profile_grants(id,pid,window_id,app_name,created_at) VALUES(?,?,?,?,?)",
            ("browser-1", 1, 1, "Browser", 1.0),
        )

    transfer = source.export_transfer("local")
    serialized = json.dumps(transfer)

    assert transfer["counts"] == {"memories": 1, "skills": 1, "chat_history": 1}
    assert "Private MCP" not in serialized
    assert "browser_profile_grants" not in serialized
    assert "super-secret-value" not in serialized
    assert "1234567890abcdefghijklmnop" not in serialized
    assert "[redacted credential]" in serialized


def test_transfer_preview_and_import_merge_without_overwriting_conflicts(tmp_path):
    source = AgentStateStore(tmp_path / "source")
    source.save_memory("Shared note", "Source body.")
    source.save_skill("Shared skill", "A description", "Source instructions.")
    source.index_conversation("source-chat", [{"role": "user", "content": "A searchable phrase."}], "Search test")
    transfer = source.export_transfer("local")

    destination = AgentStateStore(tmp_path / "destination")
    destination.save_memory("Shared note", "Destination body.")
    preview = destination.preview_transfer_import(
        transfer, ["memories", "skills", "chat_history"]
    )

    assert preview["categories"]["memories"]["conflicts"] == 1
    assert preview["categories"]["skills"]["importable"] == 1
    assert preview["categories"]["chat_history"]["importable"] == 1
    assert len(destination.list_skills()) == 0

    result = destination.import_transfer(
        transfer, ["memories", "skills", "chat_history"]
    )
    assert result["imported"] == {"memories": 0, "skills": 1, "chat_history": 1}
    assert result["skipped"]["memories"]["conflicts"] == 1
    assert destination.list_memories()[0]["body"] == "Destination body."
    assert len(destination.search_conversations("searchable phrase")) == 1

    repeated = destination.import_transfer(
        transfer, ["memories", "skills", "chat_history"]
    )
    assert repeated["imported"] == {"memories": 0, "skills": 0, "chat_history": 0}
    assert repeated["skipped"]["skills"]["duplicates"] == 1
    assert repeated["skipped"]["chat_history"]["duplicates"] == 1


def test_transfer_rejects_unknown_categories_and_oversized_or_invalid_formats(tmp_path):
    store = AgentStateStore(tmp_path / "store")
    valid = store.export_transfer("local")

    try:
        store.preview_transfer_import(valid, ["browser_profiles"])
    except ValueError as exc:
        assert "Unsupported Agent data categories" in str(exc)
    else:
        raise AssertionError("unknown transfer categories should be rejected")

    try:
        store.preview_transfer_import({"format": "unknown", "categories": {}}, ["memories"])
    except ValueError as exc:
        assert "Unsupported Agent data transfer format" in str(exc)
    else:
        raise AssertionError("unsupported transfer formats should be rejected")

    oversized = {
        "format": "token-workshed-agent-transfer-v1",
        "categories": {"memories": [{"body": "x" * (16 * 1024 * 1024)}]},
    }
    try:
        store.preview_transfer_import(oversized, ["memories"])
    except ValueError as exc:
        assert "16 MiB" in str(exc)
    else:
        raise AssertionError("oversized transfer packages should be rejected")


def test_transfer_api_requires_the_nextcore_app_session(tmp_path, monkeypatch):
    monkeypatch.setenv("TOKEN_WORKSHED_AGENT_DATA_DIR", str(tmp_path / "app-agent-data"))
    app = css_svg_ui.create_app(
        "http://127.0.0.1:8010", 256, 0.7, warm_agent_on_startup=False
    )
    app.state.agent_ui_token = "native-window-session"
    store = AgentStateStore(tmp_path / "agent-data")
    store.save_memory("Transfer test", "Private local note.")
    app.state.agent_store = store

    with TestClient(app) as client:
        denied = client.post(
            "/api/agent/data/export", json={"categories": ["memories"]}
        )
        exported = client.post(
            "/api/agent/data/export",
            headers={"X-Token-Workshed-App-Session": "native-window-session"},
            json={"categories": ["memories"]},
        )

    assert denied.status_code == 401
    assert exported.status_code == 200
    transfer = exported.json()
    assert transfer["counts"] == {"memories": 1, "skills": 0, "chat_history": 0}
    assert transfer["categories"]["skills"] == []
    assert transfer["categories"]["chat_history"] == []


def test_remote_agent_never_misdirects_local_mac_tool_requests_to_server(tmp_path, monkeypatch):
    monkeypatch.setenv("TOKEN_WORKSHED_AGENT_DATA_DIR", str(tmp_path / "roboserver-app-agent-data"))
    app = css_svg_ui.create_app(
        "http://127.0.0.1:8010", 256, 0.7, warm_agent_on_startup=False
    )
    app.state.agent_ui_token = "native-window-session"
    app.state.roboserver_mode = True
    store = AgentStateStore(tmp_path / "roboserver-agent-data")
    app.state.agent_store = store

    with TestClient(app) as client:
        response = client.post(
            "/api/chat/stream",
            headers={"X-Token-Workshed-App-Session": "native-window-session"},
            json={
                "message": "Take one screenshot of my Mac and tell me if NextCore is open.",
                "model": "test-model",
                "conversation_id": "remote-chat",
            },
        )

    assert response.status_code == 200
    events = [json.loads(line) for line in response.text.splitlines() if line.strip()]
    assert [event["type"] for event in events] == ["start", "tool_status", "answer_delta", "done"]
    assert events[-1]["metrics"]["agent_tools_executed"] == 0
    assert "no action was run" in events[-1]["reply"]
    assert len(store.search_conversations("screenshot")) == 1


def test_remote_restore_reopens_tunnel_without_starting_intentionally_stopped_service(
    monkeypatch,
):
    controller = object.__new__(ServerConnectionController)
    target = {
        "host": "mac-mini.local",
        "username": "workshed",
        "port": 22,
        "app_path": "~/Applications/token-workshed-roboserver.app",
    }
    monkeypatch.setattr(
        controller,
        "_server_config",
        lambda: {"mode": "remote", "target": target, "fingerprint": "SHA256:abcdefghijklmnopqrstuvwxyz0123456789ABCDE"},
    )
    calls = []

    def connect(selected_target, fingerprint, *, ensure_service=True):
        calls.append((selected_target, fingerprint, ensure_service))
        return {"ok": True, "connected": True}

    monkeypatch.setattr(controller, "connect", connect)

    result = controller.restore()

    assert result["connected"] is True
    assert calls == [
        (
            RemoteTarget.from_payload(target),
            "SHA256:abcdefghijklmnopqrstuvwxyz0123456789ABCDE",
            False,
        )
    ]


def test_remote_proxy_keeps_local_agent_data_controls_local(monkeypatch):
    controller = object.__new__(ServerConnectionController)
    monkeypatch.setattr(controller, "status", lambda: {"connected": True})

    assert controller.should_proxy("/api/status") is True
    assert controller.should_proxy("/api/manager/models") is True
    assert controller.should_proxy("/api/manager/agent-data/local-export") is False
    assert controller.should_proxy("/api/agent/data/import") is False
    assert controller.should_proxy("/api/agent/data/import", agent_session_valid=True) is True


def test_local_keep_running_setting_does_not_change_runtime_target(monkeypatch):
    state = {
        "server_connection": {
            "mode": "remote",
            "local_server_enabled": False,
            "target": {"host": "mac-mini.local", "username": "workshed", "port": 22},
        }
    }
    monkeypatch.setattr(
        server_control,
        "mutate_desktop_state",
        lambda update: update(state),
    )
    controller = object.__new__(ServerConnectionController)

    assert controller.set_local_server_enabled(True) is True
    assert state["server_connection"]["mode"] == "remote"
    assert state["server_connection"]["local_server_enabled"] is True
