# SPDX-License-Identifier: Apache-2.0
"""Restricted, local-first bridge for JetBrains IDE clients.

The desktop manager owns the inference runtime. IDE clients receive a scoped
credential and can use the v1 code-assistance surface plus the opt-in v2
session/Agent surface. Manager and Hermes credentials are never exposed.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from . import css_svg_ui
from .agent_profiles import mutate_desktop_state, read_desktop_state
from .version import TOKEN_WORKSHED_VERSION

IDE_PROTOCOL_VERSION = "1"
IDE_AGENT_PROTOCOL_VERSION = "2"
IDE_STATE_KEY = "ide_bridge"
IDE_AUTHORIZATIONS_KEY = "authorizations"
IDE_SESSIONS_KEY = "sessions"
IDE_PAIRING_TTL_SECONDS = 10 * 60
IDE_SCOPES = (
    "ide.status.read",
    "ide.models.read",
    "ide.models.switch",
    "ide.assist.stream",
    "ide.companion.open",
)
IDE_AGENT_SCOPES = (
    "ide.sessions.read",
    "ide.sessions.read_all",
    "ide.sessions.write",
    "ide.sessions.delete",
    "ide.sessions.chat",
    "ide.runs.read",
    "ide.runs.stop",
    "ide.runs.approve",
    "ide.skills.read",
    "ide.toolsets.read",
)
IDE_ALL_SCOPES = IDE_SCOPES + IDE_AGENT_SCOPES
_SENSITIVE_AGENT_SCOPES = frozenset(
    {"ide.sessions.read_all", "ide.sessions.delete", "ide.runs.approve"}
)
_MAX_CONTEXT_ITEMS = 8
_MAX_CONTEXT_CHARS = 420_000
_MAX_MESSAGE_CHARS = 12_000
_MAX_HISTORY_ITEMS = 20
_MAX_PATCH_CHARS = 520_000
_ARTIFACT_OPEN = "<tokenworkshed-artifacts>"
_ARTIFACT_CLOSE = "</tokenworkshed-artifacts>"
_SHELL_META = frozenset(";|&><`$")
_SHELL_EXECUTABLES = frozenset({"bash", "sh", "zsh", "fish", "cmd", "powershell"})


class _StrictPayload(BaseModel):
    """Reject undeclared fields so an IDE request cannot enable Agent paths."""

    class Config:
        extra = "forbid"


class IDEPairingRequest(_StrictPayload):
    client_id: str = Field(min_length=8, max_length=128)
    client_name: str = Field(min_length=1, max_length=120)
    ide_name: str = Field(min_length=1, max_length=120)
    plugin_version: str = Field(min_length=1, max_length=48)
    protocol_version: str = Field(min_length=1, max_length=24)
    requested_scopes: list[str] = Field(default_factory=list, max_length=32)


class IDEPairingDecision(_StrictPayload):
    approved: bool
    # Sensitive Agent scopes are never implicit. The App settings UI can make
    # an explicit "approve all requested" choice after showing the list.
    grant_all_requested: bool = False


class IDEModelSwitchRequest(_StrictPayload):
    model: str = Field(min_length=1, max_length=512)


class IDESessionCreateRequest(_StrictPayload):
    title: str = Field(default="New App Session", max_length=160)
    model: str | None = Field(default=None, max_length=512)
    cwd: str | None = Field(default=None, max_length=4096)


class IDESessionPatchRequest(_StrictPayload):
    title: str | None = Field(default=None, max_length=160)


class IDESessionForkRequest(_StrictPayload):
    title: str | None = Field(default=None, max_length=160)


class IDEApprovalRequest(_StrictPayload):
    choice: Literal["once", "session", "always", "deny"]


class IDEAgentRunnerError(RuntimeError):
    """An Agent run failed without leaking subprocess or credential details."""


class IDEContextItem(_StrictPayload):
    path: str = Field(min_length=1, max_length=4096)
    language: str = Field(default="", max_length=128)
    content: str = Field(default="", max_length=_MAX_CONTEXT_CHARS)
    selection_start_line: int | None = Field(default=None, ge=1)
    selection_end_line: int | None = Field(default=None, ge=1)
    selection_start_offset: int | None = Field(default=None, ge=0)
    selection_end_offset: int | None = Field(default=None, ge=0)
    sensitive_confirmed: bool = False


class IDESessionChatRequest(_StrictPayload):
    request_id: str = Field(min_length=8, max_length=128)
    message: str = Field(min_length=1, max_length=_MAX_MESSAGE_CHARS)
    model: str | None = Field(default=None, max_length=512)
    runtime: Literal["openclaw", "hermes"] = "hermes"
    mode: Literal["ask", "fix", "refactor", "test"] = "ask"
    context: list[IDEContextItem] = Field(default_factory=list, max_length=_MAX_CONTEXT_ITEMS)


class IDEHistoryMessage(_StrictPayload):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=48_000)


class IDEAssistRequest(_StrictPayload):
    message: str = Field(min_length=1, max_length=_MAX_MESSAGE_CHARS)
    mode: Literal["ask", "fix", "refactor", "test"] = "ask"
    model: str | None = Field(default=None, max_length=512)
    history: list[IDEHistoryMessage] = Field(default_factory=list, max_length=_MAX_HISTORY_ITEMS)
    context: list[IDEContextItem] = Field(default_factory=list, max_length=_MAX_CONTEXT_ITEMS)
    max_tokens: int = Field(default=2048, ge=128, le=8192)
    temperature: float = Field(default=0.2, ge=0.0, le=1.0)


class IDEBridgeError(RuntimeError):
    """Error with the HTTP status intended for the narrow IDE surface."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message


def _json_line(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _clean_string(value: str, *, limit: int) -> str:
    return " ".join(str(value or "").strip().split())[:limit]


def _safe_relative_path(value: str) -> str:
    normalized = str(value or "").strip().replace("\\", "/")
    if not normalized or normalized.startswith("/") or normalized.startswith("~"):
        raise IDEBridgeError(400, "IDE paths must be project-relative.")
    path = PurePosixPath(normalized)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise IDEBridgeError(400, "IDE paths must stay inside the project.")
    if path.parts and path.parts[0] == ".git":
        raise IDEBridgeError(400, "Git metadata cannot be sent to the IDE bridge.")
    return path.as_posix()


def _looks_sensitive(path: str) -> bool:
    name = PurePosixPath(path).name.lower()
    lower = path.lower()
    if name in {".env", "id_rsa", "id_ed25519", ".npmrc", ".pypirc"}:
        return True
    if name.startswith(".env.") or name.endswith((".pem", ".key", ".p12", ".pfx")):
        return True
    return any(marker in lower for marker in ("credential", "secret", "private_key"))


def _is_test_path(path: str) -> bool:
    lower = path.lower()
    name = PurePosixPath(path).name.lower()
    return (
        "/test/" in lower
        or "/tests/" in lower
        or name.startswith("test_")
        or name.endswith(("_test.py", "test.py", ".test.ts", ".test.tsx", ".spec.ts", ".spec.tsx"))
    )


def _is_direct_command(argv: list[str]) -> bool:
    if not argv or len(argv) > 32:
        return False
    executable = argv[0].strip().lower()
    if not executable or executable in _SHELL_EXECUTABLES:
        return False
    return all(
        bool(item.strip())
        and len(item) <= 2048
        and not any(character in item for character in _SHELL_META)
        for item in argv
    )


class _ArtifactBoundary:
    """Incrementally hide the structured artifact section from chat text."""

    def __init__(self) -> None:
        self._visible_buffer = ""
        self._artifact_buffer = ""
        self._inside_artifact = False

    def feed(self, value: str) -> str:
        if not value:
            return ""
        if self._inside_artifact:
            self._artifact_buffer += value
            return ""
        self._visible_buffer += value
        marker_index = self._visible_buffer.find(_ARTIFACT_OPEN)
        if marker_index >= 0:
            visible = self._visible_buffer[:marker_index]
            self._artifact_buffer += self._visible_buffer[
                marker_index + len(_ARTIFACT_OPEN) :
            ]
            self._visible_buffer = ""
            self._inside_artifact = True
            return visible
        # Retain a small tail to avoid rendering a partial marker token.
        safe_length = max(0, len(self._visible_buffer) - len(_ARTIFACT_OPEN) + 1)
        if safe_length == 0:
            return ""
        visible = self._visible_buffer[:safe_length]
        self._visible_buffer = self._visible_buffer[safe_length:]
        return visible

    def finish(self) -> tuple[str, dict[str, Any] | None, str | None]:
        if not self._inside_artifact:
            return self._visible_buffer, None, None
        artifact_text = self._artifact_buffer
        close_index = artifact_text.find(_ARTIFACT_CLOSE)
        if close_index >= 0:
            artifact_text = artifact_text[:close_index]
        artifact_text = artifact_text.strip()
        if not artifact_text:
            return "", None, "The model returned an empty change proposal."
        try:
            parsed = json.loads(artifact_text)
        except json.JSONDecodeError:
            return "", None, "The model returned an invalid structured change proposal."
        if not isinstance(parsed, dict):
            return "", None, "The model returned an invalid structured change proposal."
        return "", parsed, None


class IDEBridge:
    """Own pairing, scoped authorization, and IDE session/Agent calls.

    The optional ``agent_runner`` is supplied by the desktop launcher and is
    intentionally a narrow callback. It receives a validated session request
    and returns a safe reply object; it never receives the IDE credential.
    """

    def __init__(
        self,
        *,
        get_models: Callable[[], list[str]],
        get_active_model: Callable[[], str | None],
        switch_model: Callable[[str], tuple[bool, str]],
        get_server_url: Callable[[], str | None],
        default_server_url: str,
        default_max_tokens: int,
        default_temperature: float,
        launch_companion: Callable[[], bool] | None = None,
        agent_runner: Callable[..., dict[str, Any]] | None = None,
        agent_capabilities: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        self._get_models = get_models
        self._get_active_model = get_active_model
        self._switch_model = switch_model
        self._get_server_url = get_server_url
        self._default_server_url = default_server_url.rstrip("/")
        self._default_max_tokens = max(128, min(8192, int(default_max_tokens)))
        self._default_temperature = max(0.0, min(1.0, float(default_temperature)))
        self._launch_companion = launch_companion
        self._agent_runner = agent_runner
        self._agent_capabilities = agent_capabilities
        self._lock = threading.RLock()
        self._pending: dict[str, dict[str, Any]] = {}
        self._active_runs: dict[str, dict[str, Any]] = {}

    def status(self) -> dict[str, Any]:
        agent_ready = self._agent_runner is not None
        return {
            "ok": True,
            "protocol_version": IDE_PROTOCOL_VERSION,
            "supported_protocol_versions": [IDE_PROTOCOL_VERSION, IDE_AGENT_PROTOCOL_VERSION],
            "token_workshed_version": TOKEN_WORKSHED_VERSION,
            "pairing_required": True,
            "capabilities": list(IDE_SCOPES),
            "agent_capabilities": list(IDE_AGENT_SCOPES) if agent_ready else [],
            "agent_access": agent_ready,
            "session_sync": agent_ready,
            "terminal_access": False,
        }

    def _expire_pending_locked(self) -> None:
        cutoff = time.time() - IDE_PAIRING_TTL_SECONDS
        stale = [
            pairing_id
            for pairing_id, item in self._pending.items()
            if float(item.get("created_at", 0.0) or 0.0) < cutoff
        ]
        for pairing_id in stale:
            self._pending.pop(pairing_id, None)

    def create_pairing(self, payload: IDEPairingRequest) -> dict[str, Any]:
        client_id = _clean_string(payload.client_id, limit=128)
        if not client_id:
            raise IDEBridgeError(400, "IDE client id cannot be empty.")
        client_name = _clean_string(payload.client_name, limit=120)
        ide_name = _clean_string(payload.ide_name, limit=120)
        plugin_version = _clean_string(payload.plugin_version, limit=48)
        protocol_version = _clean_string(payload.protocol_version, limit=24)
        if not client_name or not ide_name or not plugin_version or not protocol_version:
            raise IDEBridgeError(400, "IDE pairing details cannot be empty.")
        if protocol_version not in {IDE_PROTOCOL_VERSION, IDE_AGENT_PROTOCOL_VERSION}:
            raise IDEBridgeError(
                409,
                "Incompatible IDE bridge protocol: "
                f"desktop supports {IDE_PROTOCOL_VERSION}/{IDE_AGENT_PROTOCOL_VERSION}, "
                f"plugin uses {protocol_version}.",
            )
        requested_scopes = [
            str(scope).strip()
            for scope in payload.requested_scopes
            if str(scope).strip()
        ]
        unknown_scopes = sorted(set(requested_scopes) - set(IDE_ALL_SCOPES))
        if unknown_scopes:
            raise IDEBridgeError(
                400,
                "Unsupported IDE bridge scope(s): " + ", ".join(unknown_scopes),
            )
        if protocol_version == IDE_PROTOCOL_VERSION and any(
            scope in IDE_AGENT_SCOPES for scope in requested_scopes
        ):
            raise IDEBridgeError(
                409,
                "Agent/session scopes require Bridge protocol_version=2.",
            )
        if protocol_version == IDE_AGENT_PROTOCOL_VERSION and self._agent_runner is None:
            raise IDEBridgeError(503, "The desktop Agent runtime is unavailable.")
        now = time.time()
        with self._lock:
            self._expire_pending_locked()
            pairing_id = uuid.uuid4().hex
            self._pending[pairing_id] = {
                "pairing_id": pairing_id,
                "client_id": client_id,
                "client_name": client_name,
                "ide_name": ide_name,
                "plugin_version": plugin_version,
                "protocol_version": protocol_version,
                "created_at": now,
                "status": "pending",
                "credential": "",
                "requested_scopes": list(dict.fromkeys(requested_scopes)),
            }
        return {
            "ok": True,
            "pairing_id": pairing_id,
            "status": "pending",
            "expires_in_seconds": IDE_PAIRING_TTL_SECONDS,
        }

    def pending_pairings(self) -> list[dict[str, Any]]:
        with self._lock:
            self._expire_pending_locked()
            pending = [
                item
                for item in self._pending.values()
                if str(item.get("status")) == "pending"
            ]
            pending.sort(key=lambda item: float(item.get("created_at", 0.0)))
            return [self._public_pairing(item) for item in pending]

    def pairing_status(self, pairing_id: str) -> dict[str, Any]:
        with self._lock:
            self._expire_pending_locked()
            item = self._pending.get(pairing_id)
            if item is None:
                raise IDEBridgeError(404, "Pairing request was not found or expired.")
            status = str(item.get("status") or "pending")
            result = {"ok": True, "pairing_id": pairing_id, "status": status}
            if status == "approved":
                credential = str(item.get("credential") or "")
                if credential:
                    result["credential"] = credential
                    item["credential"] = ""
                self._pending.pop(pairing_id, None)
            return result

    def decide_pairing(
        self,
        pairing_id: str,
        approved: bool,
        grant_all_requested: bool = False,
    ) -> dict[str, Any]:
        with self._lock:
            self._expire_pending_locked()
            item = self._pending.get(pairing_id)
            if item is None:
                raise IDEBridgeError(404, "Pairing request was not found or expired.")
            if not approved:
                item["status"] = "rejected"
                return self._public_pairing(item)

            credential = secrets.token_urlsafe(32)
            requested_scopes = [
                scope
                for scope in item.get("requested_scopes", [])
                if scope in IDE_AGENT_SCOPES
            ]
            granted_agent_scopes = (
                requested_scopes
                if grant_all_requested
                else [scope for scope in requested_scopes if scope not in _SENSITIVE_AGENT_SCOPES]
            )
            record = {
                "client_id": item["client_id"],
                "client_name": item["client_name"],
                "ide_name": item["ide_name"],
                "plugin_version": item["plugin_version"],
                "protocol_version": item["protocol_version"],
                "token_hash": _sha256(credential),
                "scopes": list(
                    dict.fromkeys(
                        list(IDE_SCOPES)
                        + [
                            scope for scope in granted_agent_scopes
                        ]
                    )
                ),
                "requested_scopes": list(item.get("requested_scopes", [])),
                "granted_scopes": list(granted_agent_scopes),
                "approved_at": int(time.time()),
            }

            def store(state: dict[str, Any]) -> bool:
                bridge_state = state.get(IDE_STATE_KEY)
                if not isinstance(bridge_state, dict):
                    bridge_state = {}
                authorizations = bridge_state.get(IDE_AUTHORIZATIONS_KEY)
                if not isinstance(authorizations, dict):
                    authorizations = {}
                authorizations[str(record["client_id"])] = record
                bridge_state[IDE_AUTHORIZATIONS_KEY] = authorizations
                state[IDE_STATE_KEY] = bridge_state
                return True

            mutate_desktop_state(store)
            item["status"] = "approved"
            item["credential"] = credential
            item["granted_scopes"] = list(granted_agent_scopes)
            return self._public_pairing(item)

    def authorizations(self) -> list[dict[str, Any]]:
        state = read_desktop_state()
        bridge_state = state.get(IDE_STATE_KEY)
        records = (
            bridge_state.get(IDE_AUTHORIZATIONS_KEY)
            if isinstance(bridge_state, dict)
            else {}
        )
        if not isinstance(records, dict):
            return []
        output: list[dict[str, Any]] = []
        for item in records.values():
            if isinstance(item, dict):
                output.append(self._public_authorization(item))
        return sorted(output, key=lambda item: int(item.get("approved_at", 0) or 0), reverse=True)

    def revoke_authorization(self, client_id: str) -> bool:
        normalized = _clean_string(client_id, limit=128)
        if not normalized:
            raise IDEBridgeError(400, "IDE client id cannot be empty.")

        def remove(state: dict[str, Any]) -> bool:
            bridge_state = state.get(IDE_STATE_KEY)
            if not isinstance(bridge_state, dict):
                return False
            authorizations = bridge_state.get(IDE_AUTHORIZATIONS_KEY)
            if not isinstance(authorizations, dict) or normalized not in authorizations:
                return False
            authorizations.pop(normalized, None)
            bridge_state[IDE_AUTHORIZATIONS_KEY] = authorizations
            state[IDE_STATE_KEY] = bridge_state
            return True

        return mutate_desktop_state(remove)

    def authorize(self, request: Request, required_scope: str) -> dict[str, Any]:
        raw_header = str(request.headers.get("authorization", "") or "").strip()
        prefix = "bearer "
        if not raw_header.lower().startswith(prefix):
            raise IDEBridgeError(401, "Missing IDE authorization credential.")
        token = raw_header[len(prefix) :].strip()
        if not token:
            raise IDEBridgeError(401, "Missing IDE authorization credential.")
        token_hash = _sha256(token)
        state = read_desktop_state()
        bridge_state = state.get(IDE_STATE_KEY)
        records = (
            bridge_state.get(IDE_AUTHORIZATIONS_KEY)
            if isinstance(bridge_state, dict)
            else {}
        )
        if not isinstance(records, dict):
            raise IDEBridgeError(401, "IDE credential is not authorized.")
        for record in records.values():
            if not isinstance(record, dict):
                continue
            stored_hash = str(record.get("token_hash") or "")
            scopes = record.get("scopes")
            if not isinstance(scopes, list):
                continue
            if secrets.compare_digest(stored_hash, token_hash):
                if required_scope not in scopes:
                    raise IDEBridgeError(403, "IDE credential does not have this capability.")
                return record
        raise IDEBridgeError(401, "IDE credential is not authorized.")

    def register_internal_authorization(
        self,
        *,
        client_id: str,
        token: str,
        client_name: str = "Token Workshed ACP",
        ide_name: str = "JetBrains AI Assistant",
    ) -> dict[str, Any]:
        """Register an App-owned credential for the local ACP/MCP adapters.

        This credential is generated and consumed by the App. It is not
        returned by a pairing endpoint and does not grant manager access.
        """
        normalized_id = _clean_string(client_id, limit=128)
        normalized_token = str(token or "").strip()
        if not normalized_id or not normalized_token:
            raise IDEBridgeError(400, "Internal IDE authorization is incomplete.")
        scopes = list(IDE_SCOPES) + [
            "ide.sessions.read",
            "ide.sessions.write",
            "ide.sessions.chat",
            "ide.runs.read",
            "ide.runs.stop",
            "ide.skills.read",
            "ide.toolsets.read",
        ]
        record = {
            "client_id": normalized_id,
            "client_name": client_name,
            "ide_name": ide_name,
            "plugin_version": "app-owned",
            "protocol_version": IDE_AGENT_PROTOCOL_VERSION,
            "token_hash": _sha256(normalized_token),
            "scopes": scopes,
            "requested_scopes": scopes,
            "granted_scopes": scopes,
            "approved_at": int(time.time()),
            "app_owned": True,
        }

        def store(state: dict[str, Any]) -> bool:
            bridge_state = state.get(IDE_STATE_KEY)
            if not isinstance(bridge_state, dict):
                bridge_state = {}
            authorizations = bridge_state.get(IDE_AUTHORIZATIONS_KEY)
            if not isinstance(authorizations, dict):
                authorizations = {}
            previous = authorizations.get(normalized_id)
            authorizations[normalized_id] = record
            bridge_state[IDE_AUTHORIZATIONS_KEY] = authorizations
            state[IDE_STATE_KEY] = bridge_state
            return previous != record

        mutate_desktop_state(store)
        return self._public_authorization(record)

    def models(self) -> dict[str, Any]:
        try:
            available = [str(model).strip() for model in self._get_models()]
        except Exception as exc:
            raise IDEBridgeError(503, f"Could not list desktop models: {exc}") from exc
        models = list(dict.fromkeys(model for model in available if model))
        active = str(self._get_active_model() or "").strip()
        if active and active not in models:
            models.insert(0, active)
        return {"ok": True, "models": models, "active_model": active}

    def set_active_model(self, model: str) -> dict[str, Any]:
        normalized = _clean_string(model, limit=512)
        if not normalized:
            raise IDEBridgeError(400, "Model name cannot be empty.")
        ok, message = self._switch_model(normalized)
        if not ok:
            raise IDEBridgeError(409, str(message or "Model switch failed."))
        return {
            "ok": True,
            "message": str(message or "Model switched."),
            "active_model": str(self._get_active_model() or normalized).strip(),
        }

    def open_companion(self) -> dict[str, Any]:
        """Open the separately hosted Rust/libcosmic IDE workbench."""
        if self._launch_companion is None:
            raise IDEBridgeError(503, "The desktop IDE Companion is unavailable.")
        try:
            launched = bool(self._launch_companion())
        except Exception as exc:
            raise IDEBridgeError(503, f"Could not open the IDE Companion: {exc}") from exc
        if not launched:
            raise IDEBridgeError(503, "The desktop IDE Companion did not start.")
        return {"ok": True, "opened": True}

    # ------------------------------------------------------------------
    # Bridge v2: App-owned sessions and Agent runs
    # ------------------------------------------------------------------

    def agent_status(self) -> dict[str, Any]:
        capabilities = {}
        if self._agent_capabilities is not None:
            try:
                capabilities = dict(self._agent_capabilities() or {})
            except Exception:
                capabilities = {}
        return {
            "ok": True,
            "protocol_version": IDE_AGENT_PROTOCOL_VERSION,
            "token_workshed_version": TOKEN_WORKSHED_VERSION,
            "capabilities": list(IDE_AGENT_SCOPES) if self._agent_runner else [],
            "agent_access": bool(self._agent_runner),
            "terminal_access": False,
            "runtime": capabilities,
        }

    @staticmethod
    def _safe_title(value: Any, fallback: str = "New App Session") -> str:
        text = " ".join(str(value or "").strip().split())
        return text[:160] or fallback

    @staticmethod
    def _safe_session_id(value: Any) -> str:
        raw = str(value or "").strip()
        if not raw or len(raw) > 160 or any(char in raw for char in "\r\n\x00"):
            raise IDEBridgeError(400, "Invalid App session id.")
        if not raw.startswith("ide_"):
            raise IDEBridgeError(400, "IDE sessions must use an IDE-owned id.")
        return raw

    @staticmethod
    def _safe_cwd(value: Any) -> str:
        raw = str(value or "").strip()
        if not raw:
            return ""
        if len(raw) > 4096 or "\x00" in raw or not os.path.isabs(raw):
            raise IDEBridgeError(400, "IDE session cwd must be an absolute local path.")
        return str(Path(raw).resolve(strict=False))

    @staticmethod
    def _session_public(session: dict[str, Any]) -> dict[str, Any]:
        safe = {
            key: session.get(key)
            for key in (
                "id",
                "source",
                "model",
                "title",
                "created_at",
                "updated_at",
                "message_count",
                "parent_session_id",
                "end_reason",
                "cwd",
                "last_patches",
            )
            if key in session
        }
        safe["message_count"] = int(session.get("message_count", 0) or 0)
        return safe

    @staticmethod
    def _message_public(message: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": str(message.get("id") or ""),
            "role": str(message.get("role") or "assistant"),
            "content": str(message.get("content") or ""),
            "timestamp": float(message.get("timestamp", 0.0) or 0.0),
        }

    def _read_sessions(self) -> dict[str, dict[str, Any]]:
        state = read_desktop_state()
        bridge_state = state.get(IDE_STATE_KEY)
        raw = bridge_state.get(IDE_SESSIONS_KEY) if isinstance(bridge_state, dict) else {}
        if not isinstance(raw, dict):
            return {}
        return {str(key): value for key, value in raw.items() if isinstance(value, dict)}

    def _mutate_session(self, session_id: str, mutator: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
        result: dict[str, Any] = {}

        def update(state: dict[str, Any]) -> bool:
            nonlocal result
            bridge_state = state.get(IDE_STATE_KEY)
            if not isinstance(bridge_state, dict):
                bridge_state = {}
            sessions = bridge_state.get(IDE_SESSIONS_KEY)
            if not isinstance(sessions, dict):
                sessions = {}
            current = sessions.get(session_id)
            if not isinstance(current, dict):
                raise IDEBridgeError(404, "App session was not found.")
            mutator(current)
            sessions[session_id] = current
            bridge_state[IDE_SESSIONS_KEY] = sessions
            state[IDE_STATE_KEY] = bridge_state
            result = dict(current)
            return True

        try:
            mutate_desktop_state(update)
        except IDEBridgeError:
            raise
        except Exception as exc:
            raise IDEBridgeError(503, f"Could not update App session: {exc}") from exc
        return result

    def _get_owned_session(self, record: dict[str, Any], session_id: str, *, read_all: bool = False) -> dict[str, Any]:
        session = self._read_sessions().get(session_id)
        if session is None:
            raise IDEBridgeError(404, "App session was not found.")
        owner = str(session.get("client_id") or "")
        if not read_all and owner != str(record.get("client_id") or ""):
            raise IDEBridgeError(403, "This App session belongs to another IDE client.")
        return session

    def list_sessions(self, record: dict[str, Any], *, include_all: bool = False) -> dict[str, Any]:
        sessions = self._read_sessions()
        client_id = str(record.get("client_id") or "")
        values = [
            self._session_public(item)
            for item in sessions.values()
            if include_all or str(item.get("client_id") or "") == client_id
        ]
        values.sort(key=lambda item: float(item.get("updated_at", 0.0) or 0.0), reverse=True)
        return {"object": "list", "data": values[:200], "has_more": len(values) > 200}

    def create_session(self, record: dict[str, Any], payload: IDESessionCreateRequest) -> dict[str, Any]:
        if not self._agent_runner:
            raise IDEBridgeError(503, "The desktop Agent runtime is unavailable.")
        now = time.time()
        client_id = str(record.get("client_id") or "client")[:32]
        session_id = f"ide_{re.sub(r'[^A-Za-z0-9_-]', '_', client_id)}_{uuid.uuid4().hex[:16]}"
        session = {
            "id": session_id,
            "source": "ide_bridge",
            "client_id": str(record.get("client_id") or ""),
            "model": _clean_string(payload.model or "", limit=512),
            "cwd": self._safe_cwd(getattr(payload, "cwd", None)),
            "title": self._safe_title(payload.title),
            "created_at": now,
            "updated_at": now,
            "message_count": 0,
            "parent_session_id": "",
            "end_reason": "",
            "messages": [],
            "last_patches": [],
        }

        def store(state: dict[str, Any]) -> bool:
            bridge_state = state.get(IDE_STATE_KEY)
            if not isinstance(bridge_state, dict):
                bridge_state = {}
            sessions = bridge_state.get(IDE_SESSIONS_KEY)
            if not isinstance(sessions, dict):
                sessions = {}
            sessions[session_id] = session
            bridge_state[IDE_SESSIONS_KEY] = sessions
            state[IDE_STATE_KEY] = bridge_state
            return True

        mutate_desktop_state(store)
        return {"object": "tokenworkshed.session", "session": self._session_public(session)}

    def get_session(self, record: dict[str, Any], session_id: str, *, read_all: bool = False) -> dict[str, Any]:
        session = self._get_owned_session(record, self._safe_session_id(session_id), read_all=read_all)
        return {"object": "tokenworkshed.session", "session": self._session_public(session)}

    def get_session_messages(self, record: dict[str, Any], session_id: str, *, read_all: bool = False) -> dict[str, Any]:
        session = self._get_owned_session(record, self._safe_session_id(session_id), read_all=read_all)
        messages = session.get("messages") if isinstance(session.get("messages"), list) else []
        return {
            "object": "list",
            "session_id": str(session.get("id") or session_id),
            "data": [self._message_public(item) for item in messages if isinstance(item, dict)],
        }

    def patch_session(self, record: dict[str, Any], session_id: str, payload: IDESessionPatchRequest) -> dict[str, Any]:
        normalized = self._safe_session_id(session_id)
        self._get_owned_session(record, normalized)
        changed = payload.title is not None

        def mutate(session: dict[str, Any]) -> None:
            if payload.title is not None:
                session["title"] = self._safe_title(payload.title)
            session["updated_at"] = time.time()

        if not changed:
            raise IDEBridgeError(400, "No supported App session fields were supplied.")
        return {"object": "tokenworkshed.session", "session": self._session_public(self._mutate_session(normalized, mutate))}

    def delete_session(self, record: dict[str, Any], session_id: str) -> dict[str, Any]:
        normalized = self._safe_session_id(session_id)
        self._get_owned_session(record, normalized)

        def remove(state: dict[str, Any]) -> bool:
            bridge_state = state.get(IDE_STATE_KEY)
            sessions = bridge_state.get(IDE_SESSIONS_KEY) if isinstance(bridge_state, dict) else {}
            if not isinstance(sessions, dict) or normalized not in sessions:
                raise IDEBridgeError(404, "App session was not found.")
            sessions.pop(normalized, None)
            return True

        mutate_desktop_state(remove)
        return {"object": "tokenworkshed.session.deleted", "id": normalized, "deleted": True}

    def fork_session(self, record: dict[str, Any], session_id: str, payload: IDESessionForkRequest) -> dict[str, Any]:
        source_id = self._safe_session_id(session_id)
        source = self._get_owned_session(record, source_id)
        create = IDESessionCreateRequest(
            title=payload.title or f"{self._safe_title(source.get('title'), 'App Session')} fork",
            model=source.get("model") or None,
            cwd=source.get("cwd") or None,
        )
        created = self.create_session(record, create)
        fork_id = str(created["session"]["id"])

        def copy_messages(session: dict[str, Any]) -> None:
            messages = source.get("messages") if isinstance(source.get("messages"), list) else []
            session["messages"] = [dict(item) for item in messages if isinstance(item, dict)]
            session["message_count"] = len(session["messages"])
            session["parent_session_id"] = source_id
            session["updated_at"] = time.time()

        fork = self._mutate_session(fork_id, copy_messages)
        return {"object": "tokenworkshed.session", "session": self._session_public(fork)}

    def stream_session_chat(self, record: dict[str, Any], session_id: str, payload: IDESessionChatRequest) -> Iterator[str]:
        normalized_id = self._safe_session_id(session_id)
        session = self._get_owned_session(record, normalized_id)
        if self._agent_runner is None:
            yield _json_line({"type": "error", "message": "The desktop Agent runtime is unavailable."})
            yield _json_line({"type": "done", "ok": False, "session_id": normalized_id})
            return
        prepared_context = self._prepare_context(payload.context)
        run_id = f"run_{uuid.uuid4().hex}"
        message_id = f"msg_{uuid.uuid4().hex}"
        approval_event = threading.Event()
        cancel_event = threading.Event()
        run = {
            "run_id": run_id,
            "session_id": normalized_id,
            "client_id": str(record.get("client_id") or ""),
            "approval_event": approval_event,
            "cancel_event": cancel_event,
            "choice": "",
            "started_at": time.time(),
        }
        with self._lock:
            self._active_runs[run_id] = run

        sequence = 0

        def emit(event_type: str, **fields: Any) -> str:
            nonlocal sequence
            sequence += 1
            return _json_line(
                {
                    "type": event_type,
                    "session_id": normalized_id,
                    "run_id": run_id,
                    "message_id": message_id,
                    "seq": sequence,
                    "ts": time.time(),
                    **fields,
                }
            )

        try:
            yield emit("run_started", request_id=payload.request_id, model=payload.model or session.get("model"))
            yield emit("message_started", role="assistant")
            if bool(record.get("app_owned")):
                # The ACP process is an App-owned child. Its policy is
                # enforced by the App/MCP gateway, so there is no IDE-side
                # approval prompt or second authority to bypass.
                yield emit(
                    "approval_policy",
                    owner="token-workshed-app",
                    automatic_read_only=True,
                    patch_only=True,
                )
            else:
                yield emit(
                    "approval_request",
                    tool_name="App Agent",
                    reason="Allow the App Agent to use its configured tools for this session.",
                    choices=["once", "session", "always", "deny"],
                )
                if not approval_event.wait(5 * 60):
                    raise IDEAgentRunnerError("Agent approval timed out.")
                choice = str(run.get("choice") or "deny")
                if choice == "deny":
                    raise IDEAgentRunnerError("Agent run was denied.")
            if cancel_event.is_set():
                raise IDEAgentRunnerError("Agent run stopped.")
            yield emit("tool_progress", tool_name="Agent", preview="Starting App Agent…")
            messages = session.get("messages") if isinstance(session.get("messages"), list) else []
            history = [self._message_public(item) for item in messages if isinstance(item, dict)][-20:]
            result = self._agent_runner(
                session=session,
                message=payload.message,
                model=payload.model or str(session.get("model") or self._get_active_model() or ""),
                runtime=payload.runtime,
                mode=payload.mode,
                context=prepared_context,
                history=history,
                cancel_event=cancel_event,
            )
            if not isinstance(result, dict):
                raise IDEAgentRunnerError("The App Agent returned an invalid response.")
            raw_reply = str(result.get("reply") or result.get("answer") or "").strip()
            boundary = _ArtifactBoundary()
            visible_reply = boundary.feed(raw_reply)
            tail_reply, artifact_payload, artifact_error = boundary.finish()
            reply = (visible_reply + tail_reply).strip()
            if not reply:
                raise IDEAgentRunnerError("The App Agent returned an empty response.")
            artifacts = self._parse_artifacts(
                artifact_payload,
                payload.mode,
                prepared_context,
            )
            patches = artifacts.get("patches", [])
            for patch in patches:
                yield emit(
                    "patch",
                    patch=patch,
                    apply=False,
                    proposal_only=True,
                )
            if artifact_error:
                yield emit("artifact_warning", message=artifact_error)
            for start in range(0, len(reply), 2048):
                if cancel_event.is_set():
                    raise IDEAgentRunnerError("Agent run stopped.")
                yield emit("assistant_delta", delta=reply[start : start + 2048])
            yield emit("tool_completed", tool_name="Agent", preview="Completed")
            now = time.time()
            user_item = {"id": f"{message_id}_user", "role": "user", "content": payload.message, "timestamp": now}
            assistant_item = {"id": message_id, "role": "assistant", "content": reply, "timestamp": now}

            def append(session_data: dict[str, Any]) -> None:
                current = session_data.get("messages")
                if not isinstance(current, list):
                    current = []
                current.extend([user_item, assistant_item])
                session_data["messages"] = current[-100:]
                session_data["message_count"] = len(session_data["messages"])
                session_data["last_patches"] = patches
                session_data["updated_at"] = now
                if not session_data.get("title") or session_data.get("title") == "New App Session":
                    session_data["title"] = self._safe_title(payload.message, "App Session")

            self._mutate_session(normalized_id, append)
            yield emit("assistant_completed", content=reply)
            yield emit("run_completed", completed=True, metrics=result.get("metrics") or {})
            yield emit("done", ok=True, reply=reply, model=result.get("model") or payload.model)
        except IDEAgentRunnerError as exc:
            yield emit("error", message=str(exc))
            yield emit("done", ok=False)
        except Exception as exc:
            yield emit("error", message=_clean_string(str(exc), limit=500) or "The App Agent failed.")
            yield emit("done", ok=False)
        finally:
            with self._lock:
                self._active_runs.pop(run_id, None)

    def stop_run(self, record: dict[str, Any], run_id: str) -> dict[str, Any]:
        normalized = _clean_string(run_id, limit=128)
        with self._lock:
            run = self._active_runs.get(normalized)
        if not run or str(run.get("client_id") or "") != str(record.get("client_id") or ""):
            raise IDEBridgeError(404, "Agent run was not found.")
        run["cancel_event"].set()
        run["choice"] = "deny"
        run["approval_event"].set()
        return {"ok": True, "run_id": normalized, "stopped": True}

    def approve_run(self, record: dict[str, Any], run_id: str, payload: IDEApprovalRequest) -> dict[str, Any]:
        normalized = _clean_string(run_id, limit=128)
        with self._lock:
            run = self._active_runs.get(normalized)
        if not run or str(run.get("client_id") or "") != str(record.get("client_id") or ""):
            raise IDEBridgeError(404, "Agent run was not found.")
        run["choice"] = payload.choice
        run["approval_event"].set()
        return {"ok": True, "run_id": normalized, "choice": payload.choice}

    def skills(self) -> dict[str, Any]:
        capabilities = self._agent_capabilities() if self._agent_capabilities else {}
        return {"ok": True, "skills": list(capabilities.get("skills") or [])}

    def toolsets(self) -> dict[str, Any]:
        capabilities = self._agent_capabilities() if self._agent_capabilities else {}
        return {"ok": True, "toolsets": list(capabilities.get("toolsets") or [])}

    def stream_assist(self, payload: IDEAssistRequest) -> Iterator[str]:
        prepared_context = self._prepare_context(payload.context)
        selected_model = _clean_string(payload.model or "", limit=512)
        if not selected_model:
            selected_model = str(self._get_active_model() or "").strip()
        if not selected_model:
            yield _json_line(
                {"type": "error", "message": "Choose a local model before asking for code help."}
            )
            return
        server_url = str(self._get_server_url() or self._default_server_url).strip()
        try:
            target = css_svg_ui._normalize_server_url(server_url)
        except ValueError as exc:
            yield _json_line({"type": "error", "message": str(exc)})
            return

        request_id = uuid.uuid4().hex
        system_prompt = self._system_prompt(payload.mode, prepared_context)
        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        for item in payload.history[-_MAX_HISTORY_ITEMS:]:
            messages.append({"role": item.role, "content": item.content.strip()})
        messages.append({"role": "user", "content": payload.message.strip()})
        boundary = _ArtifactBoundary()
        visible_reply = ""
        finish_reason = ""
        prompt_tokens: int | None = None
        completion_tokens: int | None = None
        total_tokens: int | None = None
        started = time.perf_counter()

        yield _json_line(
            {
                "type": "start",
                "request_id": request_id,
                "model": selected_model,
                "mode": payload.mode,
                "agent": False,
            }
        )
        try:
            response, connect_attempts = css_svg_ui._open_stream_chat_completions_with_connect_retry(
                target=target,
                payload={
                    "model": selected_model,
                    "messages": messages,
                    "max_tokens": min(payload.max_tokens, self._default_max_tokens),
                    "temperature": payload.temperature,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                },
                read_timeout_seconds=600,
            )
        except Exception as exc:
            yield _json_line(
                {
                    "type": "error",
                    "message": css_svg_ui._direct_chat_exception_detail(exc),
                }
            )
            return

        with response:
            if not response.ok:
                detail = f"Upstream error: HTTP {response.status_code}"
                try:
                    body = response.text.strip()
                    if body:
                        detail = f"{detail} | {body[:500]}"
                except Exception:
                    pass
                yield _json_line({"type": "error", "message": detail})
                return
            for raw_line in response.iter_lines(chunk_size=1, decode_unicode=True):
                if raw_line is None:
                    continue
                line = raw_line.strip()
                if not line.startswith("data:"):
                    continue
                encoded = line[5:].strip()
                if not encoded or encoded == "[DONE]":
                    continue
                try:
                    chunk = json.loads(encoded)
                except json.JSONDecodeError:
                    continue
                usage = chunk.get("usage") if isinstance(chunk, dict) else None
                if isinstance(usage, dict):
                    prompt_tokens = _as_int(usage.get("prompt_tokens"), prompt_tokens)
                    completion_tokens = _as_int(
                        usage.get("completion_tokens"), completion_tokens
                    )
                    total_tokens = _as_int(usage.get("total_tokens"), total_tokens)
                choices = chunk.get("choices") if isinstance(chunk, dict) else None
                if not isinstance(choices, list) or not choices:
                    continue
                choice = choices[0] if isinstance(choices[0], dict) else {}
                raw_finish = choice.get("finish_reason")
                if isinstance(raw_finish, str) and raw_finish.strip():
                    finish_reason = raw_finish.strip().lower()
                delta = choice.get("delta")
                if not isinstance(delta, dict):
                    continue
                content = delta.get("content")
                if not isinstance(content, str) or not content:
                    continue
                visible = boundary.feed(content)
                if visible:
                    visible_reply += visible
                    yield _json_line({"type": "text_delta", "text": visible})

        final_visible, artifacts, artifact_error = boundary.finish()
        if final_visible:
            visible_reply += final_visible
            yield _json_line({"type": "text_delta", "text": final_visible})
        if artifact_error:
            yield _json_line({"type": "warning", "message": artifact_error})
        parsed = self._parse_artifacts(artifacts, payload.mode, prepared_context)
        for patch in parsed["patches"]:
            yield _json_line({"type": "patch", "patch": patch})
        for command in parsed["commands"]:
            yield _json_line({"type": "command", "command": command})
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
        yield _json_line(
            {
                "type": "done",
                "ok": True,
                "request_id": request_id,
                "model": selected_model,
                "reply": visible_reply.strip(),
                "finish_reason": finish_reason or None,
                "metrics": {
                    "latency_ms": elapsed_ms,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "connect_retry_attempts": int(connect_attempts),
                },
                "artifact_count": len(parsed["patches"]) + len(parsed["commands"]),
            }
        )

    def _prepare_context(self, items: list[IDEContextItem]) -> list[dict[str, Any]]:
        total_chars = 0
        prepared: list[dict[str, Any]] = []
        for item in items:
            path = _safe_relative_path(item.path)
            content = str(item.content or "")
            if not content:
                continue
            total_chars += len(content)
            if total_chars > _MAX_CONTEXT_CHARS:
                raise IDEBridgeError(413, "IDE context exceeds the 420,000 character limit.")
            if _looks_sensitive(path) and not item.sensitive_confirmed:
                raise IDEBridgeError(
                    409,
                    f"{path} may contain a secret. Confirm this attachment before sending it.",
                )
            start_line = item.selection_start_line
            end_line = item.selection_end_line
            if start_line is not None and end_line is not None and end_line < start_line:
                raise IDEBridgeError(400, "Selection end must not be before selection start.")
            start_offset = item.selection_start_offset
            end_offset = item.selection_end_offset
            if (start_offset is None) != (end_offset is None):
                raise IDEBridgeError(400, "Selection offsets must be provided as a pair.")
            if (
                start_offset is not None
                and end_offset is not None
                and end_offset < start_offset
            ):
                raise IDEBridgeError(400, "Selection end must not be before selection start.")
            prepared.append(
                {
                    "path": path,
                    "language": _clean_string(item.language, limit=128),
                    "content": content,
                    "base_sha256": _sha256(content),
                    "selection_start_line": start_line,
                    "selection_end_line": end_line,
                    "selection_start_offset": start_offset,
                    "selection_end_offset": end_offset,
                }
            )
        return prepared

    def _system_prompt(self, mode: str, context: list[dict[str, Any]]) -> str:
        context_text = "\n\n".join(
            "FILE: {path}\nLANGUAGE: {language}\nBASE_SHA256: {base_sha256}\n"
            "SELECTION_OFFSETS: {selection_start_offset}..{selection_end_offset}\n"
            "--- BEGIN FILE ---\n{content}\n--- END FILE ---".format(**item)
            for item in context
        )
        return (
            "You are Token Workshed's IDE code assistant. You are running in a "
            "restricted IDE bridge: do not call tools, do not claim to run commands, "
            "and do not access files beyond the supplied context. Give a concise, "
            f"helpful response for the requested mode: {mode}.\n\n"
            "For a code change, only propose a replacement for explicitly supplied "
            "context. If a context item has selection offsets, content replaces only "
            "that selected range; otherwise it replaces the supplied full file. In test "
            "mode, one new test file is also allowed. Never "
            "propose changes under .git or outside the project.\n\n"
            "Write the user-facing explanation first. Then append exactly one artifact "
            "block using this delimiter and strict JSON; do not put the JSON in a Markdown fence:\n"
            "<tokenworkshed-artifacts>\n"
            '{"patches":[{"path":"relative/path","base_sha256":"hash or empty for new file",'
            '"content":"full replacement content","create":false}],'
            '"commands":[{"argv":["pytest","-q"],"cwd":".","reason":"why"}]}\n'
            "</tokenworkshed-artifacts>\n"
            "Use empty arrays when no proposal is needed. Commands are suggestions only; "
            "use argv, never a shell string.\n\n"
            f"SUPPLIED IDE CONTEXT:\n{context_text or '(none)'}"
        )

    def _parse_artifacts(
        self,
        artifacts: dict[str, Any] | None,
        mode: str,
        context: list[dict[str, Any]],
    ) -> dict[str, list[dict[str, Any]]]:
        if not artifacts:
            return {"patches": [], "commands": []}
        context_by_path = {str(item["path"]): item for item in context}
        patches: list[dict[str, Any]] = []
        raw_patches = artifacts.get("patches")
        if isinstance(raw_patches, list):
            for raw_patch in raw_patches[:4]:
                if not isinstance(raw_patch, dict):
                    continue
                try:
                    path = _safe_relative_path(str(raw_patch.get("path", "")))
                except IDEBridgeError:
                    continue
                create = bool(raw_patch.get("create"))
                content = raw_patch.get("content")
                if not isinstance(content, str) or not content or len(content) > _MAX_PATCH_CHARS:
                    continue
                supplied_hash = str(raw_patch.get("base_sha256", "") or "").strip()
                source = context_by_path.get(path)
                if create:
                    if mode != "test" or not _is_test_path(path) or source is not None:
                        continue
                    base_hash = ""
                else:
                    if source is None:
                        continue
                    base_hash = str(source["base_sha256"])
                    if not secrets.compare_digest(supplied_hash, base_hash):
                        continue
                patch = {
                    "path": path,
                    "base_sha256": base_hash,
                    "content": content,
                    "create": create,
                }
                if source is not None and source.get("selection_start_offset") is not None:
                    patch["selection_start_offset"] = source["selection_start_offset"]
                    patch["selection_end_offset"] = source["selection_end_offset"]
                patches.append(patch)
        commands: list[dict[str, Any]] = []
        raw_commands = artifacts.get("commands")
        if isinstance(raw_commands, list):
            for raw_command in raw_commands[:4]:
                if not isinstance(raw_command, dict):
                    continue
                argv_raw = raw_command.get("argv")
                if not isinstance(argv_raw, list) or not all(
                    isinstance(item, str) for item in argv_raw
                ):
                    continue
                argv = [item.strip() for item in argv_raw]
                if not _is_direct_command(argv):
                    continue
                cwd = str(raw_command.get("cwd", ".") or ".").strip().replace("\\", "/")
                if cwd != ".":
                    try:
                        cwd = _safe_relative_path(cwd)
                    except IDEBridgeError:
                        continue
                commands.append(
                    {
                        "argv": argv,
                        "cwd": cwd,
                        "reason": _clean_string(str(raw_command.get("reason", "")), limit=600),
                    }
                )
        return {"patches": patches, "commands": commands}

    @staticmethod
    def _public_pairing(item: dict[str, Any]) -> dict[str, Any]:
        return {
            "pairing_id": str(item.get("pairing_id") or ""),
            "client_id": str(item.get("client_id") or ""),
            "client_name": str(item.get("client_name") or ""),
            "ide_name": str(item.get("ide_name") or ""),
            "plugin_version": str(item.get("plugin_version") or ""),
            "protocol_version": str(item.get("protocol_version") or ""),
            "created_at": int(item.get("created_at", 0) or 0),
            "status": str(item.get("status") or "pending"),
            "requested_scopes": list(item.get("requested_scopes") or []),
            "granted_scopes": list(item.get("granted_scopes") or []),
        }

    @staticmethod
    def _public_authorization(item: dict[str, Any]) -> dict[str, Any]:
        return {
            "client_id": str(item.get("client_id") or ""),
            "client_name": str(item.get("client_name") or ""),
            "ide_name": str(item.get("ide_name") or ""),
            "plugin_version": str(item.get("plugin_version") or ""),
            "protocol_version": str(item.get("protocol_version") or ""),
            "scopes": list(item.get("scopes") or []),
            "requested_scopes": list(item.get("requested_scopes") or []),
            "granted_scopes": list(item.get("granted_scopes") or []),
            "approved_at": int(item.get("approved_at", 0) or 0),
        }


def _as_int(value: Any, fallback: int | None) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _request_is_loopback(request: Request) -> bool:
    client = request.client
    if client is None:
        return True
    host = str(client.host or "").strip().lower()
    return host in {"127.0.0.1", "::1", "localhost", "testclient"}


def _require_loopback(request: Request) -> None:
    if not _request_is_loopback(request):
        raise HTTPException(status_code=403, detail="IDE pairing is only available on loopback.")


def _authorized(bridge: IDEBridge, request: Request, scope: str) -> dict[str, Any]:
    try:
        return bridge.authorize(request, scope)
    except IDEBridgeError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc


def register_ide_bridge_routes(app: FastAPI, bridge: IDEBridge) -> None:
    """Attach the intentionally small IDE API to the desktop FastAPI app."""

    app.state.ide_bridge = bridge

    @app.get("/api/ide/v1/status")
    async def ide_status(request: Request) -> dict[str, Any]:
        _require_loopback(request)
        return bridge.status()

    @app.post("/api/ide/v1/pairing/requests")
    async def ide_create_pairing(
        request: Request,
        payload: IDEPairingRequest,
    ) -> dict[str, Any]:
        _require_loopback(request)
        try:
            return bridge.create_pairing(payload)
        except IDEBridgeError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc

    @app.get("/api/ide/v1/pairing/status/{pairing_id}")
    async def ide_pairing_status(request: Request, pairing_id: str) -> dict[str, Any]:
        _require_loopback(request)
        try:
            return bridge.pairing_status(pairing_id)
        except IDEBridgeError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc

    @app.get("/api/ide/v1/pairing/pending")
    async def ide_pending_pairings(request: Request) -> dict[str, Any]:
        _require_loopback(request)
        return {"ok": True, "pairings": bridge.pending_pairings()}

    @app.post("/api/ide/v1/pairing/requests/{pairing_id}/decision")
    async def ide_pairing_decision(
        request: Request,
        pairing_id: str,
        payload: IDEPairingDecision,
    ) -> dict[str, Any]:
        _require_loopback(request)
        try:
            pairing = bridge.decide_pairing(
                pairing_id,
                payload.approved,
                grant_all_requested=payload.grant_all_requested,
            )
            return {"ok": True, "pairing": pairing}
        except IDEBridgeError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc

    @app.get("/api/ide/v1/authorizations")
    async def ide_authorizations(request: Request) -> dict[str, Any]:
        _require_loopback(request)
        return {"ok": True, "authorizations": bridge.authorizations()}

    @app.post("/api/ide/v1/authorizations/{client_id}/revoke")
    async def ide_revoke_authorization(request: Request, client_id: str) -> dict[str, Any]:
        _require_loopback(request)
        try:
            revoked = bridge.revoke_authorization(client_id)
            return {"ok": True, "revoked": revoked}
        except IDEBridgeError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc

    @app.get("/api/ide/v1/models")
    async def ide_models(request: Request) -> dict[str, Any]:
        _authorized(bridge, request, "ide.models.read")
        try:
            return await run_in_threadpool(bridge.models)
        except IDEBridgeError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc

    @app.post("/api/ide/v1/models/active")
    async def ide_set_active_model(
        request: Request,
        payload: IDEModelSwitchRequest,
    ) -> dict[str, Any]:
        _authorized(bridge, request, "ide.models.switch")
        try:
            return await run_in_threadpool(bridge.set_active_model, payload.model)
        except IDEBridgeError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc

    @app.post("/api/ide/v1/companion/open")
    async def ide_open_companion(request: Request) -> dict[str, Any]:
        _authorized(bridge, request, "ide.companion.open")
        try:
            return await run_in_threadpool(bridge.open_companion)
        except IDEBridgeError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc

    @app.post("/api/ide/v1/assist/stream")
    async def ide_assist_stream(
        request: Request,
        payload: IDEAssistRequest,
    ) -> StreamingResponse:
        _authorized(bridge, request, "ide.assist.stream")
        try:
            bridge._prepare_context(payload.context)
        except IDEBridgeError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
        return StreamingResponse(
            bridge.stream_assist(payload), media_type="application/x-ndjson"
        )

    # Bridge v2 is additive: v1 clients and their existing credentials remain
    # valid, while a v2 pairing explicitly requests the session/Agent scopes.
    @app.get("/api/ide/v2/status")
    async def ide_v2_status(request: Request) -> dict[str, Any]:
        _require_loopback(request)
        return bridge.agent_status()

    @app.post("/api/ide/v2/pairing/requests")
    async def ide_v2_create_pairing(
        request: Request,
        payload: IDEPairingRequest,
    ) -> dict[str, Any]:
        _require_loopback(request)
        if payload.protocol_version != IDE_AGENT_PROTOCOL_VERSION:
            raise HTTPException(status_code=409, detail="Bridge v2 pairing requires protocol_version=2.")
        try:
            return bridge.create_pairing(payload)
        except IDEBridgeError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc

    @app.get("/api/ide/v2/pairing/status/{pairing_id}")
    async def ide_v2_pairing_status(request: Request, pairing_id: str) -> dict[str, Any]:
        _require_loopback(request)
        try:
            return bridge.pairing_status(pairing_id)
        except IDEBridgeError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc

    def _agent_authorized(request: Request, scope: str) -> dict[str, Any]:
        if not bridge._agent_runner:
            raise HTTPException(status_code=503, detail="The desktop Agent runtime is unavailable.")
        return _authorized(bridge, request, scope)

    @app.get("/api/ide/v2/sessions")
    async def ide_v2_sessions(request: Request) -> dict[str, Any]:
        record = _agent_authorized(request, "ide.sessions.read")
        include_all = str(request.query_params.get("scope") or "").lower() == "all"
        if include_all:
            record = _authorized(bridge, request, "ide.sessions.read_all")
        return await run_in_threadpool(bridge.list_sessions, record, include_all=include_all)

    @app.post("/api/ide/v2/sessions")
    async def ide_v2_create_session(
        request: Request,
        payload: IDESessionCreateRequest,
    ) -> dict[str, Any]:
        record = _agent_authorized(request, "ide.sessions.write")
        try:
            return await run_in_threadpool(bridge.create_session, record, payload)
        except IDEBridgeError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc

    @app.get("/api/ide/v2/sessions/{session_id}")
    async def ide_v2_get_session(request: Request, session_id: str) -> dict[str, Any]:
        record = _agent_authorized(request, "ide.sessions.read")
        include_all = str(request.query_params.get("scope") or "").lower() == "all"
        if include_all:
            record = _authorized(bridge, request, "ide.sessions.read_all")
        try:
            return await run_in_threadpool(bridge.get_session, record, session_id, read_all=include_all)
        except IDEBridgeError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc

    @app.patch("/api/ide/v2/sessions/{session_id}")
    async def ide_v2_patch_session(
        request: Request,
        session_id: str,
        payload: IDESessionPatchRequest,
    ) -> dict[str, Any]:
        record = _agent_authorized(request, "ide.sessions.write")
        try:
            return await run_in_threadpool(bridge.patch_session, record, session_id, payload)
        except IDEBridgeError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc

    @app.delete("/api/ide/v2/sessions/{session_id}")
    async def ide_v2_delete_session(request: Request, session_id: str) -> dict[str, Any]:
        record = _agent_authorized(request, "ide.sessions.delete")
        try:
            return await run_in_threadpool(bridge.delete_session, record, session_id)
        except IDEBridgeError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc

    @app.get("/api/ide/v2/sessions/{session_id}/messages")
    async def ide_v2_session_messages(request: Request, session_id: str) -> dict[str, Any]:
        record = _agent_authorized(request, "ide.sessions.read")
        include_all = str(request.query_params.get("scope") or "").lower() == "all"
        if include_all:
            record = _authorized(bridge, request, "ide.sessions.read_all")
        try:
            return await run_in_threadpool(bridge.get_session_messages, record, session_id, read_all=include_all)
        except IDEBridgeError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc

    @app.post("/api/ide/v2/sessions/{session_id}/fork")
    async def ide_v2_fork_session(
        request: Request,
        session_id: str,
        payload: IDESessionForkRequest,
    ) -> dict[str, Any]:
        record = _agent_authorized(request, "ide.sessions.write")
        try:
            return await run_in_threadpool(bridge.fork_session, record, session_id, payload)
        except IDEBridgeError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc

    @app.post("/api/ide/v2/sessions/{session_id}/chat/stream")
    async def ide_v2_session_chat_stream(
        request: Request,
        session_id: str,
        payload: IDESessionChatRequest,
    ) -> StreamingResponse:
        record = _agent_authorized(request, "ide.sessions.chat")
        try:
            bridge._prepare_context(payload.context)
            bridge._get_owned_session(record, bridge._safe_session_id(session_id))
        except IDEBridgeError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
        return StreamingResponse(
            bridge.stream_session_chat(record, session_id, payload),
            media_type="application/x-ndjson",
        )

    @app.post("/api/ide/v2/runs/{run_id}/stop")
    async def ide_v2_stop_run(request: Request, run_id: str) -> dict[str, Any]:
        record = _agent_authorized(request, "ide.runs.stop")
        try:
            return bridge.stop_run(record, run_id)
        except IDEBridgeError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc

    @app.post("/api/ide/v2/runs/{run_id}/approval")
    async def ide_v2_approve_run(
        request: Request,
        run_id: str,
        payload: IDEApprovalRequest,
    ) -> dict[str, Any]:
        record = _agent_authorized(request, "ide.runs.approve")
        try:
            return bridge.approve_run(record, run_id, payload)
        except IDEBridgeError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc

    @app.get("/api/ide/v2/skills")
    async def ide_v2_skills(request: Request) -> dict[str, Any]:
        _agent_authorized(request, "ide.skills.read")
        return await run_in_threadpool(bridge.skills)

    @app.get("/api/ide/v2/toolsets")
    async def ide_v2_toolsets(request: Request) -> dict[str, Any]:
        _agent_authorized(request, "ide.toolsets.read")
        return await run_in_threadpool(bridge.toolsets)
