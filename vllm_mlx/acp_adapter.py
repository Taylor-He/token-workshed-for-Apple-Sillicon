# SPDX-License-Identifier: Apache-2.0
"""A small stdlib ACP v1 adapter owned by the Token Workshed App.

The adapter keeps stdout strictly JSON-RPC, uses the App's internal bearer
credential, and never advertises filesystem or terminal capabilities.  The
official ``agent-client-protocol`` package is an optional development
dependency; this wire implementation keeps the installed App usable when a
JetBrains installation launches the adapter from a lean environment.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import uuid
from pathlib import Path
from typing import Any
from urllib import error, request

from .jetbrains_integration import acp_token_path, read_discovery_record
from .version import TOKEN_WORKSHED_VERSION

DEFAULT_APP_URL = "http://127.0.0.1:7862"
HTTP_TIMEOUT = 30.0
STREAM_TIMEOUT = 600.0


class ACPError(RuntimeError):
    pass


def _token() -> str:
    value = str(os.environ.get("TOKEN_WORKSHED_ACP_TOKEN", "") or "").strip()
    if value:
        return value
    try:
        value = acp_token_path().read_text(encoding="utf-8").strip()
    except OSError:
        value = ""
    if not value:
        raise ACPError("Token Workshed App ACP credential is unavailable.")
    return value


def _app_url() -> str:
    record = read_discovery_record()
    return str(record.get("ui_url") or DEFAULT_APP_URL).rstrip("/")


def _json_request(
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    *,
    timeout: float = HTTP_TIMEOUT,
) -> dict[str, Any]:
    body = None
    headers = {
        "Authorization": f"Bearer {_token()}",
        "Accept": "application/json",
    }
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = request.Request(f"{_app_url()}{path}", data=body, headers=headers, method=method)
    try:
        with request.urlopen(req, timeout=timeout) as response:
            raw = response.read()
    except (OSError, error.HTTPError) as exc:
        detail = getattr(exc, "reason", None) or str(exc)
        raise ACPError(f"Token Workshed App request failed: {detail}") from exc
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ACPError("Token Workshed App returned invalid JSON.") from exc
    if not isinstance(value, dict):
        raise ACPError("Token Workshed App returned an invalid response.")
    return value


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content or "")
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict):
            block_type = str(block.get("type") or "")
            if block_type == "text":
                parts.append(str(block.get("text") or ""))
            elif block_type in {"resource", "resource_link", "image"}:
                # No filesystem capability is advertised; preserve the fact
                # that a non-text block was supplied without dereferencing it.
                parts.append(f"[{block_type} block omitted by Token Workshed]")
    return "".join(parts)


class ACPServer:
    def __init__(self) -> None:
        self._write_lock = threading.RLock()
        self._active_runs: dict[str, str] = {}
        self._active_lock = threading.RLock()

    def _send(self, value: dict[str, Any]) -> None:
        encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        with self._write_lock:
            sys.stdout.write(encoded + "\n")
            sys.stdout.flush()

    def _response(self, request_id: Any, result: dict[str, Any]) -> None:
        self._send({"jsonrpc": "2.0", "id": request_id, "result": result})

    def _error(self, request_id: Any, code: int, message: str) -> None:
        self._send(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": code, "message": str(message)[:1000]},
            }
        )

    def _update(self, session_id: str, update: dict[str, Any]) -> None:
        self._send(
            {
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {"sessionId": session_id, "update": update},
            }
        )

    def _text_update(self, session_id: str, text: str) -> None:
        if not text:
            return
        self._update(
            session_id,
            {
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": text},
            },
        )

    def handle(self, message: dict[str, Any]) -> None:
        request_id = message.get("id")
        method = str(message.get("method") or "")
        params = message.get("params")
        params = params if isinstance(params, dict) else {}
        if not method:
            if request_id is not None:
                self._error(request_id, -32600, "Invalid ACP request.")
            return
        try:
            if method == "initialize":
                self._initialize(request_id)
            elif method in {"initialized", "notifications/initialized"}:
                return
            elif method == "session/new":
                self._new_session(request_id, params)
            elif method == "session/load":
                self._load_session(request_id, params)
            elif method == "session/prompt":
                threading.Thread(
                    target=self._prompt,
                    args=(request_id, params),
                    daemon=True,
                    name="token-workshed-acp-prompt",
                ).start()
            elif method == "session/cancel":
                self._cancel(request_id, params)
            else:
                if request_id is not None:
                    self._error(request_id, -32601, f"Unsupported ACP method: {method}")
        except ACPError as exc:
            if request_id is not None:
                self._error(request_id, -32000, str(exc))
        except Exception as exc:
            if request_id is not None:
                self._error(request_id, -32603, "Token Workshed ACP request failed.")
            print(f"ACP internal error: {exc}", file=sys.stderr)

    def _initialize(self, request_id: Any) -> None:
        self._response(
            request_id,
            {
                "protocolVersion": 1,
                "agentInfo": {
                    "name": "Token Workshed",
                    "title": "Token Workshed",
                    "version": TOKEN_WORKSHED_VERSION,
                },
                "agentCapabilities": {
                    "loadSession": True,
                    "promptCapabilities": {"image": False, "audio": False},
                    "mcpCapabilities": {"http": False, "sse": False},
                },
            },
        )

    def _new_session(self, request_id: Any, params: dict[str, Any]) -> None:
        cwd = str(params.get("cwd") or Path.cwd().resolve())
        title = str(params.get("title") or "JetBrains AI Assistant")[:160]
        model = str(params.get("model") or "").strip() or None
        # Incoming IDE MCP configuration is intentionally ignored. MCP is
        # App-owned and must be configured/allow-listed inside Token Workshed.
        created = _json_request(
            "POST",
            "/api/ide/v2/sessions",
            {"title": title, "model": model, "cwd": cwd},
        )
        session = created.get("session") if isinstance(created.get("session"), dict) else {}
        session_id = str(session.get("id") or "")
        if not session_id:
            raise ACPError("Token Workshed did not return an ACP session id.")
        self._response(
            request_id,
            {
                "sessionId": session_id,
                "modes": {"currentModeId": "ask", "availableModes": [{"id": "ask", "name": "Ask"}]},
            },
        )

    def _load_session(self, request_id: Any, params: dict[str, Any]) -> None:
        session_id = str(params.get("sessionId") or "").strip()
        if not session_id:
            raise ACPError("session/load requires sessionId.")
        loaded = _json_request("GET", f"/api/ide/v2/sessions/{session_id}")
        session = loaded.get("session") if isinstance(loaded.get("session"), dict) else {}
        if not session:
            raise ACPError("Token Workshed session was not found.")
        messages = _json_request("GET", f"/api/ide/v2/sessions/{session_id}/messages")
        for item in messages.get("data", []) if isinstance(messages.get("data"), list) else []:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or "")
            text = str(item.get("content") or "")
            if role == "assistant":
                self._update(
                    session_id,
                    {
                        "sessionUpdate": "agent_message_chunk",
                        "content": {"type": "text", "text": text},
                    },
                )
            elif role == "user":
                self._update(
                    session_id,
                    {
                        "sessionUpdate": "user_message_chunk",
                        "content": {"type": "text", "text": text},
                    },
                )
        self._response(request_id, {"sessionId": session_id})

    def _prompt(self, request_id: Any, params: dict[str, Any]) -> None:
        session_id = str(params.get("sessionId") or "").strip()
        prompt = _content_text(params.get("prompt") or params.get("content") or "").strip()
        if not session_id or not prompt:
            self._error(request_id, -32602, "session/prompt requires sessionId and text content.")
            return
        request_id_text = str(request_id or uuid.uuid4().hex)
        try:
            self._update(
                session_id,
                {
                    "sessionUpdate": "user_message_chunk",
                    "content": {"type": "text", "text": prompt},
                },
            )
            response = self._stream_chat(session_id, prompt, request_id_text)
            stop_reason = "end_turn" if response else "refusal"
            self._response(request_id, {"stopReason": stop_reason})
        except ACPError as exc:
            self._text_update(session_id, f"\nToken Workshed: {exc}\n")
            self._error(request_id, -32000, str(exc))
        except Exception:
            self._text_update(session_id, "\nToken Workshed ACP request failed.\n")
            self._error(request_id, -32603, "Token Workshed ACP request failed.")
        finally:
            with self._active_lock:
                self._active_runs.pop(session_id, None)

    def _stream_chat(self, session_id: str, prompt: str, request_id: str) -> bool:
        body = {
            "request_id": request_id,
            "message": prompt,
            "runtime": "hermes",
            "mode": "ask",
            "context": [],
        }
        headers = {
            "Authorization": f"Bearer {_token()}",
            "Accept": "application/x-ndjson",
            "Content-Type": "application/json",
        }
        req = request.Request(
            f"{_app_url()}/api/ide/v2/sessions/{session_id}/chat/stream",
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            response = request.urlopen(req, timeout=STREAM_TIMEOUT)
        except (OSError, error.HTTPError) as exc:
            detail = getattr(exc, "reason", None) or str(exc)
            raise ACPError(f"Token Workshed Agent is unavailable: {detail}") from exc
        with response:
            with self._active_lock:
                self._active_runs[session_id] = ""
            succeeded = False
            for raw_line in response:
                try:
                    event = json.loads(raw_line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                if not isinstance(event, dict):
                    continue
                run_id = str(event.get("run_id") or "")
                if run_id:
                    with self._active_lock:
                        self._active_runs[session_id] = run_id
                event_type = str(event.get("type") or "")
                if event_type == "assistant_delta":
                    self._text_update(session_id, str(event.get("delta") or ""))
                elif event_type == "patch":
                    patch = event.get("patch")
                    self._text_update(
                        session_id,
                        "\n[Token Workshed patch proposal; not applied]\n"
                        + json.dumps(patch, ensure_ascii=False)
                        + "\n",
                    )
                elif event_type in {"tool_started", "tool_progress"}:
                    tool_name = str(event.get("tool_name") or "App tool")
                    tool_id = run_id or f"tool_{uuid.uuid4().hex[:12]}"
                    self._update(
                        session_id,
                        {
                            "sessionUpdate": "tool_call",
                            "toolCallId": tool_id,
                            "title": tool_name,
                            "kind": "other",
                            "status": "in_progress",
                        },
                    )
                elif event_type == "tool_completed":
                    self._update(
                        session_id,
                        {
                            "sessionUpdate": "tool_call_update",
                            "toolCallId": run_id or "app-tool",
                            "status": "completed",
                        },
                    )
                elif event_type == "approval_request":
                    self._text_update(session_id, "\n[Token Workshed is waiting for App approval]\n")
                elif event_type == "approval_policy":
                    self._text_update(session_id, "\n[Tool policy: App-owned; file changes are patch-only]\n")
                elif event_type == "error":
                    self._text_update(session_id, f"\nToken Workshed: {event.get('message', 'Agent failed.')}\n")
                elif event_type == "done":
                    succeeded = bool(event.get("ok"))
            return succeeded

    def _cancel(self, request_id: Any, params: dict[str, Any]) -> None:
        session_id = str(params.get("sessionId") or "").strip()
        with self._active_lock:
            run_id = self._active_runs.get(session_id, "")
        if run_id:
            try:
                _json_request("POST", f"/api/ide/v2/runs/{run_id}/stop", {})
            except ACPError as exc:
                print(f"ACP cancel warning: {exc}", file=sys.stderr)
        if request_id is not None:
            self._response(request_id, {"cancelled": True})


def main() -> None:
    server = ACPServer()
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            server._error(None, -32700, "Invalid JSON.")
            continue
        if isinstance(message, dict):
            server.handle(message)


if __name__ == "__main__":
    main()
