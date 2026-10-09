# SPDX-License-Identifier: Apache-2.0
"""MCP stdio gateway owned by Token Workshed.

JetBrains receives only this process command. It never receives the App's
MCP server configuration or credentials. Every list/call is sent back to the
App, where project policy, approval and audit handling lives.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any
from urllib import error, request

from .acp_adapter import _app_url, _token

MCP_PROTOCOL_VERSION = "2025-06-18"


def _call_app(path: str, method: str = "GET", payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None
    headers = {
        "Authorization": f"Bearer {_token()}",
        "Accept": "application/json",
    }
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = request.Request(f"{_app_url()}{path}", data=data, headers=headers, method=method)
    try:
        with request.urlopen(req, timeout=300) as response:
            raw = response.read()
    except (OSError, error.HTTPError) as exc:
        raise RuntimeError(f"Token Workshed MCP gateway unavailable: {exc}") from exc
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Token Workshed MCP gateway returned invalid JSON.") from exc
    if not isinstance(value, dict):
        raise RuntimeError("Token Workshed MCP gateway returned an invalid response.")
    return value


class MCPServer:
    def __init__(self) -> None:
        self._initialized = False

    def send(self, value: dict[str, Any]) -> None:
        sys.stdout.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
        sys.stdout.flush()

    def result(self, request_id: Any, value: dict[str, Any]) -> None:
        self.send({"jsonrpc": "2.0", "id": request_id, "result": value})

    def error(self, request_id: Any, code: int, message: str) -> None:
        self.send({"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": str(message)[:1000]}})

    def handle(self, message: dict[str, Any]) -> None:
        request_id = message.get("id")
        method = str(message.get("method") or "")
        params = message.get("params")
        params = params if isinstance(params, dict) else {}
        if method in {"notifications/initialized", "initialized"}:
            self._initialized = True
            return
        try:
            if method == "initialize":
                requested = str(params.get("protocolVersion") or MCP_PROTOCOL_VERSION)
                self._initialized = True
                self.result(
                    request_id,
                    {
                        "protocolVersion": requested if requested else MCP_PROTOCOL_VERSION,
                        "capabilities": {"tools": {}},
                        "serverInfo": {
                            "name": "token-workshed-mcp",
                            "title": "Token Workshed MCP",
                            "version": "1.0",
                        },
                        "instructions": "Tools, credentials, approvals and audit are owned by Token Workshed App.",
                    },
                )
            elif method == "ping":
                self.result(request_id, {})
            elif method == "tools/list":
                body = _call_app("/api/jetbrains/mcp/tools")
                tools = body.get("tools", []) if isinstance(body, dict) else []
                self.result(request_id, {"tools": tools if isinstance(tools, list) else []})
            elif method == "tools/call":
                name = str(params.get("name") or "").strip()
                arguments = params.get("arguments", {})
                if not name or not isinstance(arguments, dict):
                    self.error(request_id, -32602, "tools/call requires name and object arguments.")
                    return
                body = _call_app(
                    "/api/jetbrains/mcp/call",
                    "POST",
                    {
                        "name": name,
                        "arguments": arguments,
                        "session_id": str(os.environ.get("TOKEN_WORKSHED_SESSION_ID", "") or ""),
                    },
                )
                self.result(request_id, {
                    "content": body.get("content", []),
                    "isError": bool(body.get("isError", body.get("is_error", False))),
                })
            else:
                self.error(request_id, -32601, f"Unsupported MCP method: {method}")
        except Exception as exc:
            self.error(request_id, -32000, str(exc))


def main() -> None:
    server = MCPServer()
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            server.error(None, -32700, "Invalid JSON.")
            continue
        if isinstance(message, dict):
            server.handle(message)


if __name__ == "__main__":
    main()
