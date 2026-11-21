# SPDX-License-Identifier: Apache-2.0
"""Token Workshed's public JetBrains integration surfaces.

This module is deliberately independent of the native UI.  The Rust App can
show the values from ``/api/jetbrains/integration`` while the JetBrains plugin
only consumes the loopback discovery record and the non-secret ACP command.
Provider authentication, project policy, and the Agent runtime stay in this
App-owned process. External MCP clients are given an empty tool list; the
NextCore chat surface is the only entry point for the built-in Agent tools.
"""

from __future__ import annotations

import json
import os
import secrets
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from .desktop_state import desktop_state_file, mutate_desktop_state, read_desktop_state
from .fim import build_fim_prompt, completion_diagnostics, record_completion_diagnostic
from .ide_bridge import IDEBridge, IDEBridgeError

DISCOVERY_VERSION = 1
DISCOVERY_FILENAME = "jetbrains_bridge.json"
ACP_TOKEN_FILENAME = "jetbrains_acp.token"
INTEGRATION_STATE_KEY = "jetbrains_integration"
PROVIDER_KEY_FIELD = "provider_key"
ACP_CLIENT_ID = "token-workshed-nextcore-acp"
ACP_PROTOCOL_VERSION = 1

_CHAT_FIELDS = {
    "model",
    "messages",
    "temperature",
    "top_p",
    "max_tokens",
    "stream",
    "stream_options",
    "stop",
    "tools",
    "tool_choice",
    "chat_template_kwargs",
    "response_format",
    "video_fps",
    "video_max_frames",
    "timeout",
}

_FIM_TOKENIZER_LOCK = threading.RLock()
_FIM_TOKENIZER_CACHE: dict[str, Any | None] = {}
_COMPLETION_FIELDS = {
    "model",
    "prompt",
    "suffix",
    "temperature",
    "top_p",
    "max_tokens",
    "stream",
    "stop",
    "timeout",
}

def _state_dir() -> Path:
    path = desktop_state_file().parent
    path.mkdir(parents=True, exist_ok=True)
    return path


def discovery_path() -> Path:
    return _state_dir() / DISCOVERY_FILENAME


def acp_token_path() -> Path:
    return _state_dir() / ACP_TOKEN_FILENAME


def _atomic_write(path: Path, content: str, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=str(path.parent), delete=False
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    with suppress(OSError):
        os.chmod(temporary, mode)
    temporary.replace(path)
    with suppress(OSError):
        os.chmod(path, mode)


def _valid_loopback_url(value: Any) -> str | None:
    raw = str(value or "").strip().rstrip("/")
    if not raw:
        return None
    try:
        parsed = urlparse(raw)
        port = parsed.port
    except ValueError:
        return None
    if (
        parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        return None
    if port is None or not 1 <= port <= 65535:
        return None
    return raw


def read_discovery_record() -> dict[str, Any]:
    """Read only a current, loopback-only discovery record."""
    path = discovery_path()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    try:
        version = int(raw.get("version", 0) or 0)
        pid = int(raw.get("pid", 0) or 0)
        updated_at = int(raw.get("updated_at", 0) or 0)
    except (TypeError, ValueError, OverflowError):
        return {}
    if version != DISCOVERY_VERSION or pid <= 0:
        return {}
    ui_url = _valid_loopback_url(raw.get("ui_url"))
    provider_url = _valid_loopback_url(raw.get("provider_url"))
    if ui_url is None or provider_url is None:
        return {}
    acp_command = raw.get("acp_command")
    mcp_command = raw.get("mcp_command")
    acp_args = raw.get("acp_args", [])
    mcp_args = raw.get("mcp_args", [])
    if not isinstance(acp_command, str) or not isinstance(mcp_command, str):
        return {}
    if not isinstance(acp_args, list) or not isinstance(mcp_args, list):
        return {}
    if not all(isinstance(item, str) for item in [*acp_args, *mcp_args]):
        return {}
    result = {
        "version": DISCOVERY_VERSION,
        "protocol_version": str(raw.get("protocol_version") or "2"),
        "pid": pid,
        "ui_url": ui_url,
        "provider_url": provider_url,
        "acp_command": acp_command.strip(),
        "acp_args": [item for item in acp_args if item.strip()],
        "mcp_command": mcp_command.strip(),
        "mcp_args": [item for item in mcp_args if item.strip()],
        "updated_at": updated_at,
    }
    if not result["acp_command"] or not result["mcp_command"]:
        return {}
    # Discovery is not a secret, but do not make it a generic process launch
    # channel: commands must be absolute or a Python module invocation.
    for command, args in (
        (result["acp_command"], result["acp_args"]),
        (result["mcp_command"], result["mcp_args"]),
    ):
        if not os.path.isabs(command) and command != sys.executable:
            return {}
        if any("\x00" in item or len(item) > 2048 for item in [command, *args]):
            return {}
    return result


def write_discovery_record(
    *,
    ui_url: str,
    pid: int,
    acp_command: str,
    acp_args: list[str],
    mcp_command: str,
    mcp_args: list[str],
) -> Path:
    """Publish a minimal, key-free App discovery record."""
    safe_ui = _valid_loopback_url(ui_url)
    if safe_ui is None:
        raise ValueError("JetBrains discovery requires a 127.0.0.1 UI URL.")
    payload = {
        "version": DISCOVERY_VERSION,
        "protocol_version": "2",
        "pid": int(pid),
        "ui_url": safe_ui,
        "provider_url": f"{safe_ui}/v1",
        "acp_command": str(acp_command),
        "acp_args": [str(item) for item in acp_args],
        "mcp_command": str(mcp_command),
        "mcp_args": [str(item) for item in mcp_args],
        "updated_at": int(time.time()),
    }
    path = discovery_path()
    _atomic_write(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return path


def remove_discovery_if_pid(pid: int) -> bool:
    path = discovery_path()
    try:
        current = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(current, dict) or int(current.get("pid", 0) or 0) != int(pid):
        return False
    try:
        path.unlink()
    except OSError:
        return False
    return True


def ensure_provider_key() -> str:
    """Create the App-owned limited Provider key once, with mode 0600 state."""
    holder: dict[str, str] = {}

    def store(state: dict[str, Any]) -> bool:
        integration = state.get(INTEGRATION_STATE_KEY)
        if not isinstance(integration, dict):
            integration = {}
        existing = str(integration.get(PROVIDER_KEY_FIELD) or "").strip()
        if existing:
            holder["key"] = existing
            return False
        value = secrets.token_urlsafe(32)
        integration[PROVIDER_KEY_FIELD] = value
        state[INTEGRATION_STATE_KEY] = integration
        holder["key"] = value
        return True

    mutate_desktop_state(store)
    return holder.get("key", "")


def ensure_acp_token() -> str:
    """Create the App-owned ACP/MCP bearer token outside user config files."""
    env_token = str(os.environ.get("TOKEN_WORKSHED_ACP_TOKEN", "") or "").strip()
    if env_token:
        return env_token
    path = acp_token_path()
    try:
        existing = path.read_text(encoding="utf-8").strip()
    except Exception:
        existing = ""
    if existing and len(existing) <= 512:
        return existing
    value = secrets.token_urlsafe(32)
    _atomic_write(path, value + "\n")
    return value


def _current_model(manager: Any) -> str:
    try:
        return str(manager.active_model() or "").strip()
    except Exception:
        return ""


def _completion_records() -> list[dict[str, Any]]:
    state = read_desktop_state().get(INTEGRATION_STATE_KEY)
    if isinstance(state, dict) and isinstance(
        state.get("completion_diagnostics"), list
    ):
        return [
            item for item in state["completion_diagnostics"] if isinstance(item, dict)
        ][-128:]
    return completion_diagnostics()


def _load_fim_tokenizer(model: str, manager: Any) -> Any | None:
    """Load a cached tokenizer only; completion requests never trigger downloads."""
    model_id = str(model or "").strip()
    if not model_id:
        return None
    with _FIM_TOKENIZER_LOCK:
        if model_id in _FIM_TOKENIZER_CACHE:
            return _FIM_TOKENIZER_CACHE[model_id]
        local_ref = model_id
        resolver = getattr(manager, "_resolve_managed_model", None)
        if callable(resolver):
            try:
                local_ref = str(resolver(model_id) or model_id)
            except Exception:
                local_ref = model_id
        tokenizer: Any | None = None
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                local_ref,
                local_files_only=True,
                trust_remote_code=False,
            )
        except Exception:
            tokenizer = None
        _FIM_TOKENIZER_CACHE[model_id] = tokenizer
        return tokenizer


def _prepare_fim_completion(body: dict[str, Any], manager: Any) -> dict[str, Any]:
    """Translate OpenAI ``suffix`` into a model-native FIM prompt for SGLang."""
    if "suffix" not in body or body.get("suffix") is None:
        return body
    prefix = body.get("prompt")
    suffix = body.get("suffix")
    if not isinstance(prefix, str) or not isinstance(suffix, str):
        raise HTTPException(
            status_code=400,
            detail="NextCore FIM completion requires string prompt and suffix fields.",
        )
    model = str(body.get("model") or "").strip()
    fim_prompt = build_fim_prompt(
        prefix,
        suffix,
        _load_fim_tokenizer(model, manager),
    )
    diagnostic = record_completion_diagnostic(
        model,
        suffix_requested=True,
        fim_prompt=fim_prompt,
    )

    def store(state: dict[str, Any]) -> bool:
        integration = state.get(INTEGRATION_STATE_KEY)
        if not isinstance(integration, dict):
            integration = {}
        records = integration.get("completion_diagnostics")
        if not isinstance(records, list):
            records = []
        integration["completion_diagnostics"] = [*records, diagnostic][-128:]
        state[INTEGRATION_STATE_KEY] = integration
        return True

    with suppress(Exception):
        mutate_desktop_state(store)
    prepared = dict(body)
    prepared["prompt"] = fim_prompt.prompt
    prepared.pop("suffix", None)
    return prepared


def _integration_config(manager: Any, *, include_provider_key: bool) -> dict[str, Any]:
    record = read_discovery_record()
    provider = {
        "base_url": str(record.get("provider_url") or "http://127.0.0.1:7872/v1"),
        "model": _current_model(manager),
        "allowed_routes": ["/v1/models", "/v1/chat/completions", "/v1/completions"],
        "tool_calling": True,
        "credential_scope": "models.chat.completions",
    }
    if include_provider_key:
        provider["api_key"] = ensure_provider_key()
    integration_state = read_desktop_state().get(INTEGRATION_STATE_KEY)
    integration_state = integration_state if isinstance(integration_state, dict) else {}
    enabled_projects = integration_state.get("intellij_mcp_projects", [])
    if not isinstance(enabled_projects, list):
        enabled_projects = []
    return {
        "ok": True,
        "protocol_version": "2",
        "provider": provider,
        "completion": {
            "mode": "experimental_reuse_chat_model",
            "suffix": True,
            "diagnostics": _completion_records(),
        },
        "acp": {
            "protocol_version": ACP_PROTOCOL_VERSION,
            "command": record.get("acp_command", ""),
            "args": list(record.get("acp_args", [])),
            "session_persistence": "token-workshed-nextcore-app",
        },
        "mcp": {
            "command": record.get("mcp_command", ""),
            "args": list(record.get("mcp_args", [])),
            "approval_owner": "token-workshed-nextcore-app",
            "audit_owner": "token-workshed-nextcore-app",
        },
        "intellij_mcp_return": {
            "default_enabled": False,
            "enabled_projects": [
                str(item) for item in enabled_projects if str(item).strip()
            ],
        },
    }


def _check_manager_auth(request: Request, expected_token: str | None) -> None:
    if expected_token:
        provided = str(
            request.headers.get("x-token-workshed-manager-token", "") or ""
        ).strip()
        if not provided or not secrets.compare_digest(provided, expected_token):
            raise HTTPException(status_code=401, detail="Unauthorized manager request.")
        return
    client = request.client
    if client is not None and str(client.host or "").strip().lower() not in {
        "127.0.0.1",
        "::1",
        "localhost",
        "testclient",
    }:
        raise HTTPException(
            status_code=403, detail="JetBrains integration is loopback-only."
        )


def _provider_token(request: Request, provider_key: str) -> None:
    raw = str(request.headers.get("authorization", "") or "").strip()
    if not raw.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Provider API key required.")
    provided = raw[7:].strip()
    if (
        not provided
        or not provider_key
        or not secrets.compare_digest(provided, provider_key)
    ):
        raise HTTPException(
            status_code=401, detail="Invalid Token Workshed Provider key."
        )


def _safe_upstream_url(value: Any) -> str:
    raw = str(value or "").strip().rstrip("/")
    try:
        parsed = urlparse(raw)
        port = parsed.port
    except ValueError as exc:
        raise HTTPException(
            status_code=503, detail="Active model server is invalid."
        ) from exc
    if parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or port is None:
        raise HTTPException(
            status_code=503, detail="Active model server is not loopback."
        )
    return raw


def _validate_provider_payload(body: Any, allowed: set[str]) -> dict[str, Any]:
    if not isinstance(body, dict):
        raise HTTPException(
            status_code=400, detail="Provider payload must be a JSON object."
        )
    unknown = sorted(set(body) - allowed)
    if unknown:
        raise HTTPException(
            status_code=400,
            detail="Unsupported provider field(s): " + ", ".join(unknown),
        )
    clean = dict(body)
    if (
        "tools" in clean
        and clean["tools"] is not None
        and not isinstance(clean["tools"], list)
    ):
        raise HTTPException(status_code=400, detail="Provider tools must be a list.")
    return clean


def _resolve_model(body: dict[str, Any], manager: Any) -> str:
    requested = str(body.get("model") or "").strip()
    try:
        available = [
            str(item).strip() for item in manager.discover_models() if str(item).strip()
        ]
    except Exception:
        available = []
    active = _current_model(manager)
    if not requested:
        requested = active
        body["model"] = requested
    if requested and requested not in available and requested != active:
        raise HTTPException(
            status_code=400,
            detail="Requested model is not available in Token Workshed.",
        )
    if not requested:
        raise HTTPException(
            status_code=503, detail="No active Token Workshed model is running."
        )
    return requested


def _forward_completion(
    *,
    manager: Any,
    endpoint: str,
    body: dict[str, Any],
) -> Response | StreamingResponse:
    if endpoint == "/v1/completions":
        body = _prepare_fim_completion(body, manager)
    upstream = _safe_upstream_url(manager.active_server_url())
    try:
        response = requests.post(
            f"{upstream}{endpoint}",
            json=body,
            stream=bool(body.get("stream")),
            timeout=(10, 600),
        )
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=503, detail="Token Workshed model server is unavailable."
        ) from exc
    if not body.get("stream"):
        content_type = response.headers.get("content-type", "application/json")
        return Response(
            content=response.content,
            status_code=response.status_code,
            media_type=content_type.split(";", 1)[0],
        )

    def stream() -> Any:
        try:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    yield chunk
        finally:
            response.close()

    return StreamingResponse(
        stream(),
        status_code=response.status_code,
        media_type=response.headers.get("content-type", "text/event-stream").split(
            ";", 1
        )[0],
    )


def register_jetbrains_integration_routes(
    app: FastAPI,
    *,
    manager: Any,
    ui_url: str,
    manager_api_token: str | None,
    bridge: IDEBridge,
) -> dict[str, Callable[[], None]]:
    """Register the restricted Provider and App-owned integration metadata."""
    if getattr(app.state, "jetbrains_integration_registered", False):
        return getattr(app.state, "jetbrains_integration_handles", {})

    provider_key = ensure_provider_key()
    acp_token = ensure_acp_token()
    bridge.register_internal_authorization(client_id=ACP_CLIENT_ID, token=acp_token)
    acp_command = str(Path(sys.executable).resolve())
    acp_args = ["-m", "token_workshed_nextcore.acp_adapter"]
    mcp_args = ["-m", "token_workshed_nextcore.mcp_gateway"]
    write_discovery_record(
        ui_url=ui_url,
        pid=os.getpid(),
        acp_command=acp_command,
        acp_args=acp_args,
        mcp_command=acp_command,
        mcp_args=mcp_args,
    )
    app.state.jetbrains_integration_registered = True

    def _internal_token(request: Request) -> None:
        raw = str(request.headers.get("authorization", "") or "").strip()
        if not raw.lower().startswith("bearer "):
            raise HTTPException(
                status_code=401, detail="App-owned integration token required."
            )
        token = raw[7:].strip()
        if not token or not secrets.compare_digest(token, acp_token):
            raise HTTPException(
                status_code=401, detail="Invalid App-owned integration token."
            )

    @app.get("/api/jetbrains/integration")
    async def jetbrains_integration(request: Request) -> dict[str, Any]:
        _check_manager_auth(request, manager_api_token)
        return _integration_config(manager, include_provider_key=True)

    @app.get("/api/jetbrains/completion-capabilities")
    async def jetbrains_completion_capabilities(request: Request) -> dict[str, Any]:
        _check_manager_auth(request, manager_api_token)
        return {"ok": True, "diagnostics": _completion_records()}

    @app.get("/api/ide/v2/integration")
    async def ide_v2_integration(request: Request) -> dict[str, Any]:
        try:
            bridge.authorize(request, "ide.status.read")
        except IDEBridgeError as exc:
            raise HTTPException(
                status_code=exc.status_code, detail=exc.message
            ) from exc
        config = _integration_config(manager, include_provider_key=False)
        config["provider"].pop("model", None)
        return config

    @app.get("/v1/models")
    async def jetbrains_provider_models(request: Request) -> dict[str, Any]:
        _provider_token(request, provider_key)
        models = []
        with suppress(Exception):
            models = [
                str(item).strip()
                for item in manager.discover_models()
                if str(item).strip()
            ]
        active = _current_model(manager)
        if active and active not in models:
            models.insert(0, active)
        return {
            "object": "list",
            "data": [
                {"id": item, "object": "model", "owned_by": "token-workshed"}
                for item in models
            ],
        }

    @app.post("/v1/chat/completions", response_model=None)
    async def jetbrains_provider_chat(request: Request) -> Response | StreamingResponse:
        _provider_token(request, provider_key)
        body = _validate_provider_payload(await request.json(), _CHAT_FIELDS)
        _resolve_model(body, manager)
        return _forward_completion(
            manager=manager, endpoint="/v1/chat/completions", body=body
        )

    @app.post("/v1/completions", response_model=None)
    async def jetbrains_provider_completion(
        request: Request,
    ) -> Response | StreamingResponse:
        _provider_token(request, provider_key)
        body = _validate_provider_payload(await request.json(), _COMPLETION_FIELDS)
        _resolve_model(body, manager)
        return _forward_completion(
            manager=manager, endpoint="/v1/completions", body=body
        )

    @app.get("/api/jetbrains/mcp/tools")
    async def jetbrains_mcp_tools(request: Request) -> dict[str, Any]:
        _internal_token(request)
        # The App's SGLang Agent tools (including file, terminal, browser, and
        # computer control) are callable only from the NextCore chat surface.
        # An arbitrary upstream MCP registry cannot be safely classified by
        # tool name or description, so never expose it to external MCP clients.
        return {
            "ok": True,
            "tools": [],
            "count": 0,
            "message": "Agent tools are available only from NextCore chat.",
        }

    @app.post("/api/jetbrains/mcp/call")
    async def jetbrains_mcp_call(request: Request) -> dict[str, Any]:
        _internal_token(request)
        raise HTTPException(
            status_code=403,
            detail="Agent tools are available only from NextCore chat.",
        )

    @app.post("/api/jetbrains/mcp/intellij-policy")
    async def jetbrains_intellij_policy(request: Request) -> dict[str, Any]:
        _check_manager_auth(request, manager_api_token)
        body = await request.json()
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="MCP policy must be an object.")
        project = str(body.get("project") or "").strip()
        enabled = bool(body.get("enabled"))
        if not project or not os.path.isabs(project) or "\x00" in project:
            raise HTTPException(
                status_code=400, detail="A valid absolute project path is required."
            )

        def store(state: dict[str, Any]) -> bool:
            integration = state.get(INTEGRATION_STATE_KEY)
            if not isinstance(integration, dict):
                integration = {}
            projects = integration.get("intellij_mcp_projects", [])
            if not isinstance(projects, list):
                projects = []
            projects = [str(item) for item in projects if str(item) != project]
            if enabled:
                projects.append(project)
            integration["intellij_mcp_projects"] = projects[-128:]
            state[INTEGRATION_STATE_KEY] = integration
            return True

        mutate_desktop_state(store)
        return {"ok": True, "project": project, "enabled": enabled}

    def close() -> None:
        remove_discovery_if_pid(os.getpid())

    handles: dict[str, Callable[[], None]] = {"close": close}
    app.state.jetbrains_integration_handles = handles

    @app.on_event("shutdown")
    async def _shutdown_jetbrains_integration() -> None:
        close()

    return handles
