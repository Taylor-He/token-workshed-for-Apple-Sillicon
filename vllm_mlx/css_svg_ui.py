# SPDX-License-Identifier: Apache-2.0
"""FastAPI backend API for the token-workshed native UI.

This module preserves the HTTP API that the old browser UI used while the
desktop surface is provided by the Rust Iced/libcosmic frontend.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import ipaddress
import json
import logging
import math
import os
import re
import secrets
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
import uuid
import webbrowser
from pathlib import Path
from typing import Any

import requests
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .agent_bridge import run_bridge_programmatic
from .agent_bridge.family_detect import detect_model_family
from .agent_bridge.model_meta import read_model_metadata
from .agent_bridge.registry import load_registry, match_rule, resolve_settings
from .agent_profiles import (
    PROFILE_VERSION,
    canonical_model_id,
    delete_agent_profile,
    load_agent_profile,
    save_agent_profile,
)
from .version import TOKEN_WORKSHED_VERSION, VLLM_MLX_VERSION

logger = logging.getLogger(__name__)

OPENCLAW_DIR = Path(__file__).resolve().parent.parent / "integrations" / "openclaw"
_OPENCLAW_LEGACY_DIR = (
    Path(__file__).resolve().parent.parent / "integrations" / "cypherclaw"
)
if not OPENCLAW_DIR.exists() and _OPENCLAW_LEGACY_DIR.exists():
    OPENCLAW_DIR = _OPENCLAW_LEGACY_DIR
# OpenClaw rejects tool execution when its approvals path traverses a symlink.
# Keep the compatibility alias on disk, but use the canonical directory for
# subprocess cwd, workspace, config, and state paths.
OPENCLAW_DIR = OPENCLAW_DIR.resolve()
OPENCLAW_STATE_DIR = OPENCLAW_DIR / ".token-workshed-state"
OPENCLAW_CONFIG_PATH = OPENCLAW_STATE_DIR / "openclaw.json"
OPENCLAW_WORKSPACE_DIR = OPENCLAW_STATE_DIR / "workspace"
OPENCLAW_RUNNER = OPENCLAW_DIR / "scripts" / "run-node.mjs"
OPENCLAW_DIST_ENTRY = OPENCLAW_DIR / "dist" / "entry.js"
_OPENCLAW_LEGACY_DIST_ENTRY = OPENCLAW_DIR / "dist" / "index.js"
HERMES_DIR = Path(__file__).resolve().parent.parent / "integrations" / "hermes"
HERMES_STATE_DIR = HERMES_DIR / ".token-workshed-state"
HERMES_HOME_DIR = HERMES_STATE_DIR / "home"
HERMES_CONFIG_PATH = HERMES_HOME_DIR / "config.yaml"
HERMES_ENTRY = HERMES_DIR / "hermes"
HERMES_VENV_DIR = HERMES_STATE_DIR / "venv"
HERMES_VENV_PYTHON = HERMES_VENV_DIR / "bin" / "python"
HERMES_RUNTIME_VERSION_STAMP = HERMES_STATE_DIR / "runtime-version"
HERMES_MIN_PYTHON = (3, 11)
HERMES_MAX_PYTHON = (3, 14)
TOKEN_WORKSHED_ROOT = Path(__file__).resolve().parent.parent
OPENCLAW_BUNDLED_NODE = (
    TOKEN_WORKSHED_ROOT / "integrations" / "node-runtime" / "bin" / "node"
)
OPENCLAW_NODE_OVERRIDE_ENV = "TOKEN_WORKSHED_OPENCLAW_NODE_BIN"
OPENCLAW_NODE_MINIMUMS = {
    22: (22, 22, 3),
    24: (24, 15, 0),
    25: (25, 9, 0),
}
_OPENCLAW_HINT = (
    "OpenClaw mode is enabled. Execute requested tools directly and return concrete results. "
    "For web requests, use available browser/web_fetch/web_search tools instead of capability disclaimers. "
    "Do not echo or summarize this instruction text."
)
_OPENCLAW_SESSION_ID_RE = re.compile(r"[^a-zA-Z0-9._-]+")
_OPENCLAW_TOOL_CALL_TAG_RE = re.compile(
    r"(?is)<\s*tool_call\b[^>]*>.*?<\s*/\s*tool_call\s*>"
)
_OPENCLAW_TOOL_CALL_OPEN_RE = re.compile(r"(?is)<\s*tool_call\b")
_OPENCLAW_NON_ANSWER_SUMMARIES = frozenset(
    {
        "completed",
        "complete",
        "ok",
        "success",
        "succeeded",
        "done",
        "running",
        "pending",
        "error",
        "failed",
        "timeout",
    }
)
_OPENCLAW_CAPABILITY_REFUSAL_SNIPPETS = (
    "don't have direct access",
    "do not have direct access",
    "don't have access to your local filesystem",
    "do not have access to your local filesystem",
    "can't access your local filesystem",
    "cannot access your local filesystem",
    "doesn't provide web search functionality",
    "does not provide web search functionality",
    "no web search functionality",
    "i can't search the web",
    "i cannot search the web",
    "i don't have access to search the web",
    "i do not have access to search the web",
    "web_search tool requires a brave search api key",
    "requires a brave search api key",
    "set up the web search api key",
    "openclaw configure --section web",
    "blocked: resolves to private/internal/special-use ip address",
    "could not connect to chrome. check if chrome is running.",
    "could not find devtoolsactiveport for chrome",
)
_OPENCLAW_CONFIG_LOCK = threading.Lock()
_OPENCLAW_NODE_RUN_LOCK = threading.Lock()
_OPENCLAW_AGENT_HELP_LOCK = threading.Lock()
_OPENCLAW_AGENT_CAPABILITIES_CACHE: dict[str, frozenset[str]] = {}
_OPENCLAW_GATEWAY_LOCK = threading.Lock()
_OPENCLAW_GATEWAY_PROCESS: subprocess.Popen[str] | None = None
_OPENCLAW_GATEWAY_RUNTIME_KEY = ""
_OPENCLAW_GATEWAY_PORT = 0
_OPENCLAW_WARM_CACHE_LOCK = threading.Lock()
_OPENCLAW_WARM_CACHE: dict[str, float] = {}
_OPENCLAW_GATEWAY_DEFAULT_PORT = 19123
_OPENCLAW_GATEWAY_STARTUP_TIMEOUT_DEFAULT_SECONDS = 60.0
_OPENCLAW_GATEWAY_HEALTH_ENDPOINTS = ("/readyz", "/ready", "/healthz", "/health")
_OPENCLAW_GATEWAY_LOG_PATH = OPENCLAW_STATE_DIR / "gateway.log"
_HERMES_SESSION_LOCK = threading.Lock()
_HERMES_SESSION_BY_CONVERSATION: dict[str, str] = {}
_HERMES_SESSION_ID_RE = re.compile(r"session_id:\s*([A-Za-z0-9._:-]+)", re.IGNORECASE)
_HERMES_MIN_CONTEXT_RE = re.compile(
    r"context window of\s*([0-9][0-9,]*)\s*tokens?.*?below the minimum\s*([0-9][0-9,]*)",
    re.IGNORECASE | re.DOTALL,
)
_HERMES_MIN_CONTEXT_FALLBACK_RE = re.compile(
    r"(?:minimum|at least)\s*([0-9][0-9,]*)\s*(?:tokens?|k)\s*(?:context|required)",
    re.IGNORECASE,
)
_HERMES_BOOTSTRAP_LOCK = threading.Lock()
_HERMES_BOOTSTRAP_READY = False
_HERMES_BOOTSTRAP_PYTHON = ""
_HERMES_ENV_LOCK = threading.Lock()
_HERMES_CONFIG_LOCK = threading.Lock()
_AGENT_TOOL_PROBE_LOCK = threading.Lock()
_AGENT_TOOL_PROBE_CACHE: dict[str, dict[str, Any]] = {}
_OPENCLAW_OLLAMA_DEFAULT_BASE_URL = "http://127.0.0.1:11434"
_OPENCLAW_OLLAMA_MODEL_CANDIDATES = (
    "qwen3.5",
    "glm-4.7-flash",
    "gpt-oss:20b",
    "llama3.3",
    "qwen3",
    "qwen2.5-coder",
)
_OPENCLAW_BROWSER_DEFAULT_PROFILE = "openclaw"
_OPENCLAW_BROWSER_USER_PROFILE_COLOR = "#00AA00"
THINKING_HEADER_RE = re.compile(
    r"(?is)^\s{0,3}(?:#{1,6}\s*)?(?:thinking process|reasoning|思考过程|推理过程|思考)\s*:?\s*"
)
FINAL_ANSWER_MARKER_RE = re.compile(
    r"(?is)(?:^|\n)\s{0,3}(?:#{1,6}\s*)?(?:final answer|answer|最终答案|最终回答|答案)\s*:?\s*"
)
LEADING_FINAL_ANSWER_RE = re.compile(
    r"(?is)^\s{0,3}(?:#{1,6}\s*)?(?:final answer|answer|最终答案|最终回答|答案)\s*:?\s*"
)
THINK_OPEN_TAG_RE = re.compile(r"(?is)<think>")
THINK_CLOSE_TAG_RE = re.compile(r"(?is)</think>")
TITLE_MODEL_CANDIDATES = (
    # User-requested title model: functiongemma-270m-INT4
    "mlx-community/functiongemma-270m-it-4bit",
)
_TITLE_MODEL_ALIASES = frozenset(
    {
        "functiongemma-270m-int4",
        "functiongemma-270m-it-4bit",
        "mlx-community/functiongemma-270m-int4",
        "mlx-community/functiongemma-270m-it-4bit",
    }
)
_TITLE_MODEL_LOCK = threading.Lock()
_TITLE_MODEL: Any | None = None
_TITLE_TOKENIZER: Any | None = None
_TITLE_MODEL_ID = ""
_TITLE_MODEL_DISABLED = False

_OPENCLAW_TOOL_MODE_VALUES = frozenset({"none", "single", "small", "full"})
_OPENCLAW_LIGHTWEIGHT_FILE_TOOLS = ("exec", "process", "read", "write", "edit")
_OPENCLAW_LIGHTWEIGHT_WEB_TOOLS = ("web_search", "web_fetch", "browser")
_OPENCLAW_LOCAL_NONE_FIRST_CHUNK_TIMEOUT_MS = 15_000
_OPENCLAW_LOCAL_TOOLS_FIRST_CHUNK_TIMEOUT_MS = 30_000
_OPENCLAW_DEFAULT_REPETITION_PENALTY = 1.08
_OPENCLAW_IDENTITY_STOP_SEQUENCES = (
    "My core truths are:",
    "my core truths are:",
    "Be genuinely helpful, not performatively helpful.",
    "be genuinely helpful, not performatively helpful.",
    "I am an evolving entity, designed to assist and learn alongside you.",
    "I'm an evolving entity, designed to assist and learn alongside you.",
)
_OPENCLAW_PERSONA_MANIFESTO_MARKERS = (
    "my core truths are",
    "be genuinely helpful, not performatively helpful",
    "have opinions",
    "be resourceful before asking",
    "earn trust through competence",
    "i am an evolving entity, designed to assist and learn alongside you",
    "i'm an evolving entity, designed to assist and learn alongside you",
    "核心价值观",
    "核心真理",
)
_OPENCLAW_PERSONA_LIST_LINE_MARKERS = (
    "be genuinely helpful",
    "have opinions",
    "be resourceful before asking",
    "earn trust through competence",
)


class ChatRequest(BaseModel):
    """Payload sent by the frontend chat page."""

    message: str = Field(default="")
    user_content: Any | None = None
    history: list[dict[str, Any]] = Field(default_factory=list)
    server_url: str | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    system_prompt: str | None = None
    model: str | None = None
    openclaw_enabled: bool = False
    cypherclaw_enabled: bool | None = None
    agent_runtime: str | None = None
    conversation_id: str | None = None
    session_id: str | None = None
    run_id: str | None = None
    tool_mode: str | None = None
    allowed_tools: list[str] | None = None
    model_first_chunk_timeout_ms: int | None = None


class OpenClawWarmupRequest(BaseModel):
    """Payload for warming up OpenClaw Node runtime."""

    server_url: str | None = None
    model: str | None = None


class AgentProbeRequest(BaseModel):
    """Payload for runtime/model tool-call compatibility probes."""

    server_url: str | None = None
    model: str | None = None
    runtime: str | None = None
    force: bool = False


class AgentProfileConfigureRequest(BaseModel):
    """Payload for calibrating both bundled agent runtimes for one model."""

    model: str = Field(default="")
    server_url: str | None = None
    force: bool = False


class TitleRequest(BaseModel):
    """Payload sent by frontend to generate a concise conversation title."""

    messages: list[dict[str, Any]] = Field(default_factory=list)
    fallback: str = Field(default="New Conversation")


class UIConfig(BaseModel):
    """Default settings exposed to the frontend on load."""

    server_url: str
    max_tokens: int
    temperature: float
    token_workshed_version: str = TOKEN_WORKSHED_VERSION
    vllm_mlx_version: str = VLLM_MLX_VERSION
    manager_auth_required: bool = False
    openclaw_enabled: bool = False
    cypherclaw_enabled: bool = False
    agent_runtime: str = "auto"


_AGENT_RUNTIME_ALIASES = {
    "auto": "auto",
    "openclaw": "openclaw",
    "openclaw/hermes": "openclaw",
    "openclaw-hermes": "openclaw",
    "both": "openclaw",
    "dual": "openclaw",
    "cypherclaw": "openclaw",
    "hermes": "hermes",
}


def _resolve_agent_runtime(value: str | None) -> str:
    normalized = str(value or "").strip().lower()
    if not normalized:
        return "auto"
    return _AGENT_RUNTIME_ALIASES.get(normalized, "auto")


def _resolve_agent_runtime_candidates(value: str | None) -> tuple[str, ...]:
    runtime = _resolve_agent_runtime(value)
    if runtime == "hermes":
        return ("hermes",)
    return ("openclaw",)


def _agent_runtime_label(runtime: str) -> str:
    return "Hermes" if runtime == "hermes" else "OpenClaw"


def _agent_runtime_stage(runtime: str) -> str:
    return "hermes" if runtime == "hermes" else "openclaw"


def _build_agent_runtime_error_detail(
    *,
    runtime: str,
    selected_model: str,
    reason: str,
    probe_warning: str = "",
) -> dict[str, Any]:
    runtime_label = _agent_runtime_label(runtime)
    reason_text = _truncate_error_text(str(reason or "").strip(), limit=360)
    message = (
        f"{runtime_label} failed: {reason_text}"
        if reason_text
        else f"{runtime_label} failed."
    )
    detail: dict[str, Any] = {
        "message": message,
        "error_type": "agent_runtime_failed",
        "runtime": runtime,
        "model": selected_model,
        "retryable": True,
        "reason": reason_text or None,
    }
    warning_text = _truncate_error_text(str(probe_warning or "").strip(), limit=320)
    if warning_text:
        detail["probe_warning"] = warning_text
    return detail


def _is_openclaw_enabled(payload: ChatRequest) -> bool:
    return bool(payload.openclaw_enabled) or bool(payload.cypherclaw_enabled)


def _resolve_agent_probe_ttl_seconds(default_seconds: int = 600) -> float:
    raw = os.environ.get("TOKEN_WORKSHED_AGENT_PROBE_TTL_SECONDS", "").strip()
    if not raw:
        return float(default_seconds)
    try:
        parsed = float(raw)
    except ValueError:
        return float(default_seconds)
    return max(30.0, min(3600.0, parsed))


def _should_run_agent_probe_on_chat(default: bool = False) -> bool:
    return _env_flag("TOKEN_WORKSHED_AGENT_PROBE_ON_CHAT", default)


def _resolve_agent_probe_startup_timeout_seconds(
    default_seconds: float = 90.0,
) -> float:
    raw = os.environ.get(
        "TOKEN_WORKSHED_AGENT_PROBE_STARTUP_TIMEOUT_SECONDS", ""
    ).strip()
    if not raw:
        return float(default_seconds)
    try:
        parsed = float(raw)
    except ValueError:
        return float(default_seconds)
    return max(10.0, min(600.0, parsed))


def _resolve_agent_probe_request_timeout_seconds(
    default_seconds: float = 90.0,
) -> float:
    raw = os.environ.get(
        "TOKEN_WORKSHED_AGENT_PROBE_REQUEST_TIMEOUT_SECONDS", ""
    ).strip()
    if not raw:
        return float(default_seconds)
    try:
        parsed = float(raw)
    except ValueError:
        return float(default_seconds)
    return max(4.0, min(180.0, parsed))


def _split_server_host_port(server_url: str) -> tuple[str, int]:
    parsed = urllib.parse.urlparse(server_url)
    host = (parsed.hostname or "").strip()
    if not host:
        raise ValueError(f"Invalid server URL host: {server_url}")
    if parsed.port is not None:
        return host, int(parsed.port)
    if parsed.scheme.lower() == "https":
        return host, 443
    return host, 80


def _probe_openai_tool_call(
    *,
    target: str,
    selected_model: str,
    timeout_seconds: int = 30,
) -> dict[str, Any]:
    tool_spec = {
        "type": "function",
        "function": {
            "name": "probe_tool",
            "description": "Compatibility probe tool. Return tool call only.",
            "parameters": {
                "type": "object",
                "required": ["text"],
                "properties": {
                    "text": {"type": "string"},
                },
            },
        },
    }
    payload = {
        "model": selected_model,
        "messages": [
            {
                "role": "user",
                "content": "Call probe_tool with text='ok'. Return only tool call.",
            }
        ],
        "tools": [tool_spec],
        "tool_choice": "required",
        "max_tokens": 96,
        "temperature": 0,
    }

    def _post_probe(request_payload: dict[str, Any]) -> requests.Response:
        response, _ = _post_chat_completions_with_connect_retry(
            target=target,
            payload=request_payload,
            timeout_seconds=max(10, int(timeout_seconds)),
        )
        return response

    try:
        response = _post_probe(payload)
    except Exception as exc:
        return {
            "ok": False,
            "reason": f"OpenAI chat probe request failed: {exc}",
            "status": None,
        }

    status_code = int(response.status_code)
    if not response.ok:
        body_preview = ""
        try:
            body_preview = _truncate_error_text(response.text, limit=220)
        except Exception:
            body_preview = ""
        lower_preview = body_preview.lower()
        if (
            status_code in {400, 422}
            and "tool_choice" in lower_preview
            and "required" in lower_preview
        ):
            auto_payload = dict(payload)
            auto_payload["tool_choice"] = "auto"
            try:
                response = _post_probe(auto_payload)
                status_code = int(response.status_code)
            except Exception as exc:
                return {
                    "ok": False,
                    "reason": f"OpenAI chat probe auto fallback request failed: {exc}",
                    "status": None,
                }
            if response.ok:
                try:
                    data = response.json()
                except Exception as exc:
                    return {
                        "ok": False,
                        "reason": f"OpenAI chat probe auto fallback JSON decode failed: {exc}",
                        "status": status_code,
                    }
                choices = data.get("choices") if isinstance(data, dict) else None
                if isinstance(choices, list) and choices:
                    first = choices[0] if isinstance(choices[0], dict) else {}
                    message = first.get("message") if isinstance(first, dict) else {}
                    if not isinstance(message, dict):
                        message = {}
                    tool_calls = message.get("tool_calls")
                    if isinstance(tool_calls, list) and tool_calls:
                        return {"ok": True, "reason": "", "status": status_code}
                    content_text = _extract_content_text(
                        message.get("content", "")
                    ).strip()
                    if _OPENCLAW_TOOL_CALL_OPEN_RE.search(content_text):
                        return {
                            "ok": False,
                            "reason": "Model emitted literal <tool_call> text instead of structured tool_calls.",
                            "status": status_code,
                        }
                    if content_text:
                        return {
                            "ok": True,
                            "reason": "Model returned plain text during tool probe; proceeding without strict tool-call guarantee.",
                            "status": status_code,
                        }
                return {
                    "ok": False,
                    "reason": "OpenAI chat probe auto fallback produced no structured tool_calls.",
                    "status": status_code,
                }
        detail = f"HTTP {status_code}"
        if body_preview:
            detail = f"{detail} | {body_preview}"
        return {
            "ok": False,
            "reason": f"OpenAI chat probe failed: {detail}",
            "status": status_code,
        }

    try:
        data = response.json()
    except Exception as exc:
        return {
            "ok": False,
            "reason": f"OpenAI chat probe JSON decode failed: {exc}",
            "status": status_code,
        }

    choices = data.get("choices") if isinstance(data, dict) else None
    if not isinstance(choices, list) or not choices:
        return {
            "ok": False,
            "reason": "OpenAI chat probe returned no choices.",
            "status": status_code,
        }
    first = choices[0] if isinstance(choices[0], dict) else {}
    message = first.get("message") if isinstance(first, dict) else {}
    if not isinstance(message, dict):
        message = {}

    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        return {"ok": True, "reason": "", "status": status_code}

    content_text = _extract_content_text(message.get("content", "")).strip()
    if _OPENCLAW_TOOL_CALL_OPEN_RE.search(content_text):
        return {
            "ok": False,
            "reason": "Model emitted literal <tool_call> text instead of structured tool_calls.",
            "status": status_code,
        }
    if content_text:
        return {
            "ok": True,
            "reason": "Model returned plain text during tool probe; proceeding without strict tool-call guarantee.",
            "status": status_code,
        }
    return {
        "ok": False,
        "reason": "Model returned empty response for tool-call probe.",
        "status": status_code,
    }


def _probe_ollama_tool_call(
    *,
    target: str,
    selected_model: str,
    timeout_seconds: int = 30,
) -> dict[str, Any]:
    tool_spec = {
        "type": "function",
        "function": {
            "name": "probe_tool",
            "description": "Compatibility probe tool. Return tool call only.",
            "parameters": {
                "type": "object",
                "required": ["text"],
                "properties": {
                    "text": {"type": "string"},
                },
            },
        },
    }
    payload = {
        "model": selected_model,
        "messages": [
            {
                "role": "user",
                "content": "Call probe_tool with text='ok'. Return only tool call.",
            }
        ],
        "tools": [tool_spec],
        "stream": False,
    }
    try:
        response = requests.post(
            f"{target}/api/chat",
            json=payload,
            timeout=(3.0, max(10, int(timeout_seconds))),
        )
    except Exception as exc:
        return {
            "ok": False,
            "reason": f"Ollama chat probe request failed: {exc}",
            "status": None,
        }

    status_code = int(response.status_code)
    if not response.ok:
        body_preview = ""
        try:
            body_preview = _truncate_error_text(response.text, limit=220)
        except Exception:
            body_preview = ""
        detail = f"HTTP {status_code}"
        if body_preview:
            detail = f"{detail} | {body_preview}"
        return {
            "ok": False,
            "reason": f"Ollama chat probe failed: {detail}",
            "status": status_code,
        }

    try:
        data = response.json()
    except Exception as exc:
        return {
            "ok": False,
            "reason": f"Ollama chat probe JSON decode failed: {exc}",
            "status": status_code,
        }

    message = data.get("message") if isinstance(data, dict) else None
    if not isinstance(message, dict):
        message = {}
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        return {"ok": True, "reason": "", "status": status_code}

    content_text = _extract_content_text(message.get("content", "")).strip()
    if _OPENCLAW_TOOL_CALL_OPEN_RE.search(content_text):
        return {
            "ok": False,
            "reason": "Model emitted literal <tool_call> text instead of structured tool_calls.",
            "status": status_code,
        }
    if content_text:
        return {
            "ok": True,
            "reason": "Model returned plain text during tool probe; proceeding without strict tool-call guarantee.",
            "status": status_code,
        }
    return {
        "ok": False,
        "reason": "Model returned empty response for tool-call probe.",
        "status": status_code,
    }


def _extract_context_window_hint(payload: dict[str, Any]) -> int | None:
    """Extract context-length hints from model metadata payloads."""
    if not isinstance(payload, dict):
        return None
    candidates = (
        payload.get("context_length"),
        payload.get("contextWindow"),
        payload.get("max_context_length"),
        payload.get("max_model_len"),
        payload.get("max_seq_len"),
        payload.get("max_tokens"),
    )
    for value in candidates:
        parsed = _coerce_int(value)
        if parsed is not None and parsed > 0:
            return parsed
    return None


def _query_local_model_context_window(
    *, target: str, selected_model: str
) -> int | None:
    """Best-effort local context window discovery for runtime gating."""
    model_id = str(selected_model or "").strip()
    if not model_id:
        return None

    encoded = urllib.parse.quote(model_id, safe="")
    detail_urls = (
        f"{target}/v1/models/{encoded}",
        f"{target}/api/v1/models/{encoded}",
    )
    for url in detail_urls:
        try:
            response = requests.get(url, timeout=(1.2, 2.5))
            if not response.ok:
                continue
            data = response.json()
        except Exception:
            continue
        if isinstance(data, dict):
            context_window = _extract_context_window_hint(data)
            if context_window is not None:
                return context_window
            nested = data.get("data")
            if isinstance(nested, dict):
                context_window = _extract_context_window_hint(nested)
                if context_window is not None:
                    return context_window

    list_urls = (
        f"{target}/v1/models",
        f"{target}/api/v1/models",
    )
    for url in list_urls:
        try:
            response = requests.get(url, timeout=(1.2, 2.5))
            if not response.ok:
                continue
            data = response.json()
        except Exception:
            continue
        items = data.get("data") if isinstance(data, dict) else None
        if not isinstance(items, list):
            items = data.get("models") if isinstance(data, dict) else None
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            item_id = str(
                item.get("id") or item.get("name") or item.get("key") or ""
            ).strip()
            if not item_id:
                continue
            if (
                item_id == model_id
                or item_id.endswith(f"/{model_id}")
                or model_id.endswith(f"/{item_id}")
            ):
                context_window = _extract_context_window_hint(item)
                if context_window is not None:
                    return context_window
    return None


def _run_agent_runtime_probe(
    *,
    server_url: str,
    selected_model: str,
    runtime: str,
    force: bool = False,
    tool_call_parser: str | None = None,
    reasoning_parser: str | None = None,
    chat_template: str | None = None,
) -> dict[str, Any]:
    runtime_name = (
        "hermes" if str(runtime or "").strip().lower() == "hermes" else "openclaw"
    )
    model_name = str(selected_model or "").strip() or "default"
    target = _normalize_server_url(server_url)
    cache_key = f"{runtime_name}|{target}|{model_name}"
    ttl_seconds = _resolve_agent_probe_ttl_seconds()
    now_ts = float(time.time())

    if not force:
        with _AGENT_TOOL_PROBE_LOCK:
            cached = _AGENT_TOOL_PROBE_CACHE.get(cache_key)
            if isinstance(cached, dict):
                age = now_ts - float(cached.get("checked_at", 0.0) or 0.0)
                if age <= ttl_seconds:
                    cached_copy = dict(cached)
                    cached_copy["cached"] = True
                    return cached_copy

    host, port = _split_server_host_port(target)
    startup_timeout = _resolve_agent_probe_startup_timeout_seconds()
    request_timeout = _resolve_agent_probe_request_timeout_seconds()
    try:
        bridge_report = run_bridge_programmatic(
            model=model_name,
            agent=runtime_name,
            host=host,
            port=port,
            run=True,
            dry_run=False,
            allow_download=False,
            tool_call_parser=tool_call_parser,
            reasoning_parser=reasoning_parser,
            chat_template=chat_template,
            startup_timeout=startup_timeout,
            request_timeout=request_timeout,
            stop_after_check=False,
            no_reuse_running=False,
        )
    except Exception as exc:
        normalized_result = {
            "ok": False,
            "runtime": runtime_name,
            "server_url": target,
            "model": model_name,
            "reason": f"Bridge probe execution failed: {exc}",
            "status": 500,
            "checked_at": now_ts,
            "cached": False,
            "agent_ready": False,
            "tool_calling": "failed",
            "parser_used": None,
            "reasoning_parser_used": None,
            "chat_template_used": None,
            "needs_manual_review": True,
            "failures": [f"Bridge probe execution failed: {exc}"],
            "warnings": [],
            "failure_category": "runtime_environment",
            "fix_suggestions": [
                "Check bridge dependencies and local server settings, then retry probe.",
            ],
            "bridge_report": None,
        }
        with _AGENT_TOOL_PROBE_LOCK:
            _AGENT_TOOL_PROBE_CACHE[cache_key] = dict(normalized_result)
        return normalized_result
    failures = bridge_report.get("failures")
    if isinstance(failures, list):
        first_failure = next(
            (str(item).strip() for item in failures if str(item).strip()),
            "",
        )
    else:
        first_failure = ""
    health_error = ""
    health = bridge_report.get("health_check")
    if isinstance(health, dict):
        health_error = str(health.get("error") or "").strip()
    smoke_error = ""
    smoke = bridge_report.get("smoke_test")
    if isinstance(smoke, dict):
        smoke_error = str(smoke.get("reason") or "").strip()
    recovery_attempted = (
        bool(smoke.get("recovery_attempted")) if isinstance(smoke, dict) else False
    )
    tool_choice_fallback = (
        smoke.get("tool_choice_fallback") if isinstance(smoke, dict) else None
    )
    failure_reason = (
        first_failure or smoke_error or health_error or "Agent bridge probe failed."
    )
    ok = bool(bridge_report.get("agent_ready"))
    warnings = bridge_report.get("warnings")
    if not isinstance(warnings, list):
        warnings = []
    if ok:
        failure_category = ""
        status = 200
    elif isinstance(health, dict) and health.get("model_present") is False:
        failure_category = "model_mismatch"
        status = 422
    elif isinstance(health, dict) and not bool(health.get("ok")):
        # Health failures are generally recoverable cold-start/port issues and
        # should get the retry path instead of being reported as parser 422s.
        failure_category = "server_startup"
        status = 503
    elif smoke_error or first_failure:
        reason_text = (
            f"{failure_reason} {' '.join(str(item) for item in failures or [])}".lower()
        )
        if any(
            marker in reason_text
            for marker in (
                "tool_call",
                "tool call",
                "parser",
                "plain text",
                "markup",
                "arguments",
                "chat template",
            )
        ):
            failure_category = "tool_call_compatibility"
        else:
            failure_category = "runtime_configuration"
        status = 422
    else:
        failure_category = "unknown"
        status = 422

    normalized_result = {
        "ok": ok,
        "runtime": runtime_name,
        "server_url": target,
        "model": model_name,
        "reason": "" if ok else failure_reason,
        "status": status,
        "checked_at": now_ts,
        "cached": False,
        "agent_ready": bool(bridge_report.get("agent_ready")),
        "tool_calling": str(bridge_report.get("tool_calling") or "failed"),
        "parser_used": bridge_report.get("parser_used"),
        "reasoning_parser_used": bridge_report.get("reasoning_parser_used"),
        "chat_template_used": bridge_report.get("chat_template_used"),
        "needs_manual_review": bool(bridge_report.get("needs_manual_review")),
        "failures": failures if isinstance(failures, list) else [],
        "warnings": warnings,
        "failure_category": failure_category,
        "recovery_attempted": recovery_attempted,
        "tool_choice_fallback": tool_choice_fallback,
        "fix_suggestions": (
            bridge_report.get("fix_suggestions")
            if isinstance(bridge_report.get("fix_suggestions"), list)
            else []
        ),
        "bridge_report": bridge_report,
    }
    with _AGENT_TOOL_PROBE_LOCK:
        _AGENT_TOOL_PROBE_CACHE[cache_key] = dict(normalized_result)

    return normalized_result


def _format_agent_probe_failure(runtime: str, probe: dict[str, Any]) -> str:
    """Turn probe diagnostics into a concise, actionable job error."""
    label = str(runtime or "Agent").strip() or "Agent"
    reason = str(probe.get("reason") or "tool-call probe failed").strip()
    category = str(probe.get("failure_category") or "").strip()
    suggestions = [
        str(item).strip()
        for item in (probe.get("fix_suggestions") or [])
        if str(item).strip()
    ]
    if category == "server_startup":
        recovery = "Restart or select the model server, then retry Configure."
    elif category == "tool_call_compatibility":
        recovery = "Check the selected model's tool parser/chat template and retry."
    elif category == "runtime_environment":
        recovery = "Check bridge dependencies and local runtime logs, then retry."
    elif category == "model_mismatch":
        recovery = "Switch the server to the selected model, then retry."
    else:
        recovery = "Retry Configure after checking the local server logs."
    if suggestions:
        recovery = f"{recovery} {suggestions[0]}"
    return f"{label}: {reason} ({category or 'unknown'}). {recovery}"


def _normalize_title_text(raw: str, fallback: str = "New Conversation") -> str:
    """Normalize generated conversation title for stable UI rendering."""
    text = str(raw or "")
    text = re.sub(r"[\r\n]+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"`+|[*_#>\[\]\(\)]", " ", text)
    text = re.sub(
        r"^\s*(?:title|conversation title|topic)\s*[:\-]\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"^[\"'“”‘’]+|[\"'“”‘’]+$", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[.,;:!?-]+$", "", text).strip()
    if not text:
        text = str(fallback or "").strip() or "New Conversation"
    if len(text) > 64:
        text = f"{text[:64]}..."
    return text


def _heuristic_title_from_messages(
    messages: list[dict[str, Any]],
    fallback: str = "New Conversation",
) -> str:
    """Build a readable fallback title from first meaningful user line."""
    for item in messages:
        if not isinstance(item, dict):
            continue
        if str(item.get("role", "")).strip().lower() != "user":
            continue
        raw = str(item.get("content", ""))
        for line in re.split(r"\n+", raw):
            candidate = line.strip()
            if not candidate:
                continue
            if re.match(r"^\[(image|video)\]", candidate, flags=re.IGNORECASE):
                continue
            candidate = re.sub(r"https?://\S+", "", candidate)
            candidate = re.sub(r"\s+", " ", candidate).strip()
            if candidate:
                return _normalize_title_text(candidate, fallback)
    return _normalize_title_text("", fallback)


def _build_title_seed(messages: list[dict[str, Any]]) -> str:
    """Build compact dialogue snippet for lightweight title generation."""
    snippets: list[str] = []
    for item in messages[-14:]:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip().lower()
        if role not in {"user", "assistant"}:
            continue
        content = str(item.get("content", ""))
        if not content.strip():
            continue
        content = re.sub(
            r"\[(?:image|video)\]\s*[^\n]*", "", content, flags=re.IGNORECASE
        )
        content = re.sub(r"\s+", " ", content).strip()
        if not content:
            continue
        if len(content) > 320:
            content = f"{content[:320]}..."
        prefix = "User" if role == "user" else "Assistant"
        snippets.append(f"{prefix}: {content}")
    return "\n".join(snippets[-10:])[:2200]


def _load_title_model() -> tuple[Any, Any, str] | None:
    """Lazy-load ultra-light title model (auto-download on first use)."""
    global _TITLE_MODEL, _TITLE_TOKENIZER, _TITLE_MODEL_ID, _TITLE_MODEL_DISABLED
    with _TITLE_MODEL_LOCK:
        if (
            _TITLE_MODEL is not None
            and _TITLE_TOKENIZER is not None
            and _TITLE_MODEL_ID in TITLE_MODEL_CANDIDATES
        ):
            return _TITLE_MODEL, _TITLE_TOKENIZER, _TITLE_MODEL_ID
        if _TITLE_MODEL is not None and _TITLE_MODEL_ID not in TITLE_MODEL_CANDIDATES:
            _TITLE_MODEL = None
            _TITLE_TOKENIZER = None
            _TITLE_MODEL_ID = ""
        if _TITLE_MODEL_DISABLED:
            return None

        try:
            from mlx_lm import load
        except Exception:
            _TITLE_MODEL_DISABLED = True
            return None

        for model_id in TITLE_MODEL_CANDIDATES:
            try:
                model, tokenizer = load(model_id)
                _TITLE_MODEL = model
                _TITLE_TOKENIZER = tokenizer
                _TITLE_MODEL_ID = model_id
                return model, tokenizer, model_id
            except Exception:
                continue

        _TITLE_MODEL_DISABLED = True
        return None


def _release_title_model() -> None:
    """Release title model weights to reduce memory interference with main model."""
    global _TITLE_MODEL, _TITLE_TOKENIZER, _TITLE_MODEL_ID
    with _TITLE_MODEL_LOCK:
        _TITLE_MODEL = None
        _TITLE_TOKENIZER = None
        _TITLE_MODEL_ID = ""
    try:
        import gc

        gc.collect()
    except Exception:
        pass
    try:
        import mlx.core as mx

        clear_fn = getattr(mx, "clear_cache", None)
        if callable(clear_fn):
            clear_fn()
        else:
            mx.metal.clear_cache()
    except Exception:
        pass


def _generate_title_with_light_model(seed_text: str) -> tuple[str | None, str]:
    """Generate title with configured lightweight title model, fallback on error."""
    loaded = _load_title_model()
    if loaded is None:
        return None, "fallback"

    model, tokenizer, model_id = loaded
    try:
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        prompt = (
            "You write concise conversation titles.\n"
            "Rules:\n"
            "- 3 to 8 words.\n"
            "- Keep same language as the user's conversation.\n"
            "- No markdown.\n"
            "- Output only the title text.\n\n"
            f"Conversation:\n{seed_text}\n\n"
            "Title:"
        )

        sampler = make_sampler(temp=0.1, top_p=0.9)
        generated = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=20,
            sampler=sampler,
            verbose=False,
        )
        return _normalize_title_text(generated, ""), model_id
    except Exception:
        return None, "fallback"
    finally:
        _release_title_model()


def _generate_conversation_title(
    messages: list[dict[str, Any]],
    fallback: str = "New Conversation",
) -> tuple[str, str]:
    """Generate robust title: lightweight model first, heuristic fallback second."""
    safe_fallback = _normalize_title_text("", fallback)
    seed_text = _build_title_seed(messages)
    if not seed_text.strip():
        return safe_fallback, "fallback"

    generated, source = _generate_title_with_light_model(seed_text)
    if generated:
        return _normalize_title_text(generated, safe_fallback), source

    return _heuristic_title_from_messages(messages, safe_fallback), "fallback"


def _normalize_server_url(raw_url: str) -> str:
    """Normalize user-supplied server URL and enforce local/allowlist targets."""
    value = raw_url.strip()
    if not value:
        raise ValueError("Server URL cannot be empty.")

    if not value.startswith(("http://", "https://")):
        value = f"http://{value}"

    parsed = urllib.parse.urlparse(value)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Server URL must use http or https.")

    if parsed.username or parsed.password:
        raise ValueError("Server URL must not include credentials.")

    hostname = parsed.hostname
    if not hostname:
        raise ValueError("Server URL must include a valid host.")

    default_allowlist = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}
    extra_hosts_raw = os.environ.get("TOKEN_WORKSHED_ALLOWED_SERVER_HOSTS", "")
    extra_allowlist = {
        host.strip().lower() for host in extra_hosts_raw.split(",") if host.strip()
    }
    allowlist = default_allowlist | extra_allowlist

    host_normalized = hostname.strip().lower()
    host_allowed = host_normalized in allowlist
    if not host_allowed:
        try:
            host_allowed = ipaddress.ip_address(host_normalized).is_loopback
        except ValueError:
            host_allowed = False

    if not host_allowed:
        raise ValueError(
            "Server host is not allowed. Use localhost/loopback, "
            "or set TOKEN_WORKSHED_ALLOWED_SERVER_HOSTS."
        )

    netloc_host = hostname
    if ":" in netloc_host and not netloc_host.startswith("["):
        netloc_host = f"[{netloc_host}]"

    if parsed.port is not None:
        netloc = f"{netloc_host}:{parsed.port}"
    else:
        netloc = netloc_host

    return urllib.parse.urlunparse((parsed.scheme, netloc, "", "", "", "")).rstrip("/")


def _clamp_max_tokens(value: int | None, default: int) -> int:
    if value is None:
        return default
    return max(1, min(16384, int(value)))


def _clamp_temperature(value: float | None, default: float) -> float:
    if value is None:
        return default
    return max(0.0, min(2.0, float(value)))


def _extract_content_text(content: Any) -> str:
    """Extract plain text from OpenAI-style text or multimodal content."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text = part.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
        return "\n".join(part for part in text_parts if part)
    if isinstance(content, dict):
        text = content.get("text")
        if isinstance(text, str):
            return text
    return str(content)


def _try_parse_json_fragment(raw_text: str) -> Any | None:
    """Best-effort parse for noisy JSON fragments embedded in model text."""
    text = str(raw_text or "").strip()
    if not text:
        return None

    candidates: list[str] = [text]
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            inner = "\n".join(lines[1:-1]).strip()
            if inner:
                candidates.append(inner)

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

        lines = candidate.splitlines()
        for idx, line in enumerate(lines):
            stripped = line.lstrip()
            if not (stripped.startswith("{") or stripped.startswith("[")):
                continue
            joined = "\n".join(lines[idx:]).strip()
            if not joined:
                continue
            try:
                return json.loads(joined)
            except json.JSONDecodeError:
                continue

        for open_char, close_char in (("{", "}"), ("[", "]")):
            first = candidate.find(open_char)
            last = candidate.rfind(close_char)
            if first < 0 or last <= first:
                continue
            snippet = candidate[first : last + 1]
            try:
                return json.loads(snippet)
            except json.JSONDecodeError:
                continue

    return None


def _extract_agent_message_text(payload: Any, *, _depth: int = 0) -> str | None:
    """Extract user-visible `message` text from agent envelope payloads."""
    if _depth > 5:
        return None

    if isinstance(payload, str):
        parsed = _try_parse_json_fragment(payload)
        if parsed is None:
            return None
        return _extract_agent_message_text(parsed, _depth=_depth + 1)

    if isinstance(payload, list):
        for item in payload:
            extracted = _extract_agent_message_text(item, _depth=_depth + 1)
            if extracted:
                return extracted
        return None

    if not isinstance(payload, dict):
        return None

    def _decode_mapping(value: Any) -> dict[str, Any] | None:
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            parsed_value = _try_parse_json_fragment(value)
            if isinstance(parsed_value, dict):
                return parsed_value
        return None

    def _message_from_send_mapping(mapping: dict[str, Any]) -> str | None:
        message_value = mapping.get("message")
        if not isinstance(message_value, str):
            return None
        message_text = message_value.strip()
        if not message_text:
            return None
        if {"action", "target", "channel"} & set(mapping.keys()):
            return message_text
        return None

    if (
        isinstance(payload.get("name"), str)
        and payload["name"].strip().lower() == "message"
    ):
        params = _decode_mapping(payload.get("parameters"))
        if isinstance(params, dict):
            direct_message = params.get("message")
            if isinstance(direct_message, str) and direct_message.strip():
                return direct_message.strip()
            mapped_message = _message_from_send_mapping(params)
            if mapped_message:
                return mapped_message

    mapped_payload_message = _message_from_send_mapping(payload)
    if mapped_payload_message:
        return mapped_payload_message

    params = _decode_mapping(payload.get("parameters"))
    if isinstance(params, dict):
        mapped_message = _message_from_send_mapping(params)
        if mapped_message:
            return mapped_message

    function_payload = payload.get("function")
    if isinstance(function_payload, dict):
        function_name = str(function_payload.get("name", "")).strip().lower()
        function_args = _decode_mapping(function_payload.get("arguments"))
        if function_name == "message" and isinstance(function_args, dict):
            direct_message = function_args.get("message")
            if isinstance(direct_message, str) and direct_message.strip():
                return direct_message.strip()
            mapped_message = _message_from_send_mapping(function_args)
            if mapped_message:
                return mapped_message

    tool_calls = payload.get("tool_calls")
    if isinstance(tool_calls, list):
        for call in tool_calls:
            extracted = _extract_agent_message_text(call, _depth=_depth + 1)
            if extracted:
                return extracted

    for nested_key in ("result", "payload", "data", "response"):
        if nested_key not in payload:
            continue
        extracted = _extract_agent_message_text(
            payload.get(nested_key), _depth=_depth + 1
        )
        if extracted:
            return extracted

    return None


def _normalize_agent_visible_text(raw_text: str) -> str:
    """Unwrap tool-envelope JSON text into user-visible message text when possible."""
    if not isinstance(raw_text, str):
        return str(raw_text)
    extracted = _extract_agent_message_text(raw_text)
    if isinstance(extracted, str) and extracted.strip():
        return extracted.strip()
    return raw_text


def _extract_url_like(value: Any) -> str | None:
    """Extract URL/base64 payload from common multimodal fields."""
    if isinstance(value, str):
        url = value.strip()
        return url if url else None
    if isinstance(value, dict):
        raw_url = value.get("url")
        if isinstance(raw_url, str) and raw_url.strip():
            return raw_url.strip()
    return None


def _normalize_content_for_api(content: Any) -> str | list[dict[str, Any]] | None:
    """Normalize message content into OpenAI-compatible text or content-part list."""
    if isinstance(content, str):
        text = content.strip()
        return text if text else None

    if isinstance(content, dict):
        content = [content]

    if isinstance(content, list):
        parts: list[dict[str, Any]] = []
        for raw_part in content:
            part = raw_part
            if hasattr(part, "model_dump"):
                part = part.model_dump()
            elif hasattr(part, "dict"):
                part = part.dict()

            if not isinstance(part, dict):
                continue

            part_type = str(part.get("type", "")).strip().lower()
            if part_type == "text":
                text = part.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append({"type": "text", "text": text})
                continue

            if part_type == "image_url":
                image_url = _extract_url_like(part.get("image_url"))
                if image_url is None:
                    image_url = _extract_url_like(part.get("url"))
                if image_url:
                    parts.append({"type": "image_url", "image_url": {"url": image_url}})
                continue

            if part_type == "video_url":
                video_url = _extract_url_like(part.get("video_url"))
                if video_url is None:
                    video_url = _extract_url_like(part.get("url"))
                if video_url:
                    parts.append({"type": "video_url", "video_url": {"url": video_url}})
                continue

            if part_type == "image":
                image_url = _extract_url_like(part.get("image"))
                if image_url is None:
                    image_url = _extract_url_like(part.get("url"))
                if image_url:
                    parts.append({"type": "image", "image": image_url})
                continue

            if part_type == "video":
                video_url = _extract_url_like(part.get("video"))
                if video_url is None:
                    video_url = _extract_url_like(part.get("url"))
                if video_url:
                    parts.append({"type": "video", "video": video_url})
                continue

        return parts or None

    if content is None:
        return None

    # Fallback for unknown content types.
    text = str(content).strip()
    return text if text else None


def _extract_token_usage(
    data: dict[str, Any],
) -> tuple[int | None, int | None, int | None]:
    """Extract token usage counters from OpenAI-compatible response payload."""
    usage = data.get("usage")
    if not isinstance(usage, dict):
        return None, None, None

    prompt_tokens_raw = usage.get("prompt_tokens")
    completion_tokens_raw = usage.get("completion_tokens")
    total_tokens_raw = usage.get("total_tokens")

    prompt_tokens = (
        int(prompt_tokens_raw) if isinstance(prompt_tokens_raw, (int, float)) else None
    )
    completion_tokens = (
        int(completion_tokens_raw)
        if isinstance(completion_tokens_raw, (int, float))
        else None
    )
    total_tokens = (
        int(total_tokens_raw) if isinstance(total_tokens_raw, (int, float)) else None
    )
    return prompt_tokens, completion_tokens, total_tokens


def _is_title_model_id(model_id: str) -> bool:
    """Whether model id is reserved for lightweight conversation-title generation."""
    normalized = str(model_id or "").strip().lower()
    if not normalized:
        return False
    candidates = {item.strip().lower() for item in TITLE_MODEL_CANDIDATES if item}
    return normalized in candidates or normalized in _TITLE_MODEL_ALIASES


def _fetch_models(target: str, timeout: float = 8.0) -> tuple[list[str], str]:
    """Fetch model list from OpenAI-compatible /v1/models."""
    response = requests.get(f"{target}/v1/models", timeout=timeout)
    response.raise_for_status()
    data = response.json()

    models_raw = data.get("data") if isinstance(data, dict) else None
    models: list[str] = []
    if isinstance(models_raw, list):
        for item in models_raw:
            if isinstance(item, dict):
                model_id = item.get("id")
                if isinstance(model_id, str) and model_id.strip():
                    normalized_id = model_id.strip()
                    if _is_title_model_id(normalized_id):
                        continue
                    models.append(normalized_id)

    if not models:
        models = ["default"]

    default_model = models[0]
    return models, default_model


def _coerce_float(value: Any) -> float | None:
    """Convert numeric payload values to finite floats."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        result = float(value)
        if math.isfinite(result):
            return result
    return None


def _detect_reasoning_parser(model_id: str) -> str | None:
    """Infer best reasoning parser for a model ID."""
    normalized = model_id.strip().lower()
    if not normalized:
        return None

    if "qwen3" in normalized or "qwq" in normalized:
        return "qwen3"
    if "deepseek" in normalized and "r1" in normalized:
        return "deepseek_r1"
    if "gpt-oss" in normalized:
        return "gpt_oss"
    if "harmony" in normalized:
        return "harmony"
    return None


def _detect_model_capability(model_id: str) -> dict[str, Any]:
    """Infer model capability metadata for UI hints."""
    normalized = model_id.strip().lower()
    reasoning_parser = _detect_reasoning_parser(model_id)
    deep_thinking = bool(reasoning_parser) or any(
        token in normalized for token in ("nemotron", "reasoner", "thinking")
    )
    openclaw_supported = bool(normalized) and (not _is_title_model_id(model_id))
    openclaw_recommended = deep_thinking
    return {
        "deep_thinking": deep_thinking,
        "reasoning_parser": reasoning_parser,
        "openclaw_supported": openclaw_supported,
        "openclaw_recommended": openclaw_recommended,
    }


def _build_model_capabilities(models: list[str]) -> dict[str, dict[str, Any]]:
    """Build capability map keyed by model id."""
    return {model_id: _detect_model_capability(model_id) for model_id in models}


def _compose_system_prompt_for_model(
    system_prompt: str,
    selected_model: str,
) -> tuple[str, dict[str, Any]]:
    """Keep user/system prompt untouched; only return detected model capability."""
    capability = _detect_model_capability(selected_model)
    prompt = system_prompt.strip()
    return prompt, capability


def _compose_system_prompt_for_openclaw(system_prompt: str, enabled: bool) -> str:
    """Append OpenClaw guidance when toggle is enabled in chat UI."""
    prompt = system_prompt.strip()
    if not enabled:
        return prompt

    # Keep behavior stable even when integration sources are absent.
    if not OPENCLAW_DIR.exists():
        return prompt

    lowered = prompt.lower()
    if "openclaw mode is enabled" in lowered:
        return prompt
    if "prioritize secure, defensive, implementation-ready guidance" in lowered:
        return prompt

    if prompt:
        return f"{prompt}\n\n{_OPENCLAW_HINT}"
    return _OPENCLAW_HINT


def _tune_generation_for_model(
    *,
    max_tokens: int,
    temperature: float,
    model_capability: dict[str, Any],
) -> tuple[int, float]:
    """Preserve caller-provided generation parameters without model-specific overrides."""
    return max_tokens, temperature


def _stream_json_line(payload: dict[str, Any]) -> str:
    """Encode a streaming JSON event line."""
    return json.dumps(payload, ensure_ascii=False) + "\n"


def _fetch_runtime_metrics(target: str, timeout: float = 8.0) -> dict[str, Any]:
    """Fetch runtime + memory metrics from /v1/status and local system stats."""
    response = requests.get(f"{target}/v1/status", timeout=timeout)
    response.raise_for_status()
    data = response.json()

    if not isinstance(data, dict):
        return {}

    metal = data.get("metal")
    metal_data = metal if isinstance(metal, dict) else {}

    metal_active_gb = _coerce_float(metal_data.get("active_memory_gb"))
    metal_peak_gb = _coerce_float(metal_data.get("peak_memory_gb"))
    metal_cache_gb = _coerce_float(metal_data.get("cache_memory_gb"))

    system_total_gb: float | None = None
    system_used_gb: float | None = None
    system_pressure_pct: float | None = None
    try:
        import psutil

        vm = psutil.virtual_memory()
        system_total_gb = float(vm.total) / (1024.0**3)
        system_used_gb = float(vm.used) / (1024.0**3)
        system_pressure_pct = _coerce_float(vm.percent)
    except Exception:
        # Optional best-effort metric source.
        pass

    memory_usage_gb = metal_active_gb if metal_active_gb is not None else system_used_gb

    memory_pressure_pct: float | None = None
    if (
        memory_usage_gb is not None
        and system_total_gb is not None
        and system_total_gb > 0
    ):
        memory_pressure_pct = (memory_usage_gb / system_total_gb) * 100.0
    elif system_pressure_pct is not None:
        memory_pressure_pct = system_pressure_pct

    if memory_pressure_pct is not None:
        memory_pressure_pct = min(100.0, max(0.0, memory_pressure_pct))

    return {
        "status": data.get("status"),
        "memory_usage_gb": (
            round(memory_usage_gb, 2) if isinstance(memory_usage_gb, float) else None
        ),
        "memory_pressure_pct": (
            round(memory_pressure_pct, 1)
            if isinstance(memory_pressure_pct, float)
            else None
        ),
        "metal_active_memory_gb": (
            round(metal_active_gb, 2) if isinstance(metal_active_gb, float) else None
        ),
        "metal_peak_memory_gb": (
            round(metal_peak_gb, 2) if isinstance(metal_peak_gb, float) else None
        ),
        "metal_cache_memory_gb": (
            round(metal_cache_gb, 2) if isinstance(metal_cache_gb, float) else None
        ),
        "system_total_memory_gb": (
            round(system_total_gb, 2) if isinstance(system_total_gb, float) else None
        ),
    }


def _build_messages(
    history: list[dict[str, Any]],
    system_prompt: str,
    user_message: str,
    user_content: Any | None = None,
) -> list[dict[str, Any]]:
    """Build OpenAI-compatible message history."""
    messages: list[dict[str, Any]] = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    for item in history:
        if not isinstance(item, dict):
            continue

        role = str(item.get("role", "")).strip().lower()
        if role not in {"system", "user", "assistant"}:
            continue

        content = _normalize_content_for_api(item.get("content", ""))
        if content is not None:
            messages.append({"role": role, "content": content})

    normalized_user_content = _normalize_content_for_api(
        user_content if user_content is not None else user_message
    )
    if normalized_user_content is None:
        raise ValueError("Message cannot be empty.")

    messages.append({"role": "user", "content": normalized_user_content})
    return messages


def _truncate_error_text(value: str, limit: int = 1200) -> str:
    """Trim potentially large stderr/stdout fragments for API-safe errors."""
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit]}..."


def _normalize_openclaw_session_id(
    conversation_id: str | None,
    *,
    selected_model: str | None = None,
    runtime: str | None = None,
) -> str:
    """Convert UI conversation/model/runtime scope to safe agent session ids."""
    raw_conv = str(conversation_id or "").strip() or "main"
    raw_model = str(selected_model or "").strip() or "default"
    runtime_tag = (
        "hermes" if str(runtime or "").strip().lower() == "hermes" else "openclaw"
    )
    raw = f"{raw_conv}|{runtime_tag}|{raw_model}"

    normalized = _OPENCLAW_SESSION_ID_RE.sub("-", raw).strip("-.")
    if not normalized:
        normalized = "main"

    return f"token-workshed-{normalized[:96]}"


def _make_openclaw_run_id(session_id: str) -> str:
    """Create a safe per-turn OpenClaw run id for cancellation/diagnostics."""
    raw = f"{session_id}-run-{uuid.uuid4().hex[:12]}"
    normalized = _OPENCLAW_SESSION_ID_RE.sub("-", raw).strip("-.")
    return normalized[:140] or f"token-workshed-run-{uuid.uuid4().hex[:12]}"


def _count_non_text_parts(content: Any) -> int:
    """Count non-text multimodal parts from normalized OpenAI content payload."""
    if isinstance(content, dict):
        content = [content]
    if not isinstance(content, list):
        return 0

    count = 0
    for part in content:
        if not isinstance(part, dict):
            continue
        part_type = str(part.get("type", "")).strip().lower()
        if part_type and part_type != "text":
            count += 1
    return count


def _compose_openclaw_turn_message(
    *,
    system_prompt: str,
    user_message: str,
    user_content: Any | None,
    strict_tool_mode: bool = False,
) -> str:
    """Compose a text-only turn payload for OpenClaw CLI execution."""
    normalized_user_content = _normalize_content_for_api(
        user_content if user_content is not None else user_message
    )
    text = _extract_content_text(
        normalized_user_content if normalized_user_content is not None else user_message
    ).strip()
    if not text:
        text = str(user_message or "").strip()
    if not text:
        raise ValueError("Message cannot be empty.")

    sections: list[str] = []
    if system_prompt:
        sections.append(f"Token-workshed system prompt:\n{system_prompt}")

    sections.append(
        "OpenClaw execution mode: if the user asks to operate the computer, use available tools and report concrete actions."
    )
    sections.append(
        "Do not output literal tool-call markup (for example `<tool_call>...</tool_call>`). "
        "When a tool is needed, invoke it through the runtime and then report the result in normal text."
    )
    sections.append(
        "Do not output code blocks unless the user explicitly asks for code."
    )
    sections.append(
        "Never recite internal identity/persona manifesto text (for example `My core truths are` "
        "or numbered creed lists). Answer the user's request directly."
    )
    if not _has_openclaw_web_search_credential():
        sections.append(
            "Web-search API credentials are not configured in this runtime. "
            "If the user asks for online information, use browser/web_fetch style tool flow instead of asking for API-key setup."
        )
        sections.append(
            "Browser profile policy: prefer the OpenClaw-managed browser by default (omit profile). "
            'Do not use `profile="user"` unless the user explicitly asks for the logged-in host browser.'
        )
        sections.append(
            "If web_fetch reports `Blocked: resolves to private/internal/special-use IP address`, "
            "switch to browser-based flow immediately instead of retrying web_fetch."
        )
    if strict_tool_mode:
        sections.append(
            "Runtime capability reminder: this session has tool access through OpenClaw. "
            "For actionable requests, do not claim lack of filesystem/web access. "
            "Invoke available tools first (for example exec/read/write/browser/web_fetch/web_search when configured), "
            "then report concrete results."
        )
        sections.append(
            'Browser tool guidance: omit profile or use `profile="openclaw"`; '
            'avoid `profile="user"` unless explicitly requested by the user.'
        )
    sections.append(f"User request:\n{text}")

    media_parts = _count_non_text_parts(normalized_user_content)
    if media_parts > 0:
        sections.append(
            f"[token-workshed note] {media_parts} media attachment(s) were provided but are not forwarded to openclaw CLI turns."
        )

    return "\n\n".join(section for section in sections if section).strip()


def _compose_hermes_turn_message(
    *,
    system_prompt: str,
    user_message: str,
    user_content: Any | None,
    strict_tool_mode: bool = False,
) -> str:
    """Compose a Hermes-oriented turn payload for deterministic local execution."""
    normalized_user_content = _normalize_content_for_api(
        user_content if user_content is not None else user_message
    )
    text = _extract_content_text(
        normalized_user_content if normalized_user_content is not None else user_message
    ).strip()
    if not text:
        text = str(user_message or "").strip()
    if not text:
        raise ValueError("Message cannot be empty.")

    sections: list[str] = []
    if system_prompt:
        sections.append(f"Token-workshed system prompt:\n{system_prompt}")

    sections.append(
        "Hermes execution mode: use available tools when required, then provide concrete results."
    )
    sections.append(
        "Do not output literal tool-call markup (for example `<tool_call>...</tool_call>`). "
        "Invoke tools through runtime integration and report outcomes in normal text."
    )
    if strict_tool_mode:
        sections.append(
            "Runtime capability reminder: this session has tool access through Hermes. "
            "For actionable requests, do not claim lack of filesystem/web access. "
            "Invoke available tools first, then report concrete results."
        )
    sections.append(f"User request:\n{text}")

    media_parts = _count_non_text_parts(normalized_user_content)
    if media_parts > 0:
        sections.append(
            f"[token-workshed note] {media_parts} media attachment(s) were provided but are not forwarded to Hermes CLI turns."
        )

    return "\n\n".join(section for section in sections if section).strip()


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw not in {"0", "false", "off", "no", "n"}


def _resolve_openclaw_keep_alive(default_value: str = "10m") -> str:
    """Resolve keep-alive hint (Ollama-compatible style) from env."""
    raw = (
        os.environ.get("TOKEN_WORKSHED_OPENCLAW_KEEP_ALIVE", "").strip()
        or os.environ.get("OLLAMA_KEEP_ALIVE", "").strip()
    )
    return raw or default_value


def _resolve_openclaw_num_predict(default_tokens: int = 768) -> int:
    """Resolve per-turn generation cap for OpenClaw local model calls."""
    raw = os.environ.get("TOKEN_WORKSHED_OPENCLAW_NUM_PREDICT", "").strip()
    if not raw:
        return default_tokens
    try:
        parsed = int(raw)
    except ValueError:
        return default_tokens
    return max(128, min(4096, parsed))


def _resolve_generation_repetition_penalty(
    default_value: float = _OPENCLAW_DEFAULT_REPETITION_PENALTY,
) -> float:
    """Resolve repetition penalty for chat/gateway payloads."""
    raw = os.environ.get("TOKEN_WORKSHED_REPETITION_PENALTY", "").strip()
    if not raw:
        return float(default_value)
    try:
        parsed = float(raw)
    except ValueError:
        return float(default_value)
    if not math.isfinite(parsed):
        return float(default_value)
    return max(1.0, min(2.0, parsed))


def _build_identity_guard_stop_sequences() -> list[str]:
    """Stop sequences used to cut off repeated persona/identity recitals."""
    out: list[str] = []
    seen: set[str] = set()
    for raw in _OPENCLAW_IDENTITY_STOP_SEQUENCES:
        text = str(raw or "").strip()
        if not text:
            continue
        lowered = text.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        out.append(text)
    return out


def _build_generation_guard_sampling_overrides() -> dict[str, Any]:
    """Sampling overrides to reduce identity-prompt repetition loops."""
    return {
        "repetition_penalty": _resolve_generation_repetition_penalty(),
        "stop": _build_identity_guard_stop_sequences(),
    }


def _resolve_openclaw_warm_cache_ttl_seconds(default_seconds: int = 900) -> float:
    """Resolve warm-cache TTL to skip redundant preload work."""
    raw = os.environ.get("TOKEN_WORKSHED_OPENCLAW_WARM_CACHE_TTL_SECONDS", "").strip()
    if not raw:
        return float(default_seconds)
    try:
        parsed = float(raw)
    except ValueError:
        return float(default_seconds)
    return max(30.0, min(36000.0, parsed))


def _should_serialize_openclaw_runs(default_value: bool = False) -> bool:
    """Optional global serialization for debugging race conditions."""
    return _env_flag("TOKEN_WORKSHED_OPENCLAW_SERIALIZE_RUNS", default_value)


def _openclaw_runtime_cache_key(
    *, runtime_target: dict[str, str], config_revision: str
) -> str:
    return f"{runtime_target.get('runtime_key', '')}|cfg:{config_revision}"


def _is_openclaw_runtime_warm(cache_key: str) -> bool:
    """Whether runtime was preloaded recently enough to skip expensive warmup."""
    now = time.time()
    ttl = _resolve_openclaw_warm_cache_ttl_seconds()
    with _OPENCLAW_WARM_CACHE_LOCK:
        stale_keys = [
            key
            for key, warmed_at in _OPENCLAW_WARM_CACHE.items()
            if (now - float(warmed_at)) > ttl
        ]
        for key in stale_keys:
            _OPENCLAW_WARM_CACHE.pop(key, None)
        warmed_at = _OPENCLAW_WARM_CACHE.get(cache_key)
        return isinstance(warmed_at, float) and (now - warmed_at) <= ttl


def _mark_openclaw_runtime_warm(cache_key: str) -> None:
    with _OPENCLAW_WARM_CACHE_LOCK:
        _OPENCLAW_WARM_CACHE[cache_key] = float(time.time())


def _resolve_openclaw_browser_headless(default_value: bool = True) -> bool:
    """Resolve whether OpenClaw browser runs in headless mode."""
    return _env_flag("TOKEN_WORKSHED_OPENCLAW_BROWSER_HEADLESS", default_value)


def _resolve_openclaw_browser_default_profile(
    default_profile: str = _OPENCLAW_BROWSER_DEFAULT_PROFILE,
) -> str:
    """Resolve default browser profile with a strict safe-name filter."""
    raw = (
        os.environ.get("TOKEN_WORKSHED_OPENCLAW_BROWSER_DEFAULT_PROFILE", "")
        .strip()
        .lower()
    )
    if raw and re.fullmatch(r"[a-z0-9-]+", raw):
        return raw
    return default_profile


def _derive_openclaw_browser_profile_ports(*, gateway_port: int) -> tuple[int, int]:
    """Derive stable CDP ports for openclaw/user browser profiles."""
    control_port = gateway_port + 2
    if control_port < 1 or control_port > 65535:
        control_port = 18791
    default_cdp_port = control_port + 9
    if default_cdp_port < 1 or default_cdp_port > 65535:
        default_cdp_port = 18800
    user_cdp_port = default_cdp_port + 1
    if user_cdp_port < 1 or user_cdp_port > 65535:
        user_cdp_port = default_cdp_port
    return default_cdp_port, user_cdp_port


def _has_openclaw_web_search_credential() -> bool:
    """Best-effort credential probe for web_search providers."""
    env_candidates = (
        "WEB_SEARCH_API_KEY",
        "BRAVE_API_KEY",
        "WEB_SEARCH_GEMINI_API_KEY",
        "WEB_SEARCH_GROK_API_KEY",
        "WEB_SEARCH_KIMI_API_KEY",
        "WEB_SEARCH_PERPLEXITY_API_KEY",
        "GEMINI_API_KEY",
        "XAI_API_KEY",
        "KIMI_API_KEY",
        "PERPLEXITY_API_KEY",
    )
    return any(bool(os.environ.get(name, "").strip()) for name in env_candidates)


def _resolve_openclaw_skills_prompt_limits(
    *,
    default_max_skills: int = 80,
    default_max_chars: int = 14000,
) -> dict[str, int]:
    """Resolve non-zero skills prompt budget so tool capabilities remain visible."""

    def _parse(name: str, default: int, lower: int, upper: int) -> int:
        raw = os.environ.get(name, "").strip()
        if not raw:
            return default
        try:
            parsed = int(raw)
        except ValueError:
            return default
        return max(lower, min(upper, parsed))

    return {
        "maxSkillsInPrompt": _parse(
            "TOKEN_WORKSHED_OPENCLAW_MAX_SKILLS_IN_PROMPT",
            default_max_skills,
            1,
            400,
        ),
        "maxSkillsPromptChars": _parse(
            "TOKEN_WORKSHED_OPENCLAW_MAX_SKILLS_PROMPT_CHARS",
            default_max_chars,
            1000,
            120000,
        ),
    }


def _is_openclaw_non_answer_summary(value: str) -> bool:
    normalized = re.sub(r"[\s._-]+", " ", str(value or "").strip().lower())
    return normalized in _OPENCLAW_NON_ANSWER_SUMMARIES


def _normalize_ollama_base_url(raw_url: str) -> str:
    raw = str(raw_url or "").strip() or _OPENCLAW_OLLAMA_DEFAULT_BASE_URL
    parsed = urllib.parse.urlparse(raw)
    if not parsed.scheme:
        raw = f"http://{raw}"
        parsed = urllib.parse.urlparse(raw)
    if not parsed.netloc:
        raise ValueError(
            f"Invalid Ollama base URL: {raw_url!r}. Expected format like http://127.0.0.1:11434"
        )
    path = (parsed.path or "").rstrip("/")
    if path.lower().endswith("/v1"):
        path = path[:-3]
    normalized = urllib.parse.urlunparse(
        (
            parsed.scheme.lower(),
            parsed.netloc,
            path.rstrip("/"),
            "",
            "",
            "",
        )
    ).rstrip("/")
    return normalized or _OPENCLAW_OLLAMA_DEFAULT_BASE_URL


def _list_ollama_models(base_url: str) -> list[str]:
    probe_url = f"{base_url}/api/tags"
    response = requests.get(probe_url, timeout=3.0)
    response.raise_for_status()
    data = response.json()
    names: list[str] = []
    if isinstance(data, dict) and isinstance(data.get("models"), list):
        for item in data["models"]:
            if not isinstance(item, dict):
                continue
            model_name = str(item.get("name", "")).strip()
            if model_name:
                names.append(model_name)
    return names


def _normalize_ollama_model_id(value: str) -> str:
    normalized = str(value or "").strip()
    if normalized.lower().startswith("ollama/"):
        normalized = normalized.split("/", 1)[1].strip()
    return normalized


def _pick_ollama_model(*, preferred: str, base_url: str) -> str:
    preferred_model = _normalize_ollama_model_id(preferred)
    env_model = _normalize_ollama_model_id(
        os.environ.get("TOKEN_WORKSHED_OPENCLAW_OLLAMA_MODEL", "")
    )
    env_model_set = bool(env_model)
    if env_model:
        preferred_model = env_model

    installed: list[str] = []
    try:
        installed = _list_ollama_models(base_url)
    except Exception:
        # Keep explicit overrides working even if tag discovery is unavailable.
        if preferred_model:
            return preferred_model
        raise

    if preferred_model:
        exact = {name.lower(): name for name in installed}
        matched = exact.get(preferred_model.lower())
        if matched:
            return matched
        if env_model_set:
            return preferred_model
        preferred_model = ""

    lowered = [(name.lower(), name) for name in installed]
    for candidate in _OPENCLAW_OLLAMA_MODEL_CANDIDATES:
        wanted = candidate.lower()
        for lower_name, raw_name in lowered:
            if lower_name == wanted or lower_name.startswith(f"{wanted}:"):
                return raw_name
    if installed:
        return installed[0]
    raise RuntimeError("No Ollama models found. Run `ollama pull <model>` first.")


def _resolve_openclaw_backend_mode() -> str:
    raw = os.environ.get("TOKEN_WORKSHED_OPENCLAW_BACKEND", "").strip().lower()
    if raw in {"ollama"}:
        return "ollama"
    if raw in {"openai", "openai-compat", "vllm"}:
        return "openai-compat"
    return "openai-compat"


def _resolve_openclaw_runtime_target(
    *,
    server_url: str,
    selected_model: str,
    backend_override: str | None = None,
) -> dict[str, str]:
    override_raw = str(backend_override or "").strip().lower()
    override_backend: str | None = None
    if override_raw in {"ollama"}:
        override_backend = "ollama"
    elif override_raw in {"openai", "openai-compat", "vllm"}:
        override_backend = "openai-compat"

    backend = override_backend or _resolve_openclaw_backend_mode()

    if backend == "ollama":
        ollama_base_url = _normalize_ollama_base_url(
            os.environ.get("TOKEN_WORKSHED_OPENCLAW_OLLAMA_BASE_URL", "").strip()
            or _OPENCLAW_OLLAMA_DEFAULT_BASE_URL
        )
        model_id = _pick_ollama_model(
            preferred=selected_model,
            base_url=ollama_base_url,
        )
        provider_model = f"ollama/{model_id}"
        return {
            "backend": "ollama",
            "provider_id": "ollama",
            "provider_api": "ollama",
            "model_id": model_id,
            "provider_model": provider_model,
            "base_url": ollama_base_url,
            "runtime_key": f"ollama|{ollama_base_url}|{model_id}",
        }

    model_id = str(selected_model or "").strip() or "default"
    if model_id.lower().startswith("openai/"):
        model_id = model_id.split("/", 1)[1].strip() or "default"
    if model_id.lower().startswith("ollama/"):
        model_id = model_id.split("/", 1)[1].strip() or "default"

    # Optional compatibility switch for forcing native Ollama /api/chat transport.
    # Default stays on OpenAI-compatible /v1 to match direct backend A/B paths.
    # Explicit backend overrides remain authoritative.
    if override_backend is None and _env_flag(
        "TOKEN_WORKSHED_OPENCLAW_NATIVE_CHAT", False
    ):
        provider_model = f"ollama/{model_id}"
        return {
            "backend": "ollama",
            "provider_id": "ollama",
            "provider_api": "ollama",
            "model_id": model_id,
            "provider_model": provider_model,
            "base_url": server_url,
            "runtime_key": f"ollama-native|{server_url}|{model_id}",
        }

    provider_model = f"openai/{model_id}"
    return {
        "backend": "openai-compat",
        "provider_id": "openai",
        "provider_api": "openai-completions",
        "model_id": model_id,
        "provider_model": provider_model,
        "base_url": f"{server_url}/v1",
        "runtime_key": f"openai-compat|{server_url}|{model_id}",
    }


def _build_openclaw_runtime_config(
    *,
    runtime_target: dict[str, str],
    tool_mode: str = "none",
    allowed_tools: list[str] | None = None,
    agent_profile: dict[str, Any] | None = None,
    generation_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an isolated OpenClaw config for the resolved runtime backend."""
    model_id = runtime_target["model_id"]
    provider_model = runtime_target["provider_model"]
    provider_id = runtime_target["provider_id"]
    provider_api = runtime_target["provider_api"]
    provider_base_url = runtime_target["base_url"]
    context_window = (
        _profile_positive_int(agent_profile, runtime=None, key="context_window")
        or _resolve_local_model_context_window(model_id)
        or 16384
    )

    normalized_tool_mode = str(tool_mode or "none").strip().lower()
    if normalized_tool_mode not in {"none", "single", "small", "full"}:
        normalized_tool_mode = "none"
    tool_profile = "full" if normalized_tool_mode == "full" else "minimal"
    normalized_allowed_tools: list[str] = []
    if normalized_tool_mode in {"single", "small"}:
        seen_tools: set[str] = set()
        for raw_name in allowed_tools or []:
            name = str(raw_name or "").strip()
            lowered = name.lower()
            if not name or lowered in seen_tools:
                continue
            seen_tools.add(lowered)
            normalized_allowed_tools.append(name)

    generation_settings = _agent_generation_settings(
        agent_profile,
        generation_overrides,
    )
    profile_max_tokens = _coerce_int(
        generation_settings.get("max_tokens", generation_settings.get("max_new_tokens"))
    )
    provider_entry: dict[str, Any] = {
        "api": provider_api,
        "baseUrl": provider_base_url,
        "models": [
            {
                "id": model_id,
                "name": model_id,
                "reasoning": False,
                "input": ["text", "image"],
                "cost": {
                    "input": 0,
                    "output": 0,
                    "cacheRead": 0,
                    "cacheWrite": 0,
                },
                "contextWindow": context_window,
                "maxTokens": max(1, min(32768, profile_max_tokens or 2048)),
            }
        ],
    }
    profile_runtime = (
        agent_profile.get("openclaw") if isinstance(agent_profile, dict) else None
    )
    if isinstance(profile_runtime, dict) and normalized_tool_mode != "none":
        supports_tools = _profile_runtime_supports_tools(agent_profile, "openclaw")
        compat: dict[str, Any] = {"supportsTools": supports_tools}
        reasoning_parser = (
            str(profile_runtime.get("reasoning_parser") or "").strip().lower()
        )
        thinking_formats = {
            "qwen": "qwen",
            "qwen3": "qwen",
            "qwen3_coder": "qwen",
            "deepseek": "deepseek",
            "zai": "zai",
            "openai": "openai",
        }
        if reasoning_parser in thinking_formats:
            compat["thinkingFormat"] = thinking_formats[reasoning_parser]
        provider_entry["models"][0]["compat"] = compat
    if normalized_tool_mode == "none":
        provider_entry["models"][0]["compat"] = {"supportsTools": False}
    if provider_id == "openai":
        provider_entry["authHeader"] = False
    if provider_id == "ollama":
        # Any placeholder key works for local Ollama availability checks.
        provider_entry["apiKey"] = "ollama-local"

    model_entry: dict[str, Any] = {
        "alias": "token-workshed-local",
    }
    model_params = _build_generation_guard_sampling_overrides()
    if generation_settings:
        for key in ("temperature", "top_p", "top_k", "typical_p", "repetition_penalty"):
            value = generation_settings.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                model_params[key] = value
    if provider_api == "ollama":
        keep_alive = _resolve_openclaw_keep_alive()
        num_predict = _resolve_openclaw_num_predict()
        model_params.update(
            {
                "keep_alive": keep_alive,
                "num_predict": num_predict,
                "temperature": 0.2,
            }
        )
    model_entry["params"] = model_params
    profile_timeout = _profile_positive_int(
        agent_profile,
        runtime="openclaw",
        key="timeout_seconds",
    )
    # The resolver still reads TOKEN_WORKSHED_OPENCLAW_TIMEOUT_SECONDS first,
    # so operational environment overrides retain precedence over the cache.
    timeout_seconds = (
        _resolve_openclaw_timeout_seconds(profile_timeout)
        if profile_timeout is not None
        else _resolve_openclaw_timeout_seconds()
    )
    gateway_port = _resolve_openclaw_gateway_port()
    _, user_browser_cdp_port = _derive_openclaw_browser_profile_ports(
        gateway_port=gateway_port
    )
    browser_default_profile = _resolve_openclaw_browser_default_profile()
    web_search_enabled = _has_openclaw_web_search_credential()

    config: dict[str, Any] = {
        "gateway": {
            "mode": "local",
            "auth": {
                "mode": "none",
            },
        },
        "browser": {
            "enabled": True,
            # Keep local browser automation available without forcing a visible Chromium window.
            "headless": _resolve_openclaw_browser_headless(True),
            "defaultProfile": browser_default_profile,
            "ssrfPolicy": {
                "dangerouslyAllowPrivateNetwork": True,
            },
            "profiles": {
                "user": {
                    "driver": "openclaw",
                    "cdpPort": user_browser_cdp_port,
                    "color": _OPENCLAW_BROWSER_USER_PROFILE_COLOR,
                }
            },
        },
        "agents": {
            "defaults": {
                "workspace": str(OPENCLAW_WORKSPACE_DIR),
                "repoRoot": str(TOKEN_WORKSHED_ROOT),
                "skipBootstrap": True,
                "timeoutSeconds": timeout_seconds,
                "bootstrapMaxChars": 1200,
                "bootstrapTotalMaxChars": 3600,
                "bootstrapPromptTruncationWarning": "off",
                "model": {
                    "primary": provider_model,
                },
                "models": {provider_model: model_entry},
            }
        },
        "models": {
            "mode": "merge",
            "providers": {provider_id: provider_entry},
        },
        "tools": {
            "profile": tool_profile,
            "web": {
                "search": {
                    "enabled": web_search_enabled,
                },
                "fetch": {
                    "enabled": True,
                },
            },
        },
    }
    if normalized_allowed_tools:
        config["tools"]["alsoAllow"] = normalized_allowed_tools
    skills_limits = _resolve_openclaw_skills_prompt_limits()
    if normalized_tool_mode == "none":
        skills_limits["maxSkillsInPrompt"] = 0
        skills_limits["maxSkillsPromptChars"] = 0
    config["skills"] = {"limits": skills_limits}
    return config


def _resolve_local_model_context_window(model_id: str) -> int | None:
    """Read the native model context without downloading remote metadata."""
    normalized_model = str(model_id or "").strip()
    if not normalized_model:
        return None
    try:
        metadata = read_model_metadata(normalized_model, allow_download=False)
    except Exception as exc:
        logger.debug(
            "Unable to read context metadata for %s: %s", normalized_model, exc
        )
        return None
    context_window = _coerce_int(metadata.get("max_context_length"))
    if context_window is None or context_window <= 0:
        return None
    return min(context_window, 1_048_576)


def _ensure_openclaw_runtime_config(
    *,
    runtime_target: dict[str, str],
    tool_mode: str = "none",
    allowed_tools: list[str] | None = None,
    agent_profile: dict[str, Any] | None = None,
    generation_overrides: dict[str, Any] | None = None,
) -> str:
    """Write/update isolated config and return a short revision fingerprint."""
    config_data = _build_openclaw_runtime_config(
        runtime_target=runtime_target,
        tool_mode=tool_mode,
        allowed_tools=allowed_tools,
        agent_profile=agent_profile,
        generation_overrides=generation_overrides,
    )
    serialized = json.dumps(config_data, ensure_ascii=False, indent=2) + "\n"
    config_revision = hashlib.sha1(serialized.encode("utf-8")).hexdigest()[:12]

    with _OPENCLAW_CONFIG_LOCK:
        OPENCLAW_STATE_DIR.mkdir(parents=True, exist_ok=True)
        OPENCLAW_WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
        if OPENCLAW_CONFIG_PATH.exists():
            try:
                existing = OPENCLAW_CONFIG_PATH.read_text(encoding="utf-8")
            except Exception:
                existing = ""
            if existing == serialized:
                return config_revision
        OPENCLAW_CONFIG_PATH.write_text(serialized, encoding="utf-8")
    return config_revision


def _parse_openclaw_json_output(raw_stdout: str) -> dict[str, Any]:
    """Extract structured JSON payload from OpenClaw stdout with noisy log prefixes."""
    text = str(raw_stdout or "").strip()
    if not text:
        raise RuntimeError("OpenClaw returned no output.")

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    lines = text.splitlines()
    for idx, line in enumerate(lines):
        if not line.lstrip().startswith("{"):
            continue
        candidate = "\n".join(lines[idx:]).strip()
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace >= 0 and last_brace > first_brace:
        candidate = text[first_brace : last_brace + 1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    raise RuntimeError(
        "OpenClaw returned non-JSON output: " f"{_truncate_error_text(text, limit=300)}"
    )


def _extract_openclaw_text_and_meta(
    payload: dict[str, Any],
) -> tuple[str, dict[str, Any], str]:
    """Extract assistant text, meta payload, and stop reason from OpenClaw response."""
    result = payload.get("result")
    envelope = result if isinstance(result, dict) else payload
    meta = envelope.get("meta") if isinstance(envelope, dict) else None
    meta_dict = meta if isinstance(meta, dict) else {}

    payloads = envelope.get("payloads") if isinstance(envelope, dict) else None
    chunks: list[str] = []
    seen_chunk_keys: set[str] = set()

    def _chunk_key(value: str) -> str:
        return re.sub(r"\s+", " ", str(value or "").strip().lower())

    def _is_empty_placeholder_chunk(value: str) -> bool:
        normalized = _chunk_key(value)
        return normalized in {
            "(empty response)",
            "empty response",
            "__vllm_mlx_empty_response__",
        }

    def _append_chunk(value: Any) -> None:
        if not isinstance(value, str):
            return
        text = _normalize_agent_visible_text(value).strip()
        if not text:
            return
        key = _chunk_key(text)
        if not key:
            return
        if key in seen_chunk_keys:
            return
        # Keep at most one placeholder chunk, and prefer real text when present.
        if _is_empty_placeholder_chunk(text) and any(
            not _is_empty_placeholder_chunk(existing) for existing in chunks
        ):
            return
        if chunks and chunks[-1] == text:
            return
        seen_chunk_keys.add(key)
        chunks.append(text)

    def _append_content_parts(parts: Any) -> None:
        """Join fragmented contentParts without injecting artificial line breaks."""
        if not isinstance(parts, list):
            return
        fragments: list[str] = []
        for part in parts:
            if isinstance(part, dict):
                for key in (
                    "thinking",
                    "text",
                    "content",
                    "answer",
                    "final",
                    "finalAnswer",
                ):
                    value = part.get(key)
                    if isinstance(value, str) and value:
                        fragments.append(value)
            elif isinstance(part, str) and part:
                fragments.append(part)

        stitched = "".join(fragments).strip()
        if stitched:
            _append_chunk(stitched)

    def _coalesce_micro_chunks(items: list[str]) -> list[str]:
        """Collapse pathological tokenized outputs like ['h','e','l','l','o']."""
        if len(items) < 8:
            return items
        non_empty = [item for item in items if item.strip()]
        if len(non_empty) < 8:
            return items
        micro_count = sum(1 for item in non_empty if len(item) <= 3)
        if (micro_count / len(non_empty)) < 0.7:
            return items
        stitched = "".join(non_empty).strip()
        if not stitched:
            return items
        return [stitched]

    if isinstance(payloads, list):
        for item in payloads:
            if not isinstance(item, dict):
                continue
            _append_chunk(item.get("text"))
            _append_chunk(item.get("content"))
            _append_chunk(item.get("thinking"))
            _append_chunk(item.get("answer"))
            _append_chunk(item.get("final"))
            _append_chunk(item.get("finalAnswer"))
            parts = item.get("contentParts")
            _append_content_parts(parts)
            media_urls = item.get("mediaUrls")
            if isinstance(media_urls, list):
                for media in media_urls:
                    _append_chunk(media)
            media_url = item.get("mediaUrl")
            _append_chunk(media_url)

    _append_chunk(envelope.get("text") if isinstance(envelope, dict) else None)
    _append_chunk(envelope.get("content") if isinstance(envelope, dict) else None)
    _append_chunk(envelope.get("thinking") if isinstance(envelope, dict) else None)
    _append_chunk(envelope.get("answer") if isinstance(envelope, dict) else None)
    _append_chunk(envelope.get("final") if isinstance(envelope, dict) else None)
    _append_chunk(envelope.get("finalAnswer") if isinstance(envelope, dict) else None)

    if not chunks:
        summary_candidates = [
            envelope.get("summary") if isinstance(envelope, dict) else None,
            payload.get("summary"),
            meta_dict.get("summary"),
        ]
        for summary in summary_candidates:
            if not isinstance(summary, str):
                continue
            text = summary.strip()
            if not text:
                continue
            if _is_openclaw_non_answer_summary(text):
                continue
            chunks.append(text)
            break

    stop_reason = ""
    raw_stop_reason = meta_dict.get("stopReason")
    if isinstance(raw_stop_reason, str):
        stop_reason = raw_stop_reason.strip().lower()
    elif isinstance(payload.get("status"), str):
        stop_reason = str(payload["status"]).strip().lower()

    chunks = _coalesce_micro_chunks(chunks)
    return "\n\n".join(chunks).strip(), meta_dict, stop_reason


def _split_openclaw_reasoning_and_answer(raw_text: str) -> tuple[str, str]:
    """Split combined reasoning+answer text into separate thinking and final answer segments."""
    text = str(raw_text or "").replace("\r\n", "\n").strip()
    if not text:
        return "", ""

    def _strip_final_header(value: str) -> str:
        return LEADING_FINAL_ANSWER_RE.sub("", value, count=1).strip()

    if THINK_OPEN_TAG_RE.search(text):
        thinking_chunks: list[str] = []
        answer_chunks: list[str] = []
        cursor = 0
        while cursor < len(text):
            open_match = THINK_OPEN_TAG_RE.search(text, cursor)
            if not open_match:
                answer_chunks.append(text[cursor:])
                break
            answer_chunks.append(text[cursor : open_match.start()])
            close_match = THINK_CLOSE_TAG_RE.search(text, open_match.end())
            if close_match:
                reasoning = text[open_match.end() : close_match.start()].strip()
                if reasoning:
                    thinking_chunks.append(reasoning)
                cursor = close_match.end()
                continue
            reasoning = text[open_match.end() :].strip()
            if reasoning:
                thinking_chunks.append(reasoning)
            cursor = len(text)

        thinking_text = "\n\n".join(chunk for chunk in thinking_chunks if chunk).strip()
        answer_text = "".join(answer_chunks).strip()
        marker = FINAL_ANSWER_MARKER_RE.search(answer_text)
        if marker:
            leading = answer_text[: marker.start()].strip()
            if leading:
                if thinking_text:
                    thinking_text = f"{thinking_text}\n\n{leading}".strip()
                else:
                    thinking_text = leading
            answer_text = answer_text[marker.end() :].strip()
        answer_text = _strip_final_header(answer_text)
        return thinking_text, answer_text

    marker = FINAL_ANSWER_MARKER_RE.search(text)
    if marker:
        thinking_text = THINKING_HEADER_RE.sub(
            "", text[: marker.start()].strip(), count=1
        ).strip()
        answer_text = _strip_final_header(text[marker.end() :].strip())
        return thinking_text, answer_text

    if THINKING_HEADER_RE.match(text):
        return THINKING_HEADER_RE.sub("", text, count=1).strip(), ""

    return "", _strip_final_header(text)


def _normalize_output_similarity_key(value: str) -> str:
    """Normalize text into a comparison key for repetition filtering."""
    lowered = str(value or "").strip().lower()
    if not lowered:
        return ""
    lowered = re.sub(r"[^\w\u4e00-\u9fff]+", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def _is_persona_manifesto_text(value: str) -> bool:
    """Heuristic detection for internal persona/identity manifesto text."""
    key = _normalize_output_similarity_key(value)
    if not key:
        return False
    if any(marker in key for marker in _OPENCLAW_PERSONA_MANIFESTO_MARKERS):
        return True
    hits = 0
    for marker in _OPENCLAW_PERSONA_LIST_LINE_MARKERS:
        if marker in key:
            hits += 1
    return hits >= 2


def _sanitize_assistant_output_text(raw_text: str) -> str:
    """Remove persona-manifesto leakage and repeated paragraphs from output text."""
    text = str(raw_text or "").replace("\r\n", "\n").strip()
    if not text:
        return ""

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", text) if part.strip()]
    kept_paragraphs: list[str] = []

    for paragraph in paragraphs:
        if _is_persona_manifesto_text(paragraph):
            continue
        kept_lines: list[str] = []
        for raw_line in paragraph.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if _is_persona_manifesto_text(line):
                continue
            if re.match(r"^\s*(?:[-*]|\d+[.)])\s+", line):
                lowered = line.lower()
                if any(
                    marker in lowered for marker in _OPENCLAW_PERSONA_LIST_LINE_MARKERS
                ):
                    continue
            kept_lines.append(raw_line.rstrip())
        candidate = "\n".join(kept_lines).strip()
        if candidate:
            kept_paragraphs.append(candidate)

    if not kept_paragraphs:
        if _is_persona_manifesto_text(text):
            return "已省略身份模板文案。请继续提出你的具体任务。"
        return text

    deduped: list[str] = []
    seen: set[str] = set()
    for paragraph in kept_paragraphs:
        key = _normalize_output_similarity_key(paragraph)
        if len(key) >= 8 and key in seen:
            continue
        if key:
            seen.add(key)
        deduped.append(paragraph)

    return "\n\n".join(deduped).strip()


def _synthesize_answer_from_thinking(thinking_text: str) -> str:
    """Best-effort synthesis when model returned reasoning without explicit final answer."""
    raw = str(thinking_text or "").strip()
    if not raw:
        return ""

    marker = FINAL_ANSWER_MARKER_RE.search(raw)
    if marker:
        candidate = LEADING_FINAL_ANSWER_RE.sub(
            "", raw[marker.end() :].strip(), count=1
        ).strip()
        if candidate:
            return candidate

    normalized = raw.replace("\r\n", "\n")
    cleaned_lines: list[str] = []
    for line in normalized.splitlines():
        text = line.strip()
        if not text:
            cleaned_lines.append("")
            continue
        if re.match(r"(?i)^\[(?:openclaw|cypherclaw|hermes)\]", text):
            continue
        cleaned_lines.append(text)
    compact = "\n".join(cleaned_lines).strip()
    if not compact:
        return ""

    paragraphs = [
        part.strip() for part in re.split(r"\n\s*\n+", compact) if part.strip()
    ]
    if not paragraphs:
        return ""
    tail = paragraphs[-1]

    # Avoid returning obvious analysis/control fragments.
    lowered = tail.lower()
    if any(
        token in lowered
        for token in (
            "let me",
            "i'll",
            "i will",
            "tool was blocked",
            "using the",
            "search for information",
        )
    ):
        for part in reversed(paragraphs[:-1]):
            probe = part.strip()
            if not probe:
                continue
            lowered_probe = probe.lower()
            if any(
                token in lowered_probe
                for token in (
                    "let me",
                    "i'll",
                    "i will",
                    "tool was blocked",
                    "using the",
                    "search for information",
                )
            ):
                continue
            return probe
        return ""

    return tail


def _coerce_int(value: Any) -> int | None:
    """Convert numeric values to int when finite."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        as_float = float(value)
        if math.isfinite(as_float):
            return int(as_float)
    return None


def _resolve_openclaw_timeout_seconds(default_seconds: int = 240) -> int:
    """Resolve command timeout from env, keeping sensible CLI-safe bounds."""
    raw = os.environ.get("TOKEN_WORKSHED_OPENCLAW_TIMEOUT_SECONDS", "").strip()
    if not raw:
        return default_seconds
    try:
        parsed = int(raw)
    except ValueError:
        return default_seconds
    return max(30, min(3600, parsed))


def _resolve_openclaw_timeout_retry_seconds(base_seconds: int) -> int:
    """Adaptive second-chance timeout when first OpenClaw turn times out."""
    raw = os.environ.get("TOKEN_WORKSHED_OPENCLAW_TIMEOUT_RETRY_SECONDS", "").strip()
    if raw:
        try:
            parsed = int(raw)
        except ValueError:
            parsed = 0
        if parsed > 0:
            return max(30, min(3600, parsed))

    base = max(30, int(base_seconds or 0))
    # Retry with a meaningfully larger budget while keeping an upper bound.
    return max(90, min(420, base + max(60, base // 2)))


def _resolve_hermes_timeout_seconds(default_seconds: int = 240) -> int:
    """Resolve Hermes command timeout from env with CLI-safe bounds."""
    raw = os.environ.get("TOKEN_WORKSHED_HERMES_TIMEOUT_SECONDS", "").strip()
    if not raw:
        return default_seconds
    try:
        parsed = int(raw)
    except ValueError:
        return default_seconds
    return max(30, min(3600, parsed))


def _resolve_hermes_max_turns(default_turns: int = 10) -> int:
    """Resolve Hermes max-turns from env with practical safety bounds."""
    raw = os.environ.get("TOKEN_WORKSHED_HERMES_MAX_TURNS", "").strip()
    if not raw:
        return default_turns
    try:
        parsed = int(raw)
    except ValueError:
        return default_turns
    return max(1, min(120, parsed))


def _resolve_backend_connect_retry_seconds(default_seconds: int = 35) -> int:
    """Resolve how long to retry local backend connect failures before surfacing error."""
    raw = os.environ.get("TOKEN_WORKSHED_BACKEND_CONNECT_RETRY_SECONDS", "").strip()
    if not raw:
        return default_seconds
    try:
        parsed = int(raw)
    except ValueError:
        return default_seconds
    return max(0, min(120, parsed))


class _BackendConnectRetryError(requests.exceptions.ConnectionError):
    """Raised when local backend stays unreachable after retry window."""

    def __init__(self, attempts: int, last_error: Exception | None = None) -> None:
        self.attempts = max(1, int(attempts))
        self.last_error = last_error
        detail = ""
        if last_error is not None:
            detail = f" Last error: {last_error}"
        super().__init__(
            f"Cannot connect to token-workshed backend server after {self.attempts} attempts.{detail}"
        )


def _wait_backend_ready(
    *,
    target: str,
    max_wait_seconds: float,
    probe_timeout_seconds: float = 1.5,
) -> bool:
    """Poll backend readiness via /v1/models during transient startup windows."""
    deadline = time.perf_counter() + max(0.0, float(max_wait_seconds))
    while time.perf_counter() < deadline:
        try:
            response = requests.get(
                f"{target}/v1/models",
                timeout=max(0.3, float(probe_timeout_seconds)),
            )
            if response.ok:
                return True
        except requests.RequestException:
            pass
        time.sleep(0.35)
    return False


def _post_chat_completions_with_connect_retry(
    *,
    target: str,
    payload: dict[str, Any],
    timeout_seconds: int,
) -> tuple[requests.Response, int]:
    """POST /v1/chat/completions with transient local connect-retry window."""
    retry_window = _resolve_backend_connect_retry_seconds()
    deadline = time.perf_counter() + max(0, retry_window)
    attempts = 0
    last_error: Exception | None = None
    startup_probe_done = False
    while True:
        attempts += 1
        try:
            response = requests.post(
                f"{target}/v1/chat/completions",
                json=payload,
                timeout=(3.0, max(10, int(timeout_seconds))),
            )
            return response, attempts
        except requests.exceptions.ConnectionError as exc:
            last_error = exc
            if not startup_probe_done and retry_window > 0:
                startup_probe_done = True
                _wait_backend_ready(
                    target=target,
                    max_wait_seconds=min(6.0, float(retry_window)),
                )
            if retry_window <= 0 or time.perf_counter() >= deadline:
                raise _BackendConnectRetryError(
                    attempts=attempts, last_error=last_error
                ) from exc
            time.sleep(min(0.35 * attempts, 1.5))


def _open_stream_chat_completions_with_connect_retry(
    *,
    target: str,
    payload: dict[str, Any],
    read_timeout_seconds: int,
) -> tuple[requests.Response, int]:
    """Open streaming /v1/chat/completions connection with connect-retry."""
    retry_window = _resolve_backend_connect_retry_seconds()
    deadline = time.perf_counter() + max(0, retry_window)
    attempts = 0
    last_error: Exception | None = None
    startup_probe_done = False
    while True:
        attempts += 1
        try:
            response = requests.post(
                f"{target}/v1/chat/completions",
                json=payload,
                timeout=(3.0, max(15, int(read_timeout_seconds))),
                stream=True,
            )
            return response, attempts
        except requests.exceptions.ConnectionError as exc:
            last_error = exc
            if not startup_probe_done and retry_window > 0:
                startup_probe_done = True
                _wait_backend_ready(
                    target=target,
                    max_wait_seconds=min(6.0, float(retry_window)),
                )
            if retry_window <= 0 or time.perf_counter() >= deadline:
                raise _BackendConnectRetryError(
                    attempts=attempts, last_error=last_error
                ) from exc
            time.sleep(min(0.35 * attempts, 1.5))


def _run_direct_chat_completion(
    *,
    target: str,
    selected_model: str,
    history: list[dict[str, Any]],
    system_prompt: str,
    user_message: str,
    user_content: Any | None,
    max_tokens: int,
    temperature: float,
    timeout_seconds: int = 300,
) -> dict[str, Any]:
    """Run one direct backend chat-completion turn (non-agent path)."""
    messages = _build_messages(
        history,
        system_prompt,
        user_message,
        user_content=user_content,
    )

    request_start = time.perf_counter()
    request_payload = {
        "model": selected_model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    request_payload.update(_build_generation_guard_sampling_overrides())
    response, connect_attempts = _post_chat_completions_with_connect_retry(
        target=target,
        payload=request_payload,
        timeout_seconds=timeout_seconds,
    )
    response.raise_for_status()
    data = response.json()
    latency_ms = (time.perf_counter() - request_start) * 1000.0

    choices = data.get("choices") if isinstance(data, dict) else None
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("Invalid response: missing choices.")

    first = choices[0]
    if not isinstance(first, dict):
        raise RuntimeError("Invalid response: malformed choice.")

    finish_reason = ""
    raw_finish_reason = first.get("finish_reason")
    if isinstance(raw_finish_reason, str):
        finish_reason = raw_finish_reason.strip().lower()

    message_payload = first.get("message")
    if not isinstance(message_payload, dict):
        raise RuntimeError("Invalid response: missing message.")

    assistant_text = _extract_content_text(message_payload.get("content", "")).strip()
    assistant_text = _sanitize_assistant_output_text(assistant_text)
    if not assistant_text:
        assistant_text = "(empty response)"

    prompt_tokens, completion_tokens, total_tokens = _extract_token_usage(data)
    tokens_per_second: float | None = None
    if completion_tokens is not None and latency_ms > 0:
        tokens_per_second = completion_tokens / (latency_ms / 1000.0)

    return {
        "reply": assistant_text,
        "model": data.get("model", "default"),
        "finish_reason": finish_reason or None,
        "stop_reason": finish_reason or None,
        "metrics": {
            "latency_ms": round(latency_ms, 2),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "connect_retry_attempts": int(connect_attempts),
            "tokens_per_second": (
                round(tokens_per_second, 2)
                if isinstance(tokens_per_second, float)
                else None
            ),
        },
    }


def _direct_chat_exception_status_code(exc: Exception) -> int:
    if isinstance(exc, ValueError):
        return 400
    if isinstance(exc, requests.exceptions.Timeout):
        return 504
    return 502


def _direct_chat_exception_detail(exc: Exception) -> str:
    if isinstance(exc, _BackendConnectRetryError):
        return (
            "Cannot connect to token-workshed backend server. "
            f"(retried {exc.attempts}x)"
        )
    if isinstance(exc, requests.exceptions.ConnectionError):
        return (
            "Cannot connect to token-workshed backend server. "
            "Make sure it is running and reachable."
        )
    if isinstance(exc, requests.exceptions.Timeout):
        return "Model response timed out."
    if isinstance(exc, requests.exceptions.HTTPError):
        detail = f"Upstream error: {exc}"
        response = getattr(exc, "response", None)
        if response is not None:
            try:
                body = response.text
                if body:
                    detail = f"{detail} | {body[:500]}"
            except Exception:
                pass
        return detail
    if isinstance(exc, requests.RequestException):
        return str(exc)
    return str(exc) or "Unknown direct chat failure."


def _resolve_openclaw_gateway_port(
    default_port: int = _OPENCLAW_GATEWAY_DEFAULT_PORT,
) -> int:
    """Resolve persistent gateway port with safe bounds for loopback service."""
    raw = os.environ.get("TOKEN_WORKSHED_OPENCLAW_GATEWAY_PORT", "").strip()
    if not raw:
        return default_port
    try:
        parsed = int(raw)
    except ValueError:
        return default_port
    return max(1024, min(65535, parsed))


def _resolve_openclaw_gateway_startup_timeout_seconds(
    default_seconds: float = _OPENCLAW_GATEWAY_STARTUP_TIMEOUT_DEFAULT_SECONDS,
) -> float:
    """Resolve startup health timeout for persistent OpenClaw gateway process."""
    raw = os.environ.get(
        "TOKEN_WORKSHED_OPENCLAW_GATEWAY_STARTUP_TIMEOUT_SECONDS",
        "",
    ).strip()
    if not raw:
        return float(default_seconds)
    try:
        parsed = float(raw)
    except ValueError:
        return float(default_seconds)
    if not math.isfinite(parsed):
        return float(default_seconds)
    return max(8.0, min(180.0, parsed))


def _build_openclaw_subprocess_env() -> dict[str, str]:
    """Build stable runtime env shared by OpenClaw CLI invocations."""
    env = os.environ.copy()
    env["OPENCLAW_STATE_DIR"] = str(OPENCLAW_STATE_DIR)
    env["OPENCLAW_CONFIG_PATH"] = str(OPENCLAW_CONFIG_PATH)
    env.setdefault("OPENCLAW_HIDE_BANNER", "1")
    env.setdefault("OPENCLAW_RUNNER_LOG", "0")
    env.setdefault("NO_COLOR", "1")
    env.setdefault("FORCE_COLOR", "0")
    return env


def _resolve_openclaw_agent_capabilities(
    *,
    launcher_base: list[str],
) -> frozenset[str]:
    """Detect optional agent flags so bridge calls survive OpenClaw CLI changes."""
    cache_key = "\x1f".join(str(item) for item in launcher_base)
    with _OPENCLAW_AGENT_HELP_LOCK:
        cached = _OPENCLAW_AGENT_CAPABILITIES_CACHE.get(cache_key)
        if cached is not None:
            return cached

    supported_flags = (
        "--run-id",
        "--tool-mode",
        "--allowed-tool",
        "--model-first-chunk-timeout-ms",
    )
    try:
        completed = subprocess.run(
            [*launcher_base, "agent", "--help"],
            cwd=str(OPENCLAW_DIR),
            env=_build_openclaw_subprocess_env(),
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        help_text = f"{completed.stdout}\n{completed.stderr}"
        capabilities = frozenset(flag for flag in supported_flags if flag in help_text)
    except (OSError, subprocess.SubprocessError):
        capabilities = frozenset()

    with _OPENCLAW_AGENT_HELP_LOCK:
        _OPENCLAW_AGENT_CAPABILITIES_CACHE[cache_key] = capabilities
    return capabilities


def _openai_base_url_from_server_url(server_url: str) -> str:
    normalized = str(server_url or "").strip().rstrip("/")
    if not normalized:
        return "http://127.0.0.1:8000/v1"
    if normalized.endswith("/v1"):
        return normalized
    return f"{normalized}/v1"


def _sync_hermes_home_env(*, local_openai_base: str) -> None:
    """Write deterministic Hermes home .env so local endpoint always wins."""
    HERMES_HOME_DIR.mkdir(parents=True, exist_ok=True)
    env_path = HERMES_HOME_DIR / ".env"
    desired_lines = [
        f"OPENAI_BASE_URL={local_openai_base}",
        f"OPENROUTER_BASE_URL={local_openai_base}",
        "OPENAI_API_KEY=token-workshed-local",
        "HERMES_INFERENCE_PROVIDER=custom",
        "HERMES_QUIET=1",
        "NO_COLOR=1",
        "FORCE_COLOR=0",
        "PIP_DISABLE_PIP_VERSION_CHECK=1",
    ]
    desired_content = "\n".join(desired_lines) + "\n"
    with _HERMES_ENV_LOCK:
        current = ""
        if env_path.exists():
            try:
                current = env_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                current = ""
        if current != desired_content:
            env_path.write_text(desired_content, encoding="utf-8")


def _sync_hermes_runtime_config(
    *,
    server_url: str,
    selected_model: str,
    agent_profile: dict[str, Any] | None = None,
    generation_overrides: dict[str, Any] | None = None,
) -> None:
    """Bind Hermes to the selected local model and its native context window."""
    HERMES_HOME_DIR.mkdir(parents=True, exist_ok=True)
    local_openai_base = _openai_base_url_from_server_url(server_url)
    model_id = str(selected_model or "").strip()

    with _HERMES_CONFIG_LOCK:
        config_data: dict[str, Any] = {}
        if HERMES_CONFIG_PATH.exists():
            try:
                loaded = yaml.safe_load(
                    HERMES_CONFIG_PATH.read_text(encoding="utf-8", errors="ignore")
                )
                if isinstance(loaded, dict):
                    config_data = loaded
            except Exception as exc:
                logger.warning("Unable to read Hermes runtime config: %s", exc)

        existing_model_config = config_data.get("model")
        model_config = (
            dict(existing_model_config)
            if isinstance(existing_model_config, dict)
            else {}
        )
        model_config.update(
            {
                "provider": "custom",
                "default": model_id,
                "base_url": local_openai_base,
                "api_mode": "chat_completions",
                "api_key": "token-workshed-local",
            }
        )
        context_window = _profile_positive_int(
            agent_profile, runtime=None, key="context_window"
        ) or _resolve_local_model_context_window(model_id)
        if context_window is not None:
            model_config["context_length"] = context_window
            # Hermes probes every loopback OpenAI endpoint as if it were
            # Ollama. Pinning this value prevents vllm-mlx's serving limit
            # from being mistaken for the model's native context window.
            model_config["ollama_num_ctx"] = context_window
        else:
            model_config.pop("context_length", None)
            model_config.pop("ollama_num_ctx", None)
        generation_settings = _agent_generation_settings(
            agent_profile,
            generation_overrides,
        )
        configured_max_tokens = _coerce_int(
            generation_settings.get(
                "max_tokens", generation_settings.get("max_new_tokens")
            )
        )
        if configured_max_tokens is not None and configured_max_tokens > 0:
            model_config["max_tokens"] = max(1, min(32768, configured_max_tokens))
        elif agent_profile is not None:
            model_config.pop("max_tokens", None)
        configured_temperature = _coerce_float(generation_settings.get("temperature"))
        if configured_temperature is not None and 0.0 <= configured_temperature <= 2.0:
            # Hermes reads this through its local model config and turns it
            # into a per-request override. The chat UI value is merged last,
            # so it remains a user-level override over the profile default.
            model_config["temperature"] = configured_temperature
        elif agent_profile is not None:
            model_config.pop("temperature", None)
        config_data["model"] = model_config

        display_config = config_data.get("display")
        display_config = (
            dict(display_config) if isinstance(display_config, dict) else {}
        )
        # token-workshed renders reasoning in its own thinking surface.
        display_config["show_reasoning"] = False
        config_data["display"] = display_config

        security_config = config_data.get("security")
        security_config = (
            dict(security_config) if isinstance(security_config, dict) else {}
        )
        # The bundled runtime uses Hermes' pattern fallback and should not
        # leak an optional scanner installation warning into chat output.
        security_config["tirith_enabled"] = False
        config_data["security"] = security_config

        platform_toolsets = config_data.get("platform_toolsets")
        platform_toolsets = (
            dict(platform_toolsets) if isinstance(platform_toolsets, dict) else {}
        )
        # Normal chat should not pay for the full Hermes tool schema. Explicit
        # tool requests override this per invocation with hermes-cli.
        platform_toolsets["cli"] = []
        config_data["platform_toolsets"] = platform_toolsets

        serialized = yaml.safe_dump(
            config_data,
            allow_unicode=True,
            sort_keys=False,
        )
        current = ""
        if HERMES_CONFIG_PATH.exists():
            try:
                current = HERMES_CONFIG_PATH.read_text(
                    encoding="utf-8",
                    errors="ignore",
                )
            except OSError:
                current = ""
        if current == serialized:
            return
        temporary_path = HERMES_CONFIG_PATH.with_suffix(".yaml.tmp")
        temporary_path.write_text(serialized, encoding="utf-8")
        temporary_path.replace(HERMES_CONFIG_PATH)


def _build_hermes_subprocess_env(
    *,
    server_url: str,
    selected_model: str = "",
    agent_profile: dict[str, Any] | None = None,
    generation_overrides: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Build isolated Hermes runtime env bound to token-workshed local endpoint."""
    HERMES_STATE_DIR.mkdir(parents=True, exist_ok=True)
    HERMES_HOME_DIR.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    local_openai_base = _openai_base_url_from_server_url(server_url)
    _sync_hermes_home_env(local_openai_base=local_openai_base)
    _sync_hermes_runtime_config(
        server_url=server_url,
        selected_model=selected_model,
        agent_profile=agent_profile,
        generation_overrides=generation_overrides,
    )
    env["HERMES_HOME"] = str(HERMES_HOME_DIR)
    env["OPENAI_BASE_URL"] = local_openai_base
    # Keep this compatibility alias for Hermes code paths that inspect it.
    env["OPENROUTER_BASE_URL"] = local_openai_base
    env["OPENAI_API_KEY"] = "token-workshed-local"
    env["HERMES_INFERENCE_PROVIDER"] = "custom"
    env["HERMES_QUIET"] = "1"
    env["NO_COLOR"] = "1"
    env["FORCE_COLOR"] = "0"
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    return env


def _hermes_python_is_supported(candidate: str | Path) -> bool:
    """Return whether an interpreter satisfies the current Hermes Python range."""
    try:
        completed = subprocess.run(
            [
                str(candidate),
                "-c",
                (
                    "import sys; "
                    "raise SystemExit(0 if "
                    f"({HERMES_MIN_PYTHON!r} <= sys.version_info[:2] < "
                    f"{HERMES_MAX_PYTHON!r}) else 1)"
                ),
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0


def _resolve_hermes_python_executable() -> str:
    """Prefer a dedicated Hermes venv with a supported Python interpreter."""
    candidates = (
        HERMES_VENV_PYTHON,
        HERMES_VENV_DIR / "bin" / "python3",
    )
    for candidate in candidates:
        if candidate.exists() and _hermes_python_is_supported(candidate):
            return str(candidate)
    return _resolve_hermes_bootstrap_python()


def _resolve_hermes_bootstrap_python() -> str:
    """Find a supported interpreter even when token-workshed itself uses 3.10."""
    candidates: list[str] = [sys.executable]
    for executable_name in ("python3.13", "python3.12", "python3.11"):
        resolved = shutil.which(executable_name)
        if resolved:
            candidates.append(resolved)

    seen: set[str] = set()
    for candidate in candidates:
        normalized = str(Path(candidate).expanduser())
        if normalized in seen:
            continue
        seen.add(normalized)
        if _hermes_python_is_supported(normalized):
            return normalized
    raise RuntimeError(
        "Hermes Agent requires Python >=3.11 and <3.14. Install Python 3.11, "
        "3.12, or 3.13 so token-workshed can create its isolated Hermes runtime."
    )


def _resolve_hermes_source_version() -> str:
    """Read the vendored Hermes version without importing its dependencies."""
    version_file = HERMES_DIR / "hermes_cli" / "__init__.py"
    try:
        source = version_file.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""
    match = re.search(r"__version__\s*=\s*[\"']([^\"']+)", source)
    return str(match.group(1)).strip() if match else ""


def _hermes_install_command(python_executable: str) -> list[str]:
    """Build an install command for both legacy and pyproject Hermes trees."""
    pyproject_path = HERMES_DIR / "pyproject.toml"
    if pyproject_path.exists():
        return [
            python_executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "-e",
            str(HERMES_DIR),
        ]

    requirements_path = HERMES_DIR / "requirements.txt"
    if requirements_path.exists():
        return [
            python_executable,
            "-m",
            "pip",
            "install",
            "-r",
            str(requirements_path),
        ]
    raise RuntimeError(f"Hermes dependency metadata is missing under {HERMES_DIR}.")


def _probe_hermes_runtime(*, python_executable: str) -> tuple[bool, str]:
    """Quick import check to verify Hermes runtime dependencies are available."""
    try:
        completed = subprocess.run(
            [
                python_executable,
                "-c",
                "import yaml; import hermes_cli.main",
            ],
            cwd=str(HERMES_DIR),
            capture_output=True,
            text=True,
            timeout=25,
            check=False,
        )
    except Exception as exc:
        return False, str(exc)
    if completed.returncode == 0:
        return True, ""
    stderr_text = str(completed.stderr or "").strip()
    stdout_text = str(completed.stdout or "").strip()
    detail = stderr_text or stdout_text or f"exit code {completed.returncode}"
    return False, _truncate_error_text(detail)


def _ensure_hermes_runtime_ready() -> str:
    """Ensure Hermes has an isolated venv with required dependencies installed."""
    global _HERMES_BOOTSTRAP_READY
    global _HERMES_BOOTSTRAP_PYTHON

    HERMES_STATE_DIR.mkdir(parents=True, exist_ok=True)
    HERMES_HOME_DIR.mkdir(parents=True, exist_ok=True)

    with _HERMES_BOOTSTRAP_LOCK:
        if _HERMES_BOOTSTRAP_READY and _HERMES_BOOTSTRAP_PYTHON:
            return _HERMES_BOOTSTRAP_PYTHON

        venv_needs_rebuild = (
            not HERMES_VENV_PYTHON.exists()
            or not _hermes_python_is_supported(HERMES_VENV_PYTHON)
        )
        if venv_needs_rebuild:
            try:
                bootstrap_python = _resolve_hermes_bootstrap_python()
                venv_args = [bootstrap_python, "-m", "venv"]
                if HERMES_VENV_DIR.exists():
                    venv_args.append("--clear")
                venv_args.append(str(HERMES_VENV_DIR))
                subprocess.run(
                    venv_args,
                    cwd=str(HERMES_DIR),
                    capture_output=True,
                    text=True,
                    timeout=180,
                    check=True,
                )
            except subprocess.CalledProcessError as exc:
                stderr_text = str(exc.stderr or "").strip()
                raise RuntimeError(
                    "Failed to create Hermes virtual environment "
                    f"at {HERMES_VENV_DIR}: {_truncate_error_text(stderr_text or str(exc))}"
                ) from exc
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to create Hermes virtual environment at {HERMES_VENV_DIR}: {exc}"
                ) from exc

        python_executable = _resolve_hermes_python_executable()
        source_version = _resolve_hermes_source_version()
        installed_version = ""
        try:
            installed_version = HERMES_RUNTIME_VERSION_STAMP.read_text(
                encoding="utf-8",
                errors="ignore",
            ).strip()
        except OSError:
            pass
        ok, details = _probe_hermes_runtime(python_executable=python_executable)
        needs_dependency_sync = bool(
            source_version and source_version != installed_version
        )
        if not ok or needs_dependency_sync:
            env = os.environ.copy()
            env.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
            try:
                subprocess.run(
                    [python_executable, "-m", "pip", "install", "-U", "pip"],
                    cwd=str(HERMES_DIR),
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=240,
                    check=False,
                )
                install_result = subprocess.run(
                    _hermes_install_command(python_executable),
                    cwd=str(HERMES_DIR),
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=1800,
                    check=False,
                )
            except Exception as exc:
                raise RuntimeError(
                    "Failed while installing Hermes dependencies. "
                    f"Try running {_hermes_requirements_install_hint()} manually. ({exc})"
                ) from exc

            if install_result.returncode != 0:
                stderr_text = str(install_result.stderr or "").strip()
                stdout_text = str(install_result.stdout or "").strip()
                summary = (
                    stderr_text
                    or stdout_text
                    or f"exit code {install_result.returncode}"
                )
                raise RuntimeError(
                    "Hermes dependency install failed. "
                    f"Run {_hermes_requirements_install_hint()} manually. "
                    f"Details: {_truncate_error_text(summary)}"
                )

            ok, details = _probe_hermes_runtime(python_executable=python_executable)
            if not ok:
                raise RuntimeError(
                    "Hermes runtime is still not importable after dependency install. "
                    f"Details: {details or 'unknown import error'}"
                )

        if source_version:
            try:
                HERMES_RUNTIME_VERSION_STAMP.write_text(
                    f"{source_version}\n",
                    encoding="utf-8",
                )
            except OSError as exc:
                logger.warning("Unable to write Hermes runtime version stamp: %s", exc)

        _HERMES_BOOTSTRAP_READY = True
        _HERMES_BOOTSTRAP_PYTHON = python_executable
        return python_executable


def _hermes_requirements_install_hint() -> str:
    pyproject_path = HERMES_DIR / "pyproject.toml"
    requirements_path = HERMES_DIR / "requirements.txt"
    venv_python = HERMES_VENV_PYTHON
    if pyproject_path.exists():
        if venv_python.exists():
            return f"`{venv_python} -m pip install --upgrade -e '{HERMES_DIR}'`"
        return (
            "`python3 -m venv "
            f"{HERMES_VENV_DIR} && "
            f"{venv_python} -m pip install --upgrade -e '{HERMES_DIR}'`"
        )
    if venv_python.exists():
        return f"`{venv_python} -m pip install -r {requirements_path}`"
    return (
        "`python3 -m venv "
        f"{HERMES_VENV_DIR} && "
        f"{venv_python} -m pip install -U pip && "
        f"{venv_python} -m pip install -r {requirements_path}`"
    )


def _extract_hermes_session_id(*, stderr_text: str) -> str:
    match = _HERMES_SESSION_ID_RE.search(str(stderr_text or ""))
    if not match:
        return ""
    return str(match.group(1) or "").strip()


def _extract_hermes_context_window_error(
    detail_text: str,
) -> tuple[int | None, int | None] | None:
    """Extract Hermes model-context mismatch from stderr/stdout text."""
    text = str(detail_text or "").strip()
    if not text:
        return None

    match = _HERMES_MIN_CONTEXT_RE.search(text)
    if match:
        current_raw = str(match.group(1) or "").replace(",", "").strip()
        required_raw = str(match.group(2) or "").replace(",", "").strip()
        current = int(current_raw) if current_raw.isdigit() else None
        required = int(required_raw) if required_raw.isdigit() else None
        return current, required

    lowered = text.lower()
    if "below the minimum" not in lowered and "minimum context" not in lowered:
        return None

    fallback = _HERMES_MIN_CONTEXT_FALLBACK_RE.search(text)
    required = None
    if fallback:
        required_raw = str(fallback.group(1) or "").replace(",", "").strip()
        if required_raw.isdigit():
            required = int(required_raw)
            if "k" in fallback.group(0).lower():
                required *= 1000
    return None, required


def _build_hermes_context_error_message(
    *,
    selected_model: str,
    current_tokens: int | None,
    required_tokens: int | None,
) -> str:
    required = (
        required_tokens
        if isinstance(required_tokens, int) and required_tokens > 0
        else 64000
    )
    if isinstance(current_tokens, int) and current_tokens > 0:
        return (
            "Hermes requires a larger context window for this model. "
            f"Current model '{selected_model}' exposes about {current_tokens:,} tokens, "
            f"below Hermes minimum {required:,}. "
            "Switch to a model with >=64K context or use direct chat for this model."
        )
    return (
        "Hermes rejected this model because its context window is below the minimum required "
        f"({required:,} tokens). "
        "Switch to a >=64K context model or use direct chat for this model."
    )


def _is_hermes_boilerplate_reply(reply_text: str, *, user_message: str = "") -> bool:
    """Detect recurring Hermes boilerplate stubs that are not useful final answers."""
    text = str(reply_text or "").strip().lower()
    if not text:
        return False
    if _looks_like_code_generation_request(user_message):
        return False

    compact = re.sub(r"\s+", " ", text)
    markers = (
        "this is a simple example of how to use the hermes_cli",
        "import hermes_cli",
        "hermes_cli package is installed",
        "hermes_cli library is installed",
        "this is a simple script to run a python program",
        "run the program",
    )
    if any(marker in compact for marker in markers):
        return True

    code_like = (
        ("import os" in text and "import sys" in text)
        and "def main():" in text
        and ('__name__ == "__main__"' in text or "__name__ == '__main__'" in text)
    )
    if code_like:
        return True

    if (
        text.startswith("```python")
        and "def main():" in text
        and "simple script" in compact
    ):
        return True

    return False


def _openclaw_gateway_health_url(port: int) -> str:
    return f"http://127.0.0.1:{port}/healthz"


def _openclaw_gateway_health_urls(port: int) -> tuple[str, ...]:
    """Return gateway health/readiness probe URLs in preferred order."""
    base = f"http://127.0.0.1:{port}"
    return tuple(f"{base}{path}" for path in _OPENCLAW_GATEWAY_HEALTH_ENDPOINTS)


def _is_openclaw_gateway_port_open(port: int, timeout: float = 0.4) -> bool:
    """Fast TCP probe for loopback gateway port availability."""
    try:
        with socket.create_connection(("127.0.0.1", int(port)), timeout=timeout):
            return True
    except OSError:
        return False


def _read_openclaw_gateway_log_tail(limit: int = 1200) -> str:
    """Read trailing gateway log output for actionable startup diagnostics."""
    try:
        if not _OPENCLAW_GATEWAY_LOG_PATH.exists():
            return ""
        raw = _OPENCLAW_GATEWAY_LOG_PATH.read_text(
            encoding="utf-8",
            errors="ignore",
        )
    except Exception:
        return ""
    text = raw.strip()
    if not text:
        return ""
    if len(text) > limit:
        text = text[-limit:]
    return _truncate_error_text(text, limit=limit)


def _is_openclaw_gateway_healthy(port: int) -> bool:
    """Check whether persistent OpenClaw gateway responds on readiness/health endpoints."""
    for url in _openclaw_gateway_health_urls(port):
        try:
            response = requests.get(url, timeout=0.9)
        except Exception:
            continue
        if response.ok:
            return True
    return False


def _stop_openclaw_gateway_process_unlocked() -> None:
    """Stop persistent gateway process. Caller must hold _OPENCLAW_GATEWAY_LOCK."""
    global _OPENCLAW_GATEWAY_PROCESS
    global _OPENCLAW_GATEWAY_RUNTIME_KEY
    global _OPENCLAW_GATEWAY_PORT

    process = _OPENCLAW_GATEWAY_PROCESS
    if process is not None and process.poll() is None:
        try:
            process.terminate()
            process.wait(timeout=2.0)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass
    _OPENCLAW_GATEWAY_PROCESS = None
    _OPENCLAW_GATEWAY_RUNTIME_KEY = ""
    _OPENCLAW_GATEWAY_PORT = 0


def _stop_openclaw_gateway_process() -> None:
    """Stop persistent OpenClaw gateway process safely."""
    with _OPENCLAW_GATEWAY_LOCK:
        _stop_openclaw_gateway_process_unlocked()


def _ensure_openclaw_gateway_process(
    *,
    runtime_key: str,
) -> int:
    """Ensure a resident gateway process is up (Ollama-like always-on runtime)."""
    global _OPENCLAW_GATEWAY_PROCESS
    global _OPENCLAW_GATEWAY_RUNTIME_KEY
    global _OPENCLAW_GATEWAY_PORT

    gateway_port = _resolve_openclaw_gateway_port()
    startup_timeout_seconds = _resolve_openclaw_gateway_startup_timeout_seconds()

    with _OPENCLAW_GATEWAY_LOCK:
        process = _OPENCLAW_GATEWAY_PROCESS
        if process is not None and process.poll() is not None:
            _stop_openclaw_gateway_process_unlocked()
            process = None

        if (
            process is not None
            and _OPENCLAW_GATEWAY_RUNTIME_KEY == runtime_key
            and _OPENCLAW_GATEWAY_PORT == gateway_port
        ):
            if _is_openclaw_gateway_healthy(gateway_port):
                return gateway_port
            # Transient health endpoint blips should not force hard restarts when
            # the resident process is alive and listening.
            if _is_openclaw_gateway_port_open(gateway_port):
                return gateway_port

        if process is not None:
            _stop_openclaw_gateway_process_unlocked()

        launcher_base, _ = _resolve_openclaw_launcher()
        base_args = [
            "--auth",
            "none",
            "--bind",
            "loopback",
            "--port",
            str(gateway_port),
            "--allow-unconfigured",
        ]
        launch_commands: list[list[str]] = [
            [*launcher_base, "gateway", "run", *base_args],
            [*launcher_base, "gateway", *base_args],
        ]
        # De-duplicate if launcher/version collapses to same argv.
        unique_commands: list[list[str]] = []
        seen_commands: set[tuple[str, ...]] = set()
        for cmd in launch_commands:
            key = tuple(cmd)
            if key in seen_commands:
                continue
            seen_commands.add(key)
            unique_commands.append(cmd)

        env = _build_openclaw_subprocess_env()
        OPENCLAW_STATE_DIR.mkdir(parents=True, exist_ok=True)
        _OPENCLAW_GATEWAY_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

        attempt_diagnostics: list[str] = []
        total_attempts = len(unique_commands)
        for attempt_index, command in enumerate(unique_commands, start=1):
            with _OPENCLAW_GATEWAY_LOG_PATH.open("a", encoding="utf-8") as gateway_log:
                gateway_log.write(
                    f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                    f"starting gateway on port {gateway_port} "
                    f"(attempt {attempt_index}/{total_attempts})\n"
                )
                gateway_log.write(f"command: {' '.join(command)}\n")

            gateway_log_stream = _OPENCLAW_GATEWAY_LOG_PATH.open(
                "a",
                encoding="utf-8",
            )
            try:
                process = subprocess.Popen(
                    command,
                    cwd=str(OPENCLAW_DIR),
                    env=env,
                    stdout=gateway_log_stream,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            finally:
                try:
                    gateway_log_stream.close()
                except Exception:
                    pass

            _OPENCLAW_GATEWAY_PROCESS = process
            _OPENCLAW_GATEWAY_RUNTIME_KEY = runtime_key
            _OPENCLAW_GATEWAY_PORT = gateway_port

            deadline = time.perf_counter() + startup_timeout_seconds
            port_open_since: float | None = None
            while time.perf_counter() < deadline:
                if process.poll() is not None:
                    break
                if _is_openclaw_gateway_healthy(gateway_port):
                    return gateway_port
                if _is_openclaw_gateway_port_open(gateway_port):
                    now_ts = time.perf_counter()
                    if port_open_since is None:
                        port_open_since = now_ts
                    elif (now_ts - port_open_since) >= 1.0:
                        return gateway_port
                else:
                    port_open_since = None
                time.sleep(0.15)

            exit_code = process.poll()
            _stop_openclaw_gateway_process_unlocked()
            log_tail = _read_openclaw_gateway_log_tail(limit=1200)
            if exit_code is None:
                detail = (
                    f"attempt {attempt_index}/{total_attempts}: "
                    "gateway did not become healthy in "
                    f"{round(startup_timeout_seconds, 1)}s"
                )
            else:
                detail = (
                    f"attempt {attempt_index}/{total_attempts}: "
                    f"gateway exited during startup (exit code {exit_code})"
                )
            if log_tail:
                detail = f"{detail}; recent log: {log_tail}"
            attempt_diagnostics.append(detail)

        summary = " | ".join(attempt_diagnostics)
        raise RuntimeError(
            "OpenClaw gateway startup failed after command fallbacks. " f"{summary}"
        )


def _diagnose_openclaw_backend(
    *,
    server_url: str,
    selected_model: str,
    runtime_target: dict[str, str] | None = None,
) -> str:
    """Build actionable diagnostics for OpenClaw model connectivity."""
    target = runtime_target or _resolve_openclaw_runtime_target(
        server_url=server_url,
        selected_model=selected_model,
    )
    backend = target.get("backend", "openai-compat")

    if backend == "ollama":
        base_url = target.get("base_url", _OPENCLAW_OLLAMA_DEFAULT_BASE_URL)
        try:
            installed = _list_ollama_models(base_url)
        except Exception as exc:
            return f"Cannot reach Ollama API {base_url}/api/tags: {exc}"
        requested = _normalize_ollama_model_id(target.get("model_id", ""))
        if requested and installed and requested not in installed:
            preview = ", ".join(installed[:6])
            return (
                "Ollama API is reachable, but requested model "
                f"'{requested}' is not installed. Available: {preview}"
            )
        parser_hint = "auto"
        return (
            f"Ollama API is reachable at {base_url}/api/tags. "
            "If this endpoint is backed by vllm-mlx, ensure server is started with "
            f"`--enable-auto-tool-choice --tool-call-parser {parser_hint}`."
        )

    probe_url = f"{server_url}/v1/models"
    try:
        response = requests.get(probe_url, timeout=4.0)
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        return f"Cannot reach local model API {probe_url}: {exc}"

    served_models: list[str] = []
    if isinstance(data, dict) and isinstance(data.get("data"), list):
        for item in data["data"]:
            if not isinstance(item, dict):
                continue
            model_id = str(item.get("id", "")).strip()
            if model_id:
                served_models.append(model_id)

    requested = str(target.get("model_id", "") or selected_model or "").strip()
    if requested and served_models and requested not in served_models:
        preview = ", ".join(served_models[:6])
        return (
            "Local model API is reachable, but requested model "
            f"'{requested}' is not served. Available: {preview}"
        )
    parser_hint = "auto"
    return (
        f"Local model API is reachable at {probe_url}. "
        "For reliable tool calling, start local server with "
        f"`--enable-auto-tool-choice --tool-call-parser {parser_hint}`."
    )


def _node_version(candidate: str | Path) -> tuple[int, int, int] | None:
    """Return a Node executable's semantic version, if it can be started."""
    try:
        completed = subprocess.run(
            [str(candidate), "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    version_text = (completed.stdout or completed.stderr or "").strip()
    match = re.search(r"v?(\d+)\.(\d+)\.(\d+)", version_text)
    if not match:
        return None
    major, minor, patch = (int(group) for group in match.groups())
    return major, minor, patch


def _openclaw_node_is_supported(candidate: str | Path) -> bool:
    """Match OpenClaw's supported Node release windows."""
    version = _node_version(candidate)
    if version is None:
        return False
    major = version[0]
    minimum = OPENCLAW_NODE_MINIMUMS.get(major)
    if minimum is not None:
        return version >= minimum
    return major >= 26


def _resolve_openclaw_node_executable() -> str:
    """Prefer an explicit override, then the bundled and system runtimes."""
    override = os.environ.get(OPENCLAW_NODE_OVERRIDE_ENV, "").strip()

    def resolve_candidate(candidate: str) -> str:
        path = Path(candidate).expanduser()
        if path.is_file():
            return str(path)
        return shutil.which(candidate) or ""

    if override:
        resolved_override = resolve_candidate(override)
        if resolved_override and _openclaw_node_is_supported(resolved_override):
            return resolved_override
        raise RuntimeError(
            f"{OPENCLAW_NODE_OVERRIDE_ENV} must point to a supported Node runtime "
            "(22.22.3+, 24.15.0+, or 25.9.0+)."
        )

    candidates = [str(OPENCLAW_BUNDLED_NODE), "node"]
    seen: set[str] = set()
    for candidate in candidates:
        resolved = resolve_candidate(candidate)
        if not resolved or resolved in seen:
            continue
        seen.add(resolved)
        if _openclaw_node_is_supported(resolved):
            return resolved

    raise RuntimeError(
        "OpenClaw requires Node 22.22.3+, 24.15.0+, or 25.9.0+. "
        f"Install a supported runtime or set {OPENCLAW_NODE_OVERRIDE_ENV}."
    )


def _resolve_openclaw_launcher() -> tuple[list[str], str]:
    """Prefer prebuilt dist entry for lower latency; fallback to run-node wrapper."""
    node_executable = _resolve_openclaw_node_executable()
    if OPENCLAW_DIST_ENTRY.exists():
        return [node_executable, str(OPENCLAW_DIST_ENTRY)], "dist/entry.js"
    if _OPENCLAW_LEGACY_DIST_ENTRY.exists():
        return [node_executable, str(_OPENCLAW_LEGACY_DIST_ENTRY)], "dist/index.js"
    if OPENCLAW_RUNNER.exists():
        return [node_executable, "scripts/run-node.mjs"], "scripts/run-node.mjs"
    raise RuntimeError(
        "OpenClaw launcher is missing. Expected dist/entry.js (or dist/index.js) "
        "or scripts/run-node.mjs under integrations/openclaw."
    )


def _preload_openclaw_runtime_target(*, runtime_target: dict[str, str]) -> None:
    """Best-effort preload to reduce first-turn cold start."""
    backend = runtime_target.get("backend", "openai-compat")
    base_url = str(runtime_target.get("base_url", "")).rstrip("/")
    model_id = str(runtime_target.get("model_id", "")).strip() or "default"

    if backend == "ollama":
        response = requests.post(
            f"{base_url}/api/chat",
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": "warmup"}],
                "stream": False,
                "keep_alive": _resolve_openclaw_keep_alive(),
                "options": {
                    "num_predict": 1,
                    "temperature": 0,
                },
            },
            timeout=25.0,
        )
        response.raise_for_status()
        return

    response = requests.post(
        f"{base_url}/chat/completions",
        json={
            "model": model_id,
            "messages": [{"role": "user", "content": "warmup"}],
            "max_tokens": 1,
            "temperature": 0,
        },
        timeout=25.0,
    )
    response.raise_for_status()


def _run_openclaw_warmup(*, server_url: str, selected_model: str) -> dict[str, Any]:
    """Warm up OpenClaw Node runtime and build path before first chat turn."""
    if not OPENCLAW_DIR.exists():
        raise RuntimeError(f"OpenClaw integration directory not found: {OPENCLAW_DIR}")

    runtime_target = _resolve_openclaw_runtime_target(
        server_url=server_url,
        selected_model=selected_model,
    )
    config_revision = _ensure_openclaw_runtime_config(runtime_target=runtime_target)
    runtime_cache_key = _openclaw_runtime_cache_key(
        runtime_target=runtime_target,
        config_revision=config_revision,
    )

    started = time.perf_counter()
    preload_latency_ms: float | None = None
    cache_hit = _is_openclaw_runtime_warm(runtime_cache_key)

    def _warm_once() -> int:
        nonlocal preload_latency_ms
        gateway_port = _ensure_openclaw_gateway_process(
            runtime_key=f"{runtime_target['runtime_key']}|cfg:{config_revision}",
        )
        if _env_flag("TOKEN_WORKSHED_OPENCLAW_PRELOAD", True) and not cache_hit:
            preload_started = time.perf_counter()
            _preload_openclaw_runtime_target(runtime_target=runtime_target)
            preload_latency_ms = (time.perf_counter() - preload_started) * 1000.0
            _mark_openclaw_runtime_warm(runtime_cache_key)
        return gateway_port

    # Default path avoids global run serialization so first real turn isn't blocked by warmup.
    if _should_serialize_openclaw_runs(False):
        with _OPENCLAW_NODE_RUN_LOCK:
            gateway_port = _warm_once()
    else:
        gateway_port = _warm_once()

    latency_ms = (time.perf_counter() - started) * 1000.0
    return {
        "latency_ms": round(latency_ms, 2),
        "preload_latency_ms": (
            round(preload_latency_ms, 2)
            if isinstance(preload_latency_ms, float)
            else None
        ),
        "cache_hit": cache_hit,
        "gateway_port": gateway_port,
        "backend": runtime_target["backend"],
        "model": runtime_target["model_id"],
        "keep_alive": _resolve_openclaw_keep_alive(),
    }


def _is_openclaw_timeout_response(
    *,
    reply_text: str,
    meta: dict[str, Any],
    stop_reason: str,
) -> bool:
    """Detect user-visible timeout payloads returned by OpenClaw embedded runs."""
    normalized_stop = str(stop_reason or "").strip().lower()
    if normalized_stop in {"timeout", "timed_out", "time_limit"}:
        return True

    lowered_reply = str(reply_text or "").strip().lower()
    if "request timed out before a response was generated" in lowered_reply:
        return True
    if "timed out" in lowered_reply and bool(meta.get("aborted")):
        return True
    return False


def _looks_like_openclaw_tool_request(user_message: str) -> bool:
    """Heuristic detector for user requests that should trigger a tool run."""
    lowered = re.sub(r"\s+", " ", str(user_message or "").strip().lower())
    if not lowered:
        return False
    tool_request_keywords = (
        "search the web",
        "browse the web",
        "open ",
        "launch ",
        "read the files",
        "read files",
        "list files",
        "desktop",
        "downloads",
        "run ",
        "execute ",
        "command",
        "terminal",
        "click ",
        "visit ",
        "navigate to",
        "find on the web",
        "搜索",
        "网页",
        "读取",
        "文件",
        "桌面",
        "下载",
        "运行",
        "执行",
        "命令",
        "打开",
    )
    return any(keyword in lowered for keyword in tool_request_keywords)


def _normalize_openclaw_tool_mode(value: str | None) -> str | None:
    normalized = str(value or "").strip().lower()
    if normalized in _OPENCLAW_TOOL_MODE_VALUES:
        return normalized
    return None


def _normalize_openclaw_allowed_tools(value: list[str] | None) -> list[str] | None:
    if not isinstance(value, list):
        return None
    out: list[str] = []
    seen: set[str] = set()
    for raw_name in value:
        name = str(raw_name or "").strip()
        if not name:
            continue
        lowered = name.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        out.append(lowered)
    return out or None


def _normalize_positive_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return parsed


def _looks_like_openclaw_shell_or_file_request(user_message: str) -> bool:
    lowered = re.sub(r"\s+", " ", str(user_message or "").strip().lower())
    if not lowered:
        return False
    keywords = (
        "run ",
        "execute ",
        "command",
        "terminal",
        "shell",
        "bash",
        "zsh",
        "powershell",
        "read file",
        "read files",
        "write file",
        "write files",
        "edit file",
        "edit files",
        "modify file",
        "list files",
        "list directory",
        "workspace",
        "filesystem",
        "file path",
        "目录",
        "路径",
        "终端",
        "命令",
        "文件",
        "读取",
        "写入",
        "编辑",
    )
    return any(keyword in lowered for keyword in keywords)


def _looks_like_openclaw_web_request(user_message: str) -> bool:
    lowered = re.sub(r"\s+", " ", str(user_message or "").strip().lower())
    if not lowered:
        return False
    keywords = (
        "search the web",
        "browse the web",
        "search online",
        "look up",
        "find online",
        "visit ",
        "open http",
        "open https",
        "website",
        "url",
        "news",
        "web search",
        "browser",
        "google",
        "bing",
        "duckduckgo",
        "搜索",
        "上网",
        "浏览",
        "网页",
        "查一下",
        "查询",
    )
    if "http://" in lowered or "https://" in lowered:
        return True
    return any(keyword in lowered for keyword in keywords)


def _resolve_hermes_toolsets_for_message(
    user_message: str,
    agent_profile: dict[str, Any] | None = None,
) -> str | None:
    """Choose the narrowest Hermes toolsets that satisfy the user intent."""
    if agent_profile is not None and not _profile_runtime_supports_tools(
        agent_profile, "hermes"
    ):
        # A degraded profile is intentionally direct-chat only. Passing
        # Hermes toolsets here would make models that emitted literal markup
        # during Configure fail again on an explicit tool request.
        return None
    profile_enabled = agent_profile is not None
    if not profile_enabled and not _looks_like_openclaw_tool_request(user_message):
        return None

    lowered = re.sub(r"\s+", " ", str(user_message or "").strip().lower())
    web_intent = _looks_like_openclaw_web_request(user_message)
    shell_or_file_intent = _looks_like_openclaw_shell_or_file_request(user_message)
    file_intent = any(
        keyword in lowered
        for keyword in (
            "read file",
            "read files",
            "write file",
            "write files",
            "edit file",
            "edit files",
            "modify file",
            "list files",
            "filesystem",
            "file path",
            "读取文件",
            "写入文件",
            "编辑文件",
            "文件路径",
        )
    )

    # Hermes exposes file operations as one toolset and terminal/process as a
    # second, inseparable toolset.  The profile's safe base maps to ``file``;
    # terminal (and therefore exec) is only admitted by explicit intent.
    toolsets: list[str] = ["file"] if profile_enabled else []
    if shell_or_file_intent:
        toolsets.append("terminal")
    if file_intent and not profile_enabled:
        toolsets.append("file")
    if web_intent:
        toolsets.append("web")
    if toolsets:
        return ",".join(dict.fromkeys(toolsets))

    # Preserve access to less common tools such as browser/computer-use when
    # the intent cannot be represented by the lightweight core toolsets.
    return "hermes-cli"


def _is_local_openai_compatible_openclaw_path(
    *,
    server_url: str,
    agent_runtime: str | None,
) -> bool:
    if _resolve_agent_runtime(agent_runtime) != "openclaw":
        return False
    if _resolve_openclaw_backend_mode() != "openai-compat":
        return False
    if _env_flag("TOKEN_WORKSHED_OPENCLAW_NATIVE_CHAT", False):
        return False
    parsed = urllib.parse.urlparse(str(server_url or "").strip())
    hostname = str(parsed.hostname or "").strip().lower()
    if not hostname:
        return False
    if hostname in {"localhost", "127.0.0.1", "::1", "0.0.0.0"}:
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _resolve_openclaw_turn_request_options(
    *,
    user_message: str,
    tool_mode: str | None,
    allowed_tools: list[str] | None,
    model_first_chunk_timeout_ms: int | None,
    server_url: str,
    selected_model: str,
    agent_runtime: str | None,
    agent_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    del selected_model  # reserved for future model-specific policy branches.
    explicit_tool_mode = _normalize_openclaw_tool_mode(tool_mode)
    explicit_allowed_tools = _normalize_openclaw_allowed_tools(allowed_tools)
    explicit_timeout_ms = _normalize_positive_int(model_first_chunk_timeout_ms)

    resolved_tool_mode: str
    resolved_allowed_tools: list[str] | None

    if explicit_tool_mode is not None:
        resolved_tool_mode = explicit_tool_mode
        resolved_allowed_tools = explicit_allowed_tools
    elif explicit_allowed_tools:
        resolved_tool_mode = "small"
        resolved_allowed_tools = explicit_allowed_tools
    elif agent_profile is not None:
        # A structured-tool probe is required before advertising schemas to the
        # runtime. Degraded profiles still support direct agent replies, but
        # suppress schemas until a recalibration confirms structured calls.
        if _profile_runtime_supports_tools(agent_profile, "openclaw"):
            policy = _profile_tool_policy(agent_profile)
            resolved_tool_mode = "small"
            resolved_allowed_tools = list(policy["always"])
            if _looks_like_openclaw_shell_or_file_request(user_message):
                resolved_allowed_tools.extend(policy["terminal_on_demand"])
            if _looks_like_openclaw_web_request(user_message):
                resolved_allowed_tools.extend(policy["web_on_demand"])
            resolved_allowed_tools = list(dict.fromkeys(resolved_allowed_tools))
        else:
            resolved_tool_mode = "none"
            resolved_allowed_tools = []
    else:
        web_intent = _looks_like_openclaw_web_request(user_message)
        file_intent = _looks_like_openclaw_shell_or_file_request(user_message)
        if web_intent and not file_intent:
            resolved_tool_mode = "small"
            resolved_allowed_tools = list(_OPENCLAW_LIGHTWEIGHT_WEB_TOOLS)
        elif file_intent:
            resolved_tool_mode = "small"
            resolved_allowed_tools = list(_OPENCLAW_LIGHTWEIGHT_FILE_TOOLS)
        elif web_intent:
            resolved_tool_mode = "small"
            resolved_allowed_tools = list(_OPENCLAW_LIGHTWEIGHT_WEB_TOOLS)
        else:
            resolved_tool_mode = "none"
            resolved_allowed_tools = []

    resolved_first_chunk_timeout_ms = explicit_timeout_ms
    if (
        resolved_first_chunk_timeout_ms is None
        and _is_local_openai_compatible_openclaw_path(
            server_url=server_url,
            agent_runtime=agent_runtime,
        )
    ):
        if resolved_tool_mode == "none":
            resolved_first_chunk_timeout_ms = (
                _OPENCLAW_LOCAL_NONE_FIRST_CHUNK_TIMEOUT_MS
            )
        else:
            resolved_first_chunk_timeout_ms = (
                _OPENCLAW_LOCAL_TOOLS_FIRST_CHUNK_TIMEOUT_MS
            )

    return {
        "tool_mode": resolved_tool_mode,
        "allowed_tools": resolved_allowed_tools,
        "model_first_chunk_timeout_ms": resolved_first_chunk_timeout_ms,
    }


def _looks_like_code_generation_request(user_message: str) -> bool:
    """Heuristic detector for prompts explicitly asking the model to write code."""
    lowered = re.sub(r"\s+", " ", str(user_message or "").strip().lower())
    if not lowered:
        return False
    code_keywords = (
        "write code",
        "generate code",
        "python script",
        "code snippet",
        "implement ",
        "写代码",
        "代码",
        "脚本",
        "示例程序",
    )
    return any(keyword in lowered for keyword in code_keywords)


def _is_openclaw_capability_refusal(*, reply_text: str, user_message: str) -> bool:
    """Detect plain-language capability refusals that should have been tool executions."""
    if not _looks_like_openclaw_tool_request(user_message):
        return False
    lowered = re.sub(r"\s+", " ", str(reply_text or "").strip().lower())
    if not lowered:
        return False
    if any(snippet in lowered for snippet in _OPENCLAW_CAPABILITY_REFUSAL_SNIPPETS):
        return True
    has_access_refusal = (
        "don't have access" in lowered
        or "do not have access" in lowered
        or "don't have permission" in lowered
        or "do not have permission" in lowered
        or "cannot access" in lowered
        or "can't access" in lowered
        or "unable to access" in lowered
    )
    mentions_capability = any(
        token in lowered
        for token in (
            "filesystem",
            "local file",
            "desktop",
            "web search",
            "search the web",
            "browser",
        )
    )
    return has_access_refusal and mentions_capability


_OPENCLAW_PROGRESS_ONLY_SNIPPETS = (
    "i'll search for information",
    "i will search for information",
    "using the web_fetch tool",
    "using the browser tool",
    "i'll use the browser tool",
    "tool was blocked",
    "let me try another source",
    "let me run that command",
    "the user has requested secure, defensive, implementation-ready guidance",
)


def _is_openclaw_progress_only_reply(*, reply_text: str, user_message: str) -> bool:
    """Detect intermediate tool-progress logs that are not a user-facing final answer."""
    if not _looks_like_openclaw_tool_request(user_message):
        return False
    lowered = re.sub(r"\s+", " ", str(reply_text or "").strip().lower())
    if not lowered:
        return False

    hits = sum(1 for snippet in _OPENCLAW_PROGRESS_ONLY_SNIPPETS if snippet in lowered)
    if hits <= 0:
        return False

    # Final-answer markers should bypass this guard.
    if any(
        marker in lowered
        for marker in (
            "final answer:",
            "in summary",
            "here's what i found",
            "here is what i found",
            "结论",
            "总结",
            "最终答案",
        )
    ):
        return False

    # Strong signal: repeated assistant planning statements with tool mentions.
    lines = [
        re.sub(r"\s+", " ", ln.strip().lower())
        for ln in str(reply_text or "").splitlines()
        if ln.strip()
    ]
    if lines and len(set(lines)) < len(lines):
        return True

    return bool(
        "tool was blocked" in lowered
        or ("using the web_fetch tool" in lowered and "browser tool" in lowered)
        or hits >= 2
    )


def _run_openclaw_turn(
    *,
    server_url: str,
    selected_model: str,
    system_prompt: str,
    user_message: str,
    user_content: Any | None,
    conversation_id: str | None,
    backend_override: str | None = None,
    allow_ollama_fallback: bool = True,
    allow_transport_retry: bool = True,
    allow_timeout_retry: bool = True,
    allow_capability_retry: bool = True,
    strict_tool_mode: bool = False,
    timeout_override_seconds: int | None = None,
    cancel_event: threading.Event | None = None,
    run_id: str | None = None,
    session_id: str | None = None,
    tool_mode: str | None = None,
    allowed_tools: list[str] | None = None,
    model_first_chunk_timeout_ms: int | None = None,
    agent_profile: dict[str, Any] | None = None,
    generation_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one OpenClaw local agent turn and return normalized reply + metrics."""
    if not OPENCLAW_DIR.exists():
        raise RuntimeError(f"OpenClaw integration directory not found: {OPENCLAW_DIR}")

    runtime_target = _resolve_openclaw_runtime_target(
        server_url=server_url,
        selected_model=selected_model,
        backend_override=backend_override,
    )
    normalized_tool_mode = str(tool_mode or "none").strip().lower()
    if normalized_tool_mode not in {"none", "single", "small", "full"}:
        normalized_tool_mode = "none"
    normalized_allowed_tools: list[str] = []
    seen_allowed_tools: set[str] = set()
    for raw_name in allowed_tools or []:
        name = str(raw_name or "").strip()
        lowered = name.lower()
        if not name or lowered in seen_allowed_tools:
            continue
        seen_allowed_tools.add(lowered)
        normalized_allowed_tools.append(name)
    config_revision = _ensure_openclaw_runtime_config(
        runtime_target=runtime_target,
        tool_mode=normalized_tool_mode,
        allowed_tools=normalized_allowed_tools,
        agent_profile=agent_profile,
        generation_overrides=generation_overrides,
    )
    runtime_cache_key = _openclaw_runtime_cache_key(
        runtime_target=runtime_target,
        config_revision=config_revision,
    )
    runtime_cache_warm = _is_openclaw_runtime_warm(runtime_cache_key)
    gateway_port = _ensure_openclaw_gateway_process(
        runtime_key=f"{runtime_target['runtime_key']}|cfg:{config_revision}",
    )

    effective_session_id = str(
        session_id or ""
    ).strip() or _normalize_openclaw_session_id(
        conversation_id,
        selected_model=selected_model,
        runtime="openclaw",
    )
    effective_run_id = str(run_id or "").strip() or _make_openclaw_run_id(
        effective_session_id
    )
    composed_message = _compose_openclaw_turn_message(
        system_prompt=system_prompt,
        user_message=user_message,
        user_content=user_content,
        strict_tool_mode=strict_tool_mode,
    )

    if isinstance(timeout_override_seconds, int) and timeout_override_seconds > 0:
        timeout_seconds = max(30, min(3600, int(timeout_override_seconds)))
    else:
        profile_timeout = _profile_positive_int(
            agent_profile,
            runtime="openclaw",
            key="timeout_seconds",
        )
        timeout_seconds = (
            _resolve_openclaw_timeout_seconds(profile_timeout)
            if profile_timeout is not None
            else _resolve_openclaw_timeout_seconds()
        )
    launcher_base, launcher_label = _resolve_openclaw_launcher()
    agent_capabilities = _resolve_openclaw_agent_capabilities(
        launcher_base=launcher_base,
    )
    command = [
        *launcher_base,
        "agent",
        "--json",
        "--thinking",
        "off",
        "--session-id",
        effective_session_id,
        "--timeout",
        str(timeout_seconds),
        "--message",
        composed_message,
    ]
    if "--run-id" in agent_capabilities:
        command.extend(["--run-id", effective_run_id])
    if "--tool-mode" in agent_capabilities and normalized_tool_mode in {
        "none",
        "single",
        "small",
        "full",
    }:
        command.extend(["--tool-mode", normalized_tool_mode])

    if "--allowed-tool" in agent_capabilities:
        for name in normalized_allowed_tools:
            command.extend(["--allowed-tool", name])

    first_chunk_timeout_ms: int | None = None
    if model_first_chunk_timeout_ms is not None:
        try:
            parsed_timeout = int(model_first_chunk_timeout_ms)
        except (TypeError, ValueError):
            parsed_timeout = 0
        if parsed_timeout > 0:
            first_chunk_timeout_ms = parsed_timeout
    if (
        "--model-first-chunk-timeout-ms" in agent_capabilities
        and first_chunk_timeout_ms is not None
    ):
        command.extend(["--model-first-chunk-timeout-ms", str(first_chunk_timeout_ms)])

    env = _build_openclaw_subprocess_env()
    env["OPENCLAW_GATEWAY_PORT"] = str(gateway_port)

    request_start = time.perf_counter()
    try:

        def _run_command() -> subprocess.CompletedProcess[str]:
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("OpenClaw request cancelled.")

            process = subprocess.Popen(
                command,
                cwd=str(OPENCLAW_DIR),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            deadline = time.monotonic() + timeout_seconds + 20

            def _stop_process(kill_after: float = 3.0) -> tuple[str, str]:
                if process.poll() is None:
                    try:
                        process.terminate()
                    except Exception:
                        pass
                try:
                    stdout, stderr = process.communicate(timeout=kill_after)
                except subprocess.TimeoutExpired:
                    try:
                        process.kill()
                    except Exception:
                        pass
                    stdout, stderr = process.communicate()
                return str(stdout or ""), str(stderr or "")

            while True:
                if cancel_event is not None and cancel_event.is_set():
                    _stop_process()
                    raise RuntimeError("OpenClaw request cancelled.")

                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    stdout, stderr = _stop_process()
                    raise subprocess.TimeoutExpired(
                        command,
                        timeout_seconds + 20,
                        output=stdout,
                        stderr=stderr,
                    )

                try:
                    stdout, stderr = process.communicate(timeout=min(0.25, remaining))
                    return subprocess.CompletedProcess(
                        command,
                        process.returncode,
                        str(stdout or ""),
                        str(stderr or ""),
                    )
                except subprocess.TimeoutExpired:
                    continue

        # Default path avoids unnecessary global serialization.
        if _should_serialize_openclaw_runs(False):
            with _OPENCLAW_NODE_RUN_LOCK:
                completed = _run_command()
        else:
            completed = _run_command()
    except subprocess.TimeoutExpired as exc:
        # Retry only for likely cold-start paths. On warm runtime, fail fast so
        # stream layer can degrade without burning extra CPU/RAM in repeated runs.
        if allow_timeout_retry and not runtime_cache_warm:
            retry_timeout = _resolve_openclaw_timeout_retry_seconds(timeout_seconds)
            # Strict tool-mode retry on same backend first.
            if allow_capability_retry and not strict_tool_mode:
                try:
                    recovered = _run_openclaw_turn(
                        server_url=server_url,
                        selected_model=selected_model,
                        system_prompt=system_prompt,
                        user_message=user_message,
                        user_content=user_content,
                        conversation_id=conversation_id,
                        backend_override=backend_override,
                        allow_ollama_fallback=allow_ollama_fallback,
                        allow_transport_retry=allow_transport_retry,
                        allow_timeout_retry=False,
                        allow_capability_retry=False,
                        strict_tool_mode=True,
                        timeout_override_seconds=retry_timeout,
                        cancel_event=cancel_event,
                        run_id=effective_run_id,
                        session_id=effective_session_id,
                        tool_mode=normalized_tool_mode or None,
                        allowed_tools=allowed_tools,
                        model_first_chunk_timeout_ms=first_chunk_timeout_ms,
                        agent_profile=agent_profile,
                        generation_overrides=generation_overrides,
                    )
                    metrics = recovered.get("metrics")
                    if isinstance(metrics, dict):
                        metrics["timeout_retry_strict"] = True
                        metrics["timeout_retry_seconds"] = retry_timeout
                    return recovered
                except Exception:
                    if cancel_event is not None and cancel_event.is_set():
                        raise
                    pass

        raise RuntimeError(
            f"OpenClaw timed out after {timeout_seconds} seconds."
        ) from exc

    latency_ms = (time.perf_counter() - request_start) * 1000.0
    stdout_text = str(completed.stdout or "")
    stderr_text = str(completed.stderr or "")

    if completed.returncode != 0:
        details: list[str] = [f"exit code {completed.returncode}"]
        details.append(f"launcher: {launcher_label}")
        details.append(f"gateway_port: {gateway_port}")
        details.append(f"backend: {runtime_target['backend']}")
        details.append(f"model: {runtime_target['model_id']}")
        if stderr_text.strip():
            details.append(f"stderr: {_truncate_error_text(stderr_text)}")
        if stdout_text.strip():
            details.append(f"stdout: {_truncate_error_text(stdout_text)}")
        details.append(
            _truncate_error_text(
                _diagnose_openclaw_backend(
                    server_url=server_url,
                    selected_model=selected_model,
                    runtime_target=runtime_target,
                ),
                limit=360,
            )
        )
        raise RuntimeError(f"OpenClaw command failed ({'; '.join(details)}).")

    parsed = _parse_openclaw_json_output(stdout_text)
    reply_text, meta, stop_reason = _extract_openclaw_text_and_meta(parsed)
    if _is_openclaw_timeout_response(
        reply_text=reply_text,
        meta=meta,
        stop_reason=stop_reason,
    ):
        raise RuntimeError(
            "OpenClaw request timed out before a response was generated."
        )

    if stop_reason in {"error", "failed", "timeout"}:
        detail = reply_text or "No error text returned."
        lowered_detail = str(detail).strip().lower()
        if (
            "network connection error" in lowered_detail
            or "connection error" in lowered_detail
        ):
            diagnosis = _diagnose_openclaw_backend(
                server_url=server_url,
                selected_model=selected_model,
                runtime_target=runtime_target,
            )
            raise RuntimeError(
                "OpenClaw could not reach token-workshed local model backend. "
                f"{diagnosis}"
            )
        raise RuntimeError(f"OpenClaw run failed ({stop_reason}): {detail}")

    if (
        _OPENCLAW_TOOL_CALL_TAG_RE.search(reply_text)
        or _OPENCLAW_TOOL_CALL_OPEN_RE.search(reply_text)
        or "[calling tool:" in reply_text.lower()
    ):
        # Strict retry on same backend can recover some small-model alignment drifts.
        if allow_capability_retry and not strict_tool_mode:
            try:
                recovered = _run_openclaw_turn(
                    server_url=server_url,
                    selected_model=selected_model,
                    system_prompt=system_prompt,
                    user_message=user_message,
                    user_content=user_content,
                    conversation_id=conversation_id,
                    backend_override=backend_override,
                    allow_ollama_fallback=allow_ollama_fallback,
                    allow_transport_retry=allow_transport_retry,
                    allow_timeout_retry=allow_timeout_retry,
                    allow_capability_retry=False,
                    strict_tool_mode=True,
                    cancel_event=cancel_event,
                    run_id=effective_run_id,
                    session_id=effective_session_id,
                    tool_mode=normalized_tool_mode or None,
                    allowed_tools=allowed_tools,
                    model_first_chunk_timeout_ms=first_chunk_timeout_ms,
                    agent_profile=agent_profile,
                    generation_overrides=generation_overrides,
                )
                metrics = recovered.get("metrics")
                if isinstance(metrics, dict):
                    metrics["strict_retry"] = True
                return recovered
            except Exception:
                if cancel_event is not None and cancel_event.is_set():
                    raise
                pass
        raise RuntimeError(
            "OpenClaw model returned literal <tool_call> markup instead of executing tools. "
            "This usually means local tool-call parsing is not enabled (or model cannot tool-call reliably). "
            "Fix: restart local backend with "
            "`--enable-auto-tool-choice --tool-call-parser auto` "
            "(auto parser is recommended for Granite-style outputs), then retry. "
            f"Current model: {selected_model}."
        )

    if _is_openclaw_capability_refusal(
        reply_text=reply_text, user_message=user_message
    ):
        if allow_capability_retry and not strict_tool_mode:
            recovered = _run_openclaw_turn(
                server_url=server_url,
                selected_model=selected_model,
                system_prompt=system_prompt,
                user_message=user_message,
                user_content=user_content,
                conversation_id=conversation_id,
                backend_override=backend_override,
                allow_ollama_fallback=allow_ollama_fallback,
                allow_transport_retry=allow_transport_retry,
                allow_timeout_retry=allow_timeout_retry,
                allow_capability_retry=False,
                strict_tool_mode=True,
                cancel_event=cancel_event,
                run_id=effective_run_id,
                session_id=effective_session_id,
                tool_mode=normalized_tool_mode or None,
                allowed_tools=allowed_tools,
                model_first_chunk_timeout_ms=first_chunk_timeout_ms,
                agent_profile=agent_profile,
                generation_overrides=generation_overrides,
            )
            metrics = recovered.get("metrics")
            if isinstance(metrics, dict):
                metrics["capability_retry"] = True
            return recovered
        raise RuntimeError(
            "OpenClaw model replied with a capability-refusal message instead of executing tools. "
            "This usually means tool context was dropped or the model failed tool-use alignment. "
            f"Current model: {selected_model}. Reply: {_truncate_error_text(reply_text, limit=220)}"
        )

    # Let frontend fallback logic handle truly empty final answers.
    # Avoid returning literal "(empty response)" text, which can be duplicated by upstream payloads.
    if _chunk_key := re.sub(r"\s+", " ", str(reply_text or "").strip().lower()):
        if _chunk_key in {
            "(empty response)",
            "empty response",
            "__vllm_mlx_empty_response__",
        }:
            reply_text = ""
    else:
        reply_text = ""

    # Some models/backends occasionally finish with stop_reason=stop but emit no
    # assistant payload text in native Ollama transport. Retry once via
    # OpenAI-compatible transport while keeping OpenClaw runtime semantics.
    if (
        not reply_text
        and allow_transport_retry
        and str(runtime_target.get("backend") or "").strip().lower() == "ollama"
    ):
        try:
            recovered = _run_openclaw_turn(
                server_url=server_url,
                selected_model=selected_model,
                system_prompt=system_prompt,
                user_message=user_message,
                user_content=user_content,
                conversation_id=conversation_id,
                backend_override="openai-compat",
                allow_ollama_fallback=allow_ollama_fallback,
                allow_transport_retry=False,
                allow_timeout_retry=allow_timeout_retry,
                allow_capability_retry=allow_capability_retry,
                strict_tool_mode=strict_tool_mode,
                timeout_override_seconds=timeout_override_seconds,
                cancel_event=cancel_event,
                run_id=effective_run_id,
                session_id=effective_session_id,
                tool_mode=normalized_tool_mode or None,
                allowed_tools=allowed_tools,
                model_first_chunk_timeout_ms=first_chunk_timeout_ms,
                agent_profile=agent_profile,
                generation_overrides=generation_overrides,
            )
            metrics = recovered.get("metrics")
            if isinstance(metrics, dict):
                metrics["transport_retry_openai_compat"] = True
                metrics["transport_retry_reason"] = "empty_reply"
            return recovered
        except Exception as exc:
            if cancel_event is not None and cancel_event.is_set():
                raise
            logger.warning(
                "[openclaw] empty-reply transport retry failed (openai-compat): %s",
                _truncate_error_text(str(exc), limit=220),
            )

    agent_meta = (
        meta.get("agentMeta") if isinstance(meta.get("agentMeta"), dict) else {}
    )
    provider = (
        str(agent_meta.get("provider", runtime_target["provider_id"])).strip()
        or runtime_target["provider_id"]
    )
    model_name = str(agent_meta.get("model", runtime_target["model_id"])).strip() or (
        str(runtime_target["model_id"]).strip() or "default"
    )
    resolved_model = f"openclaw/{provider}/{model_name}"

    usage = (
        agent_meta.get("lastCallUsage")
        if isinstance(agent_meta.get("lastCallUsage"), dict)
        else {}
    )
    prompt_tokens = _coerce_int(usage.get("input"))
    completion_tokens = _coerce_int(usage.get("output"))
    total_tokens = _coerce_int(usage.get("total"))
    if total_tokens is None and (
        prompt_tokens is not None or completion_tokens is not None
    ):
        total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)

    duration_ms = (
        _coerce_float(meta.get("durationMs")) if isinstance(meta, dict) else None
    )
    effective_latency_ms = (
        float(duration_ms)
        if isinstance(duration_ms, float) and duration_ms > 0
        else latency_ms
    )
    tokens_per_second: float | None = None
    if completion_tokens is not None and effective_latency_ms > 0:
        tokens_per_second = completion_tokens / (effective_latency_ms / 1000.0)
    _mark_openclaw_runtime_warm(runtime_cache_key)

    return {
        "reply": reply_text,
        "model": resolved_model,
        "session_id": effective_session_id,
        "stop_reason": stop_reason or None,
        "metrics": {
            "latency_ms": round(effective_latency_ms, 2),
            "openclaw_run_id": effective_run_id,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "tokens_per_second": (
                round(tokens_per_second, 2)
                if isinstance(tokens_per_second, float)
                else None
            ),
        },
    }


def _run_hermes_turn(
    *,
    server_url: str,
    selected_model: str,
    system_prompt: str,
    user_message: str,
    user_content: Any | None,
    conversation_id: str | None,
    strict_tool_mode: bool = False,
    timeout_override_seconds: int | None = None,
    cancel_event: threading.Event | None = None,
    allowed_toolsets: tuple[str, ...] | None = None,
    agent_profile: dict[str, Any] | None = None,
    generation_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one Hermes local agent turn and return normalized reply + metrics."""
    if not HERMES_DIR.exists():
        raise RuntimeError(f"Hermes integration directory not found: {HERMES_DIR}")
    if not HERMES_ENTRY.exists():
        raise RuntimeError(
            f"Hermes launcher is missing. Expected executable at: {HERMES_ENTRY}"
        )

    raw_conversation_id = str(conversation_id or "").strip()
    conversation_key = _normalize_openclaw_session_id(
        raw_conversation_id,
        selected_model=selected_model,
        runtime="hermes",
    )
    with _HERMES_SESSION_LOCK:
        resume_session_id = (
            _HERMES_SESSION_BY_CONVERSATION.get(conversation_key, "")
            if raw_conversation_id
            else ""
        )

    composed_message = _compose_hermes_turn_message(
        system_prompt=system_prompt,
        user_message=user_message,
        user_content=user_content,
        strict_tool_mode=strict_tool_mode,
    )

    if isinstance(timeout_override_seconds, int) and timeout_override_seconds > 0:
        timeout_seconds = max(30, min(3600, int(timeout_override_seconds)))
    else:
        profile_timeout = _profile_positive_int(
            agent_profile,
            runtime="hermes",
            key="timeout_seconds",
        )
        timeout_seconds = (
            _resolve_hermes_timeout_seconds(profile_timeout)
            if profile_timeout is not None
            else _resolve_hermes_timeout_seconds()
        )

    hermes_python = _ensure_hermes_runtime_ready()
    provider_candidates = ("custom",)
    hermes_toolsets = _resolve_hermes_toolsets_for_message(
        user_message,
        agent_profile=agent_profile,
    )
    if allowed_toolsets is not None:
        allowed = {str(item).strip().lower() for item in allowed_toolsets if str(item).strip()}
        hermes_toolsets = ",".join(
            item for item in str(hermes_toolsets or "").split(",")
            if item.strip().lower() in allowed
        ) or None
    needs_tools = hermes_toolsets is not None
    profile_max_turns = _profile_positive_int(
        agent_profile,
        runtime="hermes",
        key="max_turns",
    )
    requested_max_turns = (
        _resolve_hermes_max_turns(profile_max_turns)
        if profile_max_turns is not None
        else _resolve_hermes_max_turns()
    )
    if needs_tools:
        max_turns = requested_max_turns
    else:
        # Leave room for a small model that elects to call one tool before
        # producing its final response, even for an apparently simple prompt.
        max_turns = min(max(2, requested_max_turns), 3)
    if strict_tool_mode:
        max_turns = max(max_turns, 6)
    model_hint = str(selected_model or "").strip()
    env = _build_hermes_subprocess_env(
        server_url=server_url,
        selected_model=model_hint,
        agent_profile=agent_profile,
        generation_overrides=generation_overrides,
    )

    request_start = time.perf_counter()
    attempt_errors: list[str] = []
    attempt_resume_session_id = resume_session_id
    selected_provider = provider_candidates[0]
    completed: subprocess.CompletedProcess[str] | None = None

    for provider_name in provider_candidates:
        selected_provider = provider_name
        command = [
            hermes_python,
            str(HERMES_ENTRY),
            "chat",
            "--quiet",
            "--provider",
            provider_name,
            "--query",
            composed_message,
            "--max-turns",
            str(max_turns),
            "--yolo",
            "--ignore-rules",
            "--source",
            "token-workshed",
        ]
        if model_hint:
            command.extend(["--model", model_hint])
        if hermes_toolsets:
            command.extend(["--toolsets", hermes_toolsets])
        if attempt_resume_session_id:
            command.extend(["--resume", attempt_resume_session_id])

        try:
            # Keep the pipes drained while polling so a verbose Hermes
            # subprocess cannot deadlock on a full stderr/stdout pipe. This
            # also gives the App Bridge's stop endpoint a real cancellation
            # path instead of merely abandoning a worker thread.
            process = subprocess.Popen(
                command,
                cwd=str(HERMES_DIR),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
            deadline = time.monotonic() + timeout_seconds + 20
            while True:
                try:
                    stdout_text, stderr_text = process.communicate(timeout=0.25)
                    completed = subprocess.CompletedProcess(
                        command,
                        process.returncode,
                        stdout_text,
                        stderr_text,
                    )
                    break
                except subprocess.TimeoutExpired:
                    if cancel_event is not None and cancel_event.is_set():
                        process.terminate()
                        try:
                            process.communicate(timeout=3)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.communicate()
                        raise RuntimeError("Hermes request stopped.")
                    if time.monotonic() >= deadline:
                        process.terminate()
                        try:
                            process.communicate(timeout=3)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.communicate()
                        raise subprocess.TimeoutExpired(command, timeout_seconds + 20)
        except subprocess.TimeoutExpired as exc:
            attempt_errors.append(
                f"provider={provider_name} timed out after {timeout_seconds} seconds"
            )
            if provider_name == provider_candidates[-1]:
                raise RuntimeError(
                    f"Hermes request timed out after {timeout_seconds} seconds."
                ) from exc
            continue

        stderr_text = str(completed.stderr or "")
        extracted_session_id = _extract_hermes_session_id(stderr_text=stderr_text)
        if extracted_session_id:
            attempt_resume_session_id = extracted_session_id

        if completed.returncode == 0:
            break

        missing_module_match = re.search(
            r"ModuleNotFoundError:\s+No module named ['\"]([^'\"]+)['\"]",
            stderr_text,
            re.IGNORECASE,
        )
        if missing_module_match:
            missing_module = (
                str(missing_module_match.group(1) or "").strip() or "unknown"
            )
            raise RuntimeError(
                "Hermes Python dependencies are missing "
                f"(`{missing_module}`). Install them first with: "
                f"{_hermes_requirements_install_hint()}."
            )

        stdout_text = str(completed.stdout or "")
        context_mismatch = _extract_hermes_context_window_error(
            f"{stderr_text}\n{stdout_text}".strip()
        )
        if context_mismatch is not None:
            current_ctx, required_ctx = context_mismatch
            raise RuntimeError(
                _build_hermes_context_error_message(
                    selected_model=model_hint or selected_model,
                    current_tokens=current_ctx,
                    required_tokens=required_ctx,
                )
            )

        details: list[str] = [
            f"provider={provider_name}",
            f"exit code {completed.returncode}",
            f"model: {model_hint or 'default'}",
        ]
        if (
            "run:  hermes setup" in stderr_text.lower()
            or "isn't configured yet" in stderr_text.lower()
        ):
            details.append(
                "Hermes runtime is not configured. Local endpoint env was injected, "
                "but Hermes still reported missing provider setup."
            )
        if stderr_text.strip():
            details.append(f"stderr: {_truncate_error_text(stderr_text)}")
        if stdout_text.strip():
            details.append(f"stdout: {_truncate_error_text(stdout_text)}")
        attempt_errors.append("; ".join(details))

    latency_ms = (time.perf_counter() - request_start) * 1000.0
    if completed is None or completed.returncode != 0:
        for candidate in attempt_errors:
            context_mismatch = _extract_hermes_context_window_error(candidate)
            if context_mismatch is not None:
                current_ctx, required_ctx = context_mismatch
                raise RuntimeError(
                    _build_hermes_context_error_message(
                        selected_model=model_hint or selected_model,
                        current_tokens=current_ctx,
                        required_tokens=required_ctx,
                    )
                )
        joined = (
            " | ".join(attempt_errors) if attempt_errors else "unknown Hermes failure"
        )
        raise RuntimeError(f"Hermes command failed ({joined}).")

    stdout_text = str(completed.stdout or "")
    stderr_text = str(completed.stderr or "")
    session_id = (
        _extract_hermes_session_id(stderr_text=stderr_text)
        or attempt_resume_session_id
        or conversation_key
    )
    if raw_conversation_id:
        with _HERMES_SESSION_LOCK:
            _HERMES_SESSION_BY_CONVERSATION[conversation_key] = session_id

    reply_text = _normalize_agent_visible_text(str(stdout_text or "")).strip()
    if _chunk_key := re.sub(r"\s+", " ", reply_text.lower()):
        if _chunk_key in {
            "(empty)",
            "(empty response)",
            "empty response",
            "__vllm_mlx_empty_response__",
        }:
            reply_text = ""

    if _is_hermes_boilerplate_reply(reply_text, user_message=user_message):
        raise RuntimeError(
            "Hermes returned boilerplate template output instead of a grounded final answer. "
            "This usually means the selected local model is not Hermes-agent compatible."
        )

    resolved_model = f"hermes/{selected_provider}/{model_hint or 'default'}"
    return {
        "reply": reply_text,
        "model": resolved_model,
        "session_id": session_id,
        "stop_reason": None,
        "metrics": {
            "latency_ms": round(latency_ms, 2),
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "tokens_per_second": None,
        },
    }


_AGENT_PROFILE_BASE_TOOLS = ("read", "write", "edit", "process")
_AGENT_PROFILE_TERMINAL_TOOLS = ("exec",)
_AGENT_PROFILE_WEB_TOOLS = ("web_search", "web_fetch", "browser")


def _enforce_agent_profile_auth(request: Request, app: FastAPI) -> None:
    """Apply manager-token protection when the desktop is remotely exposed."""
    expected = str(getattr(app.state, "manager_api_token", "") or "").strip()
    if not expected:
        return
    provided = str(
        request.headers.get("x-token-workshed-manager-token", "") or ""
    ).strip()
    if not provided or not secrets.compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail="Unauthorized manager request.")


def _profile_generation_defaults(metadata: dict[str, Any]) -> dict[str, Any]:
    """Keep only portable generation defaults from local model metadata."""
    config = metadata.get("generation_config")
    if not isinstance(config, dict):
        return {}
    allowed = (
        "temperature",
        "top_p",
        "top_k",
        "typical_p",
        "repetition_penalty",
        "do_sample",
        "max_new_tokens",
        "max_tokens",
    )
    defaults: dict[str, Any] = {}
    for key in allowed:
        value = config.get(key)
        if isinstance(value, bool) or isinstance(value, (str, int, float)):
            defaults[key] = value
    return defaults


def _profile_context_window(
    metadata: dict[str, Any], *, server_url: str, model_id: str
) -> int:
    """Resolve context without downloading a model as part of Configure."""
    metadata_value = _coerce_int(metadata.get("max_context_length"))
    if metadata_value is not None and metadata_value > 0:
        return min(metadata_value, 1_048_576)
    server_value = _query_local_model_context_window(
        target=server_url,
        selected_model=model_id,
    )
    if server_value is not None and server_value > 0:
        return min(server_value, 1_048_576)
    return 16_384


def _profile_probe_summary(probe: dict[str, Any]) -> dict[str, Any]:
    """Persist useful diagnostics, never commands, environment, or credentials."""
    fields = (
        "checked_at",
        "reason",
        "status",
        "cached",
        "agent_ready",
        "tool_calling",
        "tool_call_format",
        "degraded",
        "parser_used",
        "reasoning_parser_used",
        "chat_template_used",
        "needs_manual_review",
        "failures",
        "warnings",
        "failure_category",
        "recovery_attempted",
        "tool_choice_fallback",
        "fix_suggestions",
    )
    return {key: probe.get(key) for key in fields if key in probe}


def _profile_runtime_section(
    *,
    runtime: str,
    probe: dict[str, Any],
    settings: dict[str, Any],
) -> dict[str, Any]:
    default_timeout = (
        _resolve_hermes_timeout_seconds()
        if runtime == "hermes"
        else _resolve_openclaw_timeout_seconds()
    )
    tool_calling = str(probe.get("tool_calling") or "").strip().lower()
    if tool_calling not in {"passed", "degraded"}:
        tool_calling = "passed" if bool(probe.get("ok")) else "failed"
    section: dict[str, Any] = {
        "ready": bool(probe.get("ok")),
        "tool_calling": tool_calling,
        "tool_call_format": str(probe.get("tool_call_format") or "").strip()
        or ("tool_calls" if tool_calling == "passed" else None),
        "tool_call_parser": str(
            probe.get("parser_used") or settings.get("tool_call_parser") or ""
        ).strip()
        or None,
        "reasoning_parser": str(
            probe.get("reasoning_parser_used") or settings.get("reasoning_parser") or ""
        ).strip()
        or None,
        "timeout_seconds": default_timeout,
        "probe": _profile_probe_summary(probe),
    }
    if runtime == "hermes":
        section["max_turns"] = _resolve_hermes_max_turns()
    return section


def _build_agent_profile(
    *,
    model_id: str,
    server_url: str,
    metadata: dict[str, Any],
    family: str,
    settings: dict[str, Any],
    registry_warnings: list[str],
    openclaw_probe: dict[str, Any],
    hermes_probe: dict[str, Any],
) -> dict[str, Any]:
    warnings: list[str] = []
    for item in [*registry_warnings, *(metadata.get("warnings") or [])]:
        text = str(item or "").strip()
        if text and text not in warnings:
            warnings.append(text)
    for runtime, probe in (("openclaw", openclaw_probe), ("hermes", hermes_probe)):
        for item in probe.get("warnings") or []:
            warning_text = str(item or "").strip()
            if warning_text and warning_text not in warnings:
                warnings.append(warning_text)
        if bool(probe.get("recovery_attempted")):
            fallback = str(probe.get("tool_choice_fallback") or "auto").strip()
            warning = (
                f"{runtime} required a compatibility retry with tool_choice={fallback}."
            )
            if warning not in warnings:
                warnings.append(warning)
        if bool(probe.get("needs_manual_review")):
            warning = f"{runtime} requested manual review during calibration."
            if warning not in warnings:
                warnings.append(warning)
        if str(probe.get("tool_calling") or "").strip().lower() == "degraded":
            warning = (
                f"{runtime} structured tool calls were not confirmed; "
                "direct agent replies remain available and tool use is limited."
            )
            if warning not in warnings:
                warnings.append(warning)

    return {
        "profile_version": PROFILE_VERSION,
        "model_id": canonical_model_id(model_id),
        "configured_at": int(time.time()),
        "family": family or "unknown",
        "context_window": _profile_context_window(
            metadata,
            server_url=server_url,
            model_id=model_id,
        ),
        "chat_template": metadata.get("chat_template")
        or settings.get("chat_template")
        or None,
        "generation_defaults": _profile_generation_defaults(metadata),
        "tool_policy": {
            "always": list(_AGENT_PROFILE_BASE_TOOLS),
            "terminal_on_demand": list(_AGENT_PROFILE_TERMINAL_TOOLS),
            "web_on_demand": list(_AGENT_PROFILE_WEB_TOOLS),
        },
        "openclaw": _profile_runtime_section(
            runtime="openclaw", probe=openclaw_probe, settings=settings
        ),
        "hermes": _profile_runtime_section(
            runtime="hermes", probe=hermes_probe, settings=settings
        ),
        "warnings": warnings,
    }


def _set_agent_profile_job(app: FastAPI, job_id: str, **updates: Any) -> dict[str, Any]:
    lock = app.state.agent_profile_jobs_lock
    with lock:
        jobs = app.state.agent_profile_jobs
        current = dict(jobs.get(job_id) or {})
        current["job_id"] = job_id
        diagnostics = updates.get("diagnostics")
        if isinstance(diagnostics, dict) and isinstance(
            current.get("diagnostics"), dict
        ):
            merged_diagnostics = dict(current["diagnostics"])
            merged_diagnostics.update(diagnostics)
            updates = {**updates, "diagnostics": merged_diagnostics}
        current.update(updates)
        jobs[job_id] = current
        return dict(current)


def _get_agent_profile_job(app: FastAPI, job_id: str) -> dict[str, Any] | None:
    lock = app.state.agent_profile_jobs_lock
    with lock:
        item = app.state.agent_profile_jobs.get(job_id)
        return dict(item) if isinstance(item, dict) else None


_AGENT_PROFILE_JOB_RETENTION_SECONDS = 60 * 60


def _prune_agent_profile_jobs_locked(app: FastAPI, now: float | None = None) -> None:
    """Drop old terminal jobs while retaining recent diagnostics for the UI."""
    current_time = float(now if now is not None else time.time())
    jobs = app.state.agent_profile_jobs
    expired: list[str] = []
    for job_id, item in jobs.items():
        if not isinstance(item, dict) or item.get("state") not in {
            "configured",
            "failed",
        }:
            continue
        ended_at = float(item.get("ended_at", 0.0) or 0.0)
        if ended_at and current_time - ended_at > _AGENT_PROFILE_JOB_RETENTION_SECONDS:
            expired.append(job_id)
    for job_id in expired:
        jobs.pop(job_id, None)


def _claim_agent_profile_job(
    app: FastAPI,
    job: dict[str, Any],
) -> dict[str, Any] | None:
    """Atomically reserve a model configure slot and return an active job.

    The old check-then-submit flow allowed two rapid Configure clicks to both
    pass the active-job check before either one was stored.  Reserving the
    queued record under the same lock closes that race without serialising the
    actual probe work on the request thread.
    """
    lock = app.state.agent_profile_jobs_lock
    with lock:
        _prune_agent_profile_jobs_locked(app)
        model_id = str(job.get("model") or "")
        for item in app.state.agent_profile_jobs.values():
            if (
                isinstance(item, dict)
                and item.get("model") == model_id
                and item.get("state") in {"queued", "running"}
            ):
                return dict(item)
        job_id = str(job.get("job_id") or "").strip()
        if not job_id:
            raise ValueError("Agent Profile job ID cannot be empty.")
        app.state.agent_profile_jobs[job_id] = dict(job)
        return None


def _find_active_agent_profile_job(
    app: FastAPI, model_id: str
) -> dict[str, Any] | None:
    lock = app.state.agent_profile_jobs_lock
    with lock:
        _prune_agent_profile_jobs_locked(app)
        for item in app.state.agent_profile_jobs.values():
            if (
                isinstance(item, dict)
                and item.get("model") == model_id
                and item.get("state") in {"queued", "running"}
            ):
                return dict(item)
    return None


def _run_profile_probe_with_retry(
    app: FastAPI,
    *,
    job_id: str,
    runtime: str,
    server_url: str,
    selected_model: str,
    tool_call_parser: str | None = None,
    reasoning_parser: str | None = None,
    chat_template: str | None = None,
) -> dict[str, Any]:
    """Retry only transient bridge failures during the initial cold start."""
    result: dict[str, Any] = {}
    for attempt in (1, 2):
        try:
            result = _run_agent_runtime_probe(
                server_url=server_url,
                selected_model=selected_model,
                runtime=runtime,
                force=True,
                tool_call_parser=tool_call_parser,
                reasoning_parser=reasoning_parser,
                chat_template=chat_template,
            )
        except Exception as exc:
            result = {
                "ok": False,
                "runtime": runtime,
                "status": 500,
                "reason": f"Bridge probe execution failed: {exc}",
                "needs_manual_review": True,
                "failures": [f"Bridge probe execution failed: {exc}"],
            }
        if bool(result.get("ok")):
            return result
        try:
            status = int(result.get("status") or 0)
        except (TypeError, ValueError):
            status = 0
        # A 422 is a real tool/parser incompatibility and should be surfaced
        # immediately.  5xx failures commonly come from a cold bridge/runtime.
        if status < 500 or attempt >= 2:
            return result
        _set_agent_profile_job(
            app,
            job_id,
            diagnostics={
                **(_get_agent_profile_job(app, job_id) or {}).get("diagnostics", {}),
                f"{runtime}_retry": {
                    "attempt": attempt + 1,
                    "reason": str(result.get("reason") or "transient probe failure"),
                },
            },
        )
        time.sleep(0.25)
    return result


def _run_agent_profile_configure_job(
    app: FastAPI,
    *,
    job_id: str,
    model_id: str,
    server_url: str,
) -> None:
    """Calibrate both runtimes before writing a model-level profile.

    A runtime probe can be ``passed`` (structured tool calls confirmed) or
    ``degraded`` (the model answered normally but did not expose the structured
    tool-call contract). Both are healthy enough for direct agent replies;
    only a real health/runtime failure blocks persistence.
    """
    try:
        _set_agent_profile_job(
            app,
            job_id,
            state="running",
            phase="metadata",
            progress=0,
        )
        try:
            metadata = read_model_metadata(model_id, allow_download=False)
        except Exception as exc:
            # Configure must remain useful when a model has no local metadata
            # cache.  The parser registry and conservative context fallback can
            # still produce a valid profile without downloading weights.
            metadata = {"warnings": [f"Local metadata unavailable: {exc}"]}
        if not isinstance(metadata, dict):
            metadata = {"warnings": ["Local metadata returned an invalid shape."]}
        elif not isinstance(metadata.get("warnings"), list):
            metadata["warnings"] = []
        try:
            family = detect_model_family(model_id, metadata)
        except Exception as exc:
            family = "unknown"
            metadata.setdefault("warnings", []).append(
                f"Model family detection unavailable: {exc}"
            )
        rules, registry_warnings = load_registry()
        matched_rule, _needs_manual_review = match_rule(
            model_id=model_id,
            family=family,
            metadata=metadata,
            rules=rules,
        )
        settings = resolve_settings(
            family=family,
            matched_rule=matched_rule,
            override_tool_call_parser=None,
            override_reasoning_parser=None,
            override_chat_template=None,
        )
        _set_agent_profile_job(app, job_id, phase="metadata", progress=20)

        _set_agent_profile_job(app, job_id, phase="openclaw probe", progress=20)
        openclaw_probe = _run_profile_probe_with_retry(
            app,
            job_id=job_id,
            server_url=server_url,
            selected_model=model_id,
            runtime="openclaw",
            tool_call_parser=str(settings.get("tool_call_parser") or "").strip()
            or None,
            reasoning_parser=str(settings.get("reasoning_parser") or "").strip()
            or None,
            chat_template=str(settings.get("chat_template") or "").strip() or None,
        )
        _set_agent_profile_job(
            app,
            job_id,
            phase="openclaw probe",
            progress=55,
            diagnostics={"openclaw": _profile_probe_summary(openclaw_probe)},
        )

        _set_agent_profile_job(app, job_id, phase="hermes probe", progress=55)
        hermes_probe = _run_profile_probe_with_retry(
            app,
            job_id=job_id,
            server_url=server_url,
            selected_model=model_id,
            runtime="hermes",
            tool_call_parser=str(settings.get("tool_call_parser") or "").strip()
            or None,
            reasoning_parser=str(settings.get("reasoning_parser") or "").strip()
            or None,
            chat_template=str(settings.get("chat_template") or "").strip() or None,
        )
        diagnostics = {
            "openclaw": _profile_probe_summary(openclaw_probe),
            "hermes": _profile_probe_summary(hermes_probe),
        }
        _set_agent_profile_job(
            app,
            job_id,
            phase="hermes probe",
            progress=90,
            diagnostics=diagnostics,
        )

        failed_runtimes = [
            runtime
            for runtime, probe in (
                ("OpenClaw", openclaw_probe),
                ("Hermes", hermes_probe),
            )
            if not bool(probe.get("ok"))
        ]
        if failed_runtimes:
            reasons = [
                _format_agent_probe_failure(runtime, probe)
                for runtime, probe in (
                    ("OpenClaw", openclaw_probe),
                    ("Hermes", hermes_probe),
                )
                if not bool(probe.get("ok"))
            ]
            _set_agent_profile_job(
                app,
                job_id,
                state="failed",
                phase="failed",
                progress=90,
                ended_at=time.time(),
                error="; ".join(reasons)
                or f"{', '.join(failed_runtimes)} calibration failed.",
                diagnostics=diagnostics,
            )
            return

        _set_agent_profile_job(app, job_id, phase="persist", progress=90)
        profile = _build_agent_profile(
            model_id=model_id,
            server_url=server_url,
            metadata=metadata,
            family=family,
            settings=settings,
            registry_warnings=registry_warnings,
            openclaw_probe=openclaw_probe,
            hermes_probe=hermes_probe,
        )
        saved_profile = save_agent_profile(profile)
        _set_agent_profile_job(
            app,
            job_id,
            state="configured",
            phase="persist",
            progress=100,
            ended_at=time.time(),
            profile=saved_profile,
            error="",
            diagnostics=diagnostics,
        )
    except Exception as exc:
        _set_agent_profile_job(
            app,
            job_id,
            state="failed",
            phase="failed",
            progress=90,
            ended_at=time.time(),
            error=f"Profile configuration failed: {exc}",
            diagnostics={
                "failure_category": "runtime_environment",
                "fix_suggestions": [
                    "Check the local server and bridge dependencies, then retry Configure."
                ],
            },
        )


def _configured_agent_profile(model_id: str) -> dict[str, Any] | None:
    """Return the valid model-level profile required for agent-mode requests."""
    return load_agent_profile(canonical_model_id(model_id))


def _profile_positive_int(
    profile: dict[str, Any] | None,
    *,
    runtime: str | None,
    key: str,
) -> int | None:
    """Read one bounded positive integer from a validated profile."""
    source: Any = profile
    if runtime:
        source = profile.get(runtime) if isinstance(profile, dict) else None
    if not isinstance(source, dict):
        return None
    value = _coerce_int(source.get(key))
    return value if value is not None and value > 0 else None


def _profile_tool_policy(profile: dict[str, Any] | None) -> dict[str, list[str]]:
    """Return a safe normalised tool policy, with profile defaults as fallback."""
    configured = profile.get("tool_policy") if isinstance(profile, dict) else None
    configured = configured if isinstance(configured, dict) else {}
    defaults = {
        "always": _AGENT_PROFILE_BASE_TOOLS,
        "terminal_on_demand": _AGENT_PROFILE_TERMINAL_TOOLS,
        "web_on_demand": _AGENT_PROFILE_WEB_TOOLS,
    }
    policy: dict[str, list[str]] = {}
    for key, fallback in defaults.items():
        raw = configured.get(key)
        values = raw if isinstance(raw, list) else fallback
        seen: set[str] = set()
        cleaned: list[str] = []
        for item in values:
            name = str(item or "").strip().lower()
            if name and name not in seen:
                seen.add(name)
                cleaned.append(name)
        policy[key] = cleaned
    return policy


def _profile_runtime_supports_tools(
    profile: dict[str, Any] | None,
    runtime: str,
) -> bool:
    """Return whether a calibrated runtime passed structured tool probing."""
    section = profile.get(runtime) if isinstance(profile, dict) else None
    if not isinstance(section, dict) or not bool(section.get("ready")):
        return False
    state = str(section.get("tool_calling") or "").strip().lower()
    # Older in-memory/fixture profiles predate the capability field.  A
    # runtime explicitly marked ready by that schema keeps the historical
    # behaviour; newly calibrated degraded profiles carry the explicit state.
    return state in {"", "passed"}


def _agent_generation_settings(
    profile: dict[str, Any] | None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge model profile defaults with a current user's generation choices."""
    allowed = {
        "temperature",
        "top_p",
        "top_k",
        "typical_p",
        "repetition_penalty",
        "max_new_tokens",
        "max_tokens",
    }
    merged: dict[str, Any] = {}
    defaults = profile.get("generation_defaults") if isinstance(profile, dict) else None
    for source in (defaults, overrides):
        if not isinstance(source, dict):
            continue
        for key, value in source.items():
            if key not in allowed or isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                merged[key] = value
    return merged


def create_app(
    default_server_url: str,
    default_max_tokens: int,
    default_temperature: float,
) -> FastAPI:
    """Create the FastAPI backend used by the native desktop UI."""
    app = FastAPI(title="token-workshed UI API", version=TOKEN_WORKSHED_VERSION)
    app.state.server_url = _normalize_server_url(default_server_url)
    app.state.max_tokens = _clamp_max_tokens(default_max_tokens, 512)
    app.state.temperature = _clamp_temperature(default_temperature, 0.7)
    # desktop_ui fills this when --allow-remote-ui is active. Keeping the
    # profile API on this app means native and browser controllers share the
    # exact same routes while still respecting remote manager authentication.
    app.state.manager_api_token = None
    app.state.agent_profile_jobs_lock = threading.Lock()
    app.state.agent_profile_jobs: dict[str, dict[str, Any]] = {}
    app.state.agent_profile_executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="agent-profile",
    )
    app.state.openclaw_warmup_lock = threading.Lock()
    app.state.openclaw_warmup_executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="openclaw-warmup",
    )
    app.state.openclaw_warmup_future = None
    app.state.openclaw_warmup_status = {
        "state": "idle",
        "started_at": 0.0,
        "ended_at": 0.0,
        "latency_ms": None,
        "error": "",
    }

    @app.on_event("shutdown")
    async def _shutdown_openclaw_runtime() -> None:
        executor = getattr(app.state, "openclaw_warmup_executor", None)
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
        profile_executor = getattr(app.state, "agent_profile_executor", None)
        if profile_executor is not None:
            profile_executor.shutdown(wait=False, cancel_futures=True)
        _stop_openclaw_gateway_process()

    @app.get("/")
    async def index() -> dict[str, Any]:
        return {
            "ok": True,
            "app": "token-workshed UI API",
            "version": TOKEN_WORKSHED_VERSION,
        }

    @app.get("/api/config")
    async def config() -> UIConfig:
        return UIConfig(
            server_url=app.state.server_url,
            max_tokens=app.state.max_tokens,
            temperature=app.state.temperature,
            token_workshed_version=TOKEN_WORKSHED_VERSION,
            vllm_mlx_version=VLLM_MLX_VERSION,
            manager_auth_required=bool(
                getattr(app.state, "manager_auth_required", False)
            ),
            openclaw_enabled=False,
            cypherclaw_enabled=False,
            agent_runtime="auto",
        )

    @app.post("/api/openclaw/warmup")
    async def openclaw_warmup(payload: OpenClawWarmupRequest) -> dict[str, Any]:
        """Start/inspect OpenClaw warmup job for faster first turn."""
        raw_url = (
            payload.server_url
            if payload.server_url is not None
            else app.state.server_url
        )
        try:
            target = _normalize_server_url(raw_url)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        selected_model = (payload.model or "").strip() or "default"
        lock = app.state.openclaw_warmup_lock
        with lock:
            future = getattr(app.state, "openclaw_warmup_future", None)
            status = dict(getattr(app.state, "openclaw_warmup_status", {}) or {})

            if future is not None and future.done():
                app.state.openclaw_warmup_future = None
                try:
                    result = future.result()
                    status.update(
                        {
                            "state": "ready",
                            "ended_at": time.time(),
                            "latency_ms": result.get("latency_ms"),
                            "error": "",
                        }
                    )
                except Exception as exc:
                    status.update(
                        {
                            "state": "failed",
                            "ended_at": time.time(),
                            "latency_ms": None,
                            "error": str(exc),
                        }
                    )
                app.state.openclaw_warmup_status = status
                future = None

            if future is not None:
                status["state"] = "running"
                app.state.openclaw_warmup_status = status
                return {
                    "ok": True,
                    "started": False,
                    "status": status,
                }

            status = {
                "state": "running",
                "started_at": time.time(),
                "ended_at": 0.0,
                "latency_ms": None,
                "error": "",
            }
            app.state.openclaw_warmup_status = status
            try:
                app.state.openclaw_warmup_future = (
                    app.state.openclaw_warmup_executor.submit(
                        _run_openclaw_warmup,
                        server_url=target,
                        selected_model=selected_model,
                    )
                )
            except Exception as exc:
                status.update(
                    {
                        "state": "failed",
                        "ended_at": time.time(),
                        "error": str(exc),
                    }
                )
                app.state.openclaw_warmup_status = status
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to start openclaw warmup: {exc}",
                ) from exc

            return {
                "ok": True,
                "started": True,
                "status": status,
            }

    @app.post("/api/cypherclaw/warmup")
    async def cypherclaw_warmup_alias(payload: OpenClawWarmupRequest) -> dict[str, Any]:
        """Backward-compatible alias for older frontend builds."""
        return await openclaw_warmup(payload)

    @app.post("/api/agent/probe")
    async def agent_probe(
        request: Request, payload: AgentProbeRequest
    ) -> dict[str, Any]:
        """Probe tool-call compatibility for selected runtime/model/backend."""
        _enforce_agent_profile_auth(request, app)
        raw_url = (
            payload.server_url
            if payload.server_url is not None
            else app.state.server_url
        )
        try:
            target = _normalize_server_url(raw_url)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        selected_model = (payload.model or "").strip() or "default"
        runtime = (
            "hermes"
            if _resolve_agent_runtime(payload.runtime) == "hermes"
            else "openclaw"
        )
        result = await run_in_threadpool(
            _run_agent_runtime_probe,
            server_url=target,
            selected_model=selected_model,
            runtime=runtime,
            force=bool(payload.force),
        )
        return {
            "ok": bool(result.get("ok")),
            "runtime": runtime,
            "server_url": target,
            "model": selected_model,
            "reason": str(result.get("reason") or "").strip(),
            "status": result.get("status"),
            "cached": bool(result.get("cached")),
            "checked_at": result.get("checked_at"),
            "agent_ready": bool(result.get("agent_ready")),
            "tool_calling": str(result.get("tool_calling") or "failed"),
            "parser_used": result.get("parser_used"),
            "reasoning_parser_used": result.get("reasoning_parser_used"),
            "chat_template_used": result.get("chat_template_used"),
            "needs_manual_review": bool(result.get("needs_manual_review")),
            "failures": (
                result.get("failures")
                if isinstance(result.get("failures"), list)
                else []
            ),
            "warnings": (
                result.get("warnings")
                if isinstance(result.get("warnings"), list)
                else []
            ),
            "failure_category": str(result.get("failure_category") or ""),
            "recovery_attempted": bool(result.get("recovery_attempted")),
            "tool_choice_fallback": result.get("tool_choice_fallback"),
            "fix_suggestions": (
                result.get("fix_suggestions")
                if isinstance(result.get("fix_suggestions"), list)
                else []
            ),
        }

    @app.get("/api/agent/profile")
    async def agent_profile_get(
        request: Request, model: str = Query(...)
    ) -> dict[str, Any]:
        """Read the valid cached Agent Profile for a model, if one exists."""
        _enforce_agent_profile_auth(request, app)
        canonical = canonical_model_id(model)
        if not canonical:
            raise HTTPException(status_code=400, detail="Model name cannot be empty.")
        profile = await run_in_threadpool(_configured_agent_profile, canonical)
        return {
            "ok": True,
            "model": canonical,
            "configured": profile is not None,
            "profile": profile,
        }

    @app.post("/api/agent/profile/configure")
    async def agent_profile_configure(
        request: Request,
        payload: AgentProfileConfigureRequest,
    ) -> dict[str, Any]:
        """Start a background OpenClaw + Hermes calibration job."""
        _enforce_agent_profile_auth(request, app)
        canonical = canonical_model_id(payload.model)
        if not canonical:
            raise HTTPException(status_code=400, detail="Model name cannot be empty.")
        raw_url = (
            payload.server_url
            if payload.server_url is not None
            else app.state.server_url
        )
        try:
            target = _normalize_server_url(raw_url)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        active_job = await run_in_threadpool(
            _find_active_agent_profile_job, app, canonical
        )
        if active_job is not None:
            return {
                "ok": True,
                "model": canonical,
                "job_id": active_job.get("job_id"),
                "started": False,
                "reused": False,
                "job": active_job,
            }

        if not payload.force:
            existing = await run_in_threadpool(_configured_agent_profile, canonical)
            if existing is not None:
                return {
                    "ok": True,
                    "model": canonical,
                    "job_id": None,
                    "started": False,
                    "reused": True,
                    "profile": existing,
                }
        job_id = uuid.uuid4().hex
        job = {
            "job_id": job_id,
            "model": canonical,
            "server_url": target,
            "force": bool(payload.force),
            "state": "queued",
            "phase": "metadata",
            "progress": 0,
            "started_at": time.time(),
            "ended_at": 0.0,
            "error": "",
            "diagnostics": {},
        }
        active_job = _claim_agent_profile_job(app, job)
        if active_job is not None:
            return {
                "ok": True,
                "model": canonical,
                "job_id": active_job.get("job_id"),
                "started": False,
                "reused": False,
                "job": active_job,
            }
        try:
            # Recalibration is transactional: keep an existing valid profile
            # available while probes run, and replace it only after both
            # runtimes complete successfully (or in explicitly degraded mode).
            # A transient failure must not take a previously usable model
            # offline.
            app.state.agent_profile_executor.submit(
                _run_agent_profile_configure_job,
                app,
                job_id=job_id,
                model_id=canonical,
                server_url=target,
            )
        except Exception as exc:
            _set_agent_profile_job(
                app,
                job_id,
                state="failed",
                phase="failed",
                ended_at=time.time(),
                error=f"Could not start profile configuration: {exc}",
            )
            raise HTTPException(
                status_code=500, detail="Could not start profile configuration."
            ) from exc
        return {
            "ok": True,
            "model": canonical,
            "job_id": job_id,
            "started": True,
            "reused": False,
            "job": _get_agent_profile_job(app, job_id),
        }

    @app.get("/api/agent/profile/configure/{job_id}")
    async def agent_profile_configure_status(
        request: Request, job_id: str
    ) -> dict[str, Any]:
        """Read configure progress without exposing bridge command details."""
        _enforce_agent_profile_auth(request, app)
        job = _get_agent_profile_job(app, job_id)
        if job is None:
            raise HTTPException(
                status_code=404, detail="Agent Profile job was not found."
            )
        return {"ok": True, "job": job, **job}

    @app.delete("/api/agent/profile")
    async def agent_profile_delete(
        request: Request, model: str = Query(...)
    ) -> dict[str, Any]:
        """Delete one model's cached profile without touching model weights."""
        _enforce_agent_profile_auth(request, app)
        canonical = canonical_model_id(model)
        if not canonical:
            raise HTTPException(status_code=400, detail="Model name cannot be empty.")
        deleted = await run_in_threadpool(delete_agent_profile, canonical)
        return {
            "ok": True,
            "model": canonical,
            "deleted": deleted,
            "message": (
                "Agent Profile deleted." if deleted else "No Agent Profile was stored."
            ),
        }

    @app.get("/api/status")
    async def status(server_url: str | None = Query(default=None)) -> dict[str, Any]:
        raw_url = server_url if server_url is not None else app.state.server_url

        try:
            target = _normalize_server_url(raw_url)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        try:
            models, model_name = await run_in_threadpool(_fetch_models, target)
            model_capabilities = _build_model_capabilities(models)

            runtime_metrics: dict[str, Any] | None = None
            runtime_error: str | None = None
            try:
                runtime_metrics = await run_in_threadpool(
                    _fetch_runtime_metrics, target
                )
            except requests.RequestException as exc:
                runtime_error = str(exc)

            return {
                "ok": True,
                "server_url": target,
                "model": model_name,
                "models": models,
                "model_count": len(models),
                "model_capabilities": model_capabilities,
                "deep_thinking_models": [
                    model_id
                    for model_id, cap in model_capabilities.items()
                    if bool(cap.get("deep_thinking"))
                ],
                "runtime": runtime_metrics or {},
                "runtime_error": runtime_error,
            }
        except requests.RequestException as exc:
            return {
                "ok": False,
                "server_url": target,
                "error": f"Cannot reach {target}/v1/models: {exc}",
                "models": [],
                "model_count": 0,
                "model_capabilities": {},
                "deep_thinking_models": [],
                "runtime": {},
            }

    @app.post("/api/chat/title")
    async def chat_title(payload: TitleRequest) -> dict[str, Any]:
        fallback = str(payload.fallback or "").strip() or "New Conversation"
        title, source = await run_in_threadpool(
            _generate_conversation_title,
            payload.messages,
            fallback,
        )
        return {
            "ok": True,
            "title": _normalize_title_text(title, fallback),
            "source": source,
        }

    @app.post("/api/chat")
    async def chat(payload: ChatRequest) -> dict[str, Any]:
        user_message = payload.message.strip()
        user_content = payload.user_content
        openclaw_enabled = _is_openclaw_enabled(payload)
        agent_runtime_candidates = _resolve_agent_runtime_candidates(
            payload.agent_runtime
        )

        raw_url = (
            payload.server_url
            if payload.server_url is not None
            else app.state.server_url
        )
        try:
            target = _normalize_server_url(raw_url)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        max_tokens = _clamp_max_tokens(payload.max_tokens, app.state.max_tokens)
        temperature = _clamp_temperature(payload.temperature, app.state.temperature)
        original_system_prompt = (payload.system_prompt or "").strip()
        system_prompt = original_system_prompt
        direct_fallback_system_prompt = original_system_prompt
        selected_model = (payload.model or "").strip() or "default"
        model_capability = _detect_model_capability(selected_model)
        use_openclaw_agent = openclaw_enabled
        agent_profile: dict[str, Any] | None = None
        if use_openclaw_agent:
            agent_profile = _configured_agent_profile(selected_model)
            if agent_profile is None:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "message": (
                            "Configure AI agents for this model before enabling "
                            "OpenClaw or Hermes."
                        ),
                        "error_type": "agent_profile_required",
                        "model": canonical_model_id(selected_model),
                        "retryable": False,
                    },
                )

        if use_openclaw_agent:
            system_prompt = _compose_system_prompt_for_openclaw(system_prompt, True)
        else:
            system_prompt, model_capability = _compose_system_prompt_for_model(
                system_prompt,
                selected_model,
            )
            direct_fallback_system_prompt = system_prompt
            max_tokens, temperature = _tune_generation_for_model(
                max_tokens=max_tokens,
                temperature=temperature,
                model_capability=model_capability,
            )

        selected_agent_runtime = (
            agent_runtime_candidates[0] if agent_runtime_candidates else "openclaw"
        )
        probe_warning = ""

        if use_openclaw_agent:
            try:
                if _should_run_agent_probe_on_chat(False):
                    try:
                        probe_result = await run_in_threadpool(
                            _run_agent_runtime_probe,
                            server_url=target,
                            selected_model=selected_model,
                            runtime=selected_agent_runtime,
                            force=False,
                        )
                    except Exception as probe_exc:
                        probe_warning = _truncate_error_text(
                            str(probe_exc) or "tool-call probe failed",
                            limit=320,
                        )
                        logger.warning(
                            "[agent_probe] runtime=%s model=%s probe exception: %s",
                            selected_agent_runtime,
                            selected_model,
                            probe_warning,
                        )
                    else:
                        if not bool(probe_result.get("ok")):
                            probe_warning = _truncate_error_text(
                                str(
                                    probe_result.get("reason")
                                    or "tool-call probe failed"
                                ),
                                limit=320,
                            )
                            logger.warning(
                                "[agent_probe] advisory failure for runtime=%s model=%s: %s",
                                selected_agent_runtime,
                                selected_model,
                                probe_warning,
                            )

                runner = (
                    _run_hermes_turn
                    if selected_agent_runtime == "hermes"
                    else _run_openclaw_turn
                )
                runner_kwargs: dict[str, Any] = {
                    "server_url": target,
                    "selected_model": selected_model,
                    "system_prompt": system_prompt,
                    "user_message": user_message,
                    "user_content": user_content,
                    "conversation_id": payload.conversation_id,
                    "agent_profile": agent_profile,
                    "generation_overrides": {
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                    },
                }
                if selected_agent_runtime == "openclaw":
                    request_options = _resolve_openclaw_turn_request_options(
                        user_message=user_message,
                        tool_mode=payload.tool_mode,
                        allowed_tools=payload.allowed_tools,
                        model_first_chunk_timeout_ms=payload.model_first_chunk_timeout_ms,
                        server_url=target,
                        selected_model=selected_model,
                        agent_runtime=selected_agent_runtime,
                        agent_profile=agent_profile,
                    )
                    runner_kwargs["tool_mode"] = request_options["tool_mode"]
                    runner_kwargs["allowed_tools"] = request_options["allowed_tools"]
                    runner_kwargs["model_first_chunk_timeout_ms"] = request_options[
                        "model_first_chunk_timeout_ms"
                    ]
                openclaw_result = await run_in_threadpool(
                    runner,
                    **runner_kwargs,
                )
                raw_reply = str(openclaw_result.get("reply", "")).strip()
                if not raw_reply:
                    runtime_label = _agent_runtime_label(selected_agent_runtime)
                    raise RuntimeError(f"{runtime_label} returned an empty response.")
                thinking_text, answer_text = _split_openclaw_reasoning_and_answer(
                    raw_reply
                )
                final_answer_text = _sanitize_assistant_output_text(
                    str(answer_text or "").strip()
                )
                if not final_answer_text:
                    runtime_label = _agent_runtime_label(selected_agent_runtime)
                    raise RuntimeError(
                        f"{runtime_label} returned no final answer. "
                        "Try increasing max tokens or ask for a shorter thinking process."
                    )
            except Exception as exc:
                error_detail = _build_agent_runtime_error_detail(
                    runtime=selected_agent_runtime,
                    selected_model=selected_model,
                    reason=str(exc),
                    probe_warning=probe_warning,
                )
                logger.warning(
                    "[agent_runtime] failed for runtime=%s model=%s: %s",
                    selected_agent_runtime,
                    selected_model,
                    error_detail.get("message"),
                )
                raise HTTPException(status_code=502, detail=error_detail) from exc
            else:
                return {
                    "ok": True,
                    "reply": final_answer_text,
                    "model": selected_model,
                    "request_model": selected_model,
                    "model_capability": model_capability,
                    "effective_params": {
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                    },
                    "metrics": openclaw_result["metrics"],
                    "thinking": thinking_text,
                    "answer": final_answer_text,
                    "openclaw": {
                        "enabled": True,
                        "runtime": selected_agent_runtime,
                        "session_id": openclaw_result["session_id"],
                        "stop_reason": openclaw_result["stop_reason"],
                        "agent_model": openclaw_result["model"],
                        "probe_warning": probe_warning or None,
                    },
                    "cypherclaw": {
                        "enabled": True,
                        "runtime": selected_agent_runtime,
                        "session_id": openclaw_result["session_id"],
                        "stop_reason": openclaw_result["stop_reason"],
                        "agent_model": openclaw_result["model"],
                        "probe_warning": probe_warning or None,
                    },
                }

        try:
            direct_result = await run_in_threadpool(
                _run_direct_chat_completion,
                target=target,
                selected_model=selected_model,
                history=payload.history,
                system_prompt=direct_fallback_system_prompt,
                user_message=user_message,
                user_content=user_content,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout_seconds=300,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=_direct_chat_exception_status_code(exc),
                detail=_direct_chat_exception_detail(exc),
            ) from exc

        response_payload: dict[str, Any] = {
            "ok": True,
            "reply": direct_result["reply"],
            "model": direct_result["model"],
            "request_model": selected_model,
            "finish_reason": direct_result["finish_reason"],
            "stop_reason": direct_result["stop_reason"],
            "model_capability": model_capability,
            "effective_params": {
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            "metrics": direct_result["metrics"],
        }

        return response_payload

    @app.post("/api/chat/stream")
    async def chat_stream(payload: ChatRequest) -> StreamingResponse:
        """Stream chat response with separate thinking and answer deltas."""
        user_message = payload.message.strip()
        user_content = payload.user_content

        raw_url = (
            payload.server_url
            if payload.server_url is not None
            else app.state.server_url
        )
        try:
            target = _normalize_server_url(raw_url)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        max_tokens = _clamp_max_tokens(payload.max_tokens, app.state.max_tokens)
        temperature = _clamp_temperature(payload.temperature, app.state.temperature)
        selected_model = (payload.model or "").strip() or "default"
        original_system_prompt = (payload.system_prompt or "").strip()
        system_prompt = original_system_prompt
        direct_fallback_system_prompt = original_system_prompt
        model_capability = _detect_model_capability(selected_model)
        openclaw_enabled = _is_openclaw_enabled(payload)
        runtime_candidates = _resolve_agent_runtime_candidates(payload.agent_runtime)
        use_openclaw_agent = openclaw_enabled
        agent_profile: dict[str, Any] | None = None
        if use_openclaw_agent:
            agent_profile = _configured_agent_profile(selected_model)
            if agent_profile is None:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "message": (
                            "Configure AI agents for this model before enabling "
                            "OpenClaw or Hermes."
                        ),
                        "error_type": "agent_profile_required",
                        "model": canonical_model_id(selected_model),
                        "retryable": False,
                    },
                )
        if use_openclaw_agent:
            system_prompt = _compose_system_prompt_for_openclaw(system_prompt, True)
        else:
            system_prompt, model_capability = _compose_system_prompt_for_model(
                system_prompt,
                selected_model,
            )
            direct_fallback_system_prompt = system_prompt
            max_tokens, temperature = _tune_generation_for_model(
                max_tokens=max_tokens,
                temperature=temperature,
                model_capability=model_capability,
            )

        if use_openclaw_agent:
            selected_runtime = runtime_candidates[0]
            probe_warning = ""
            if _should_run_agent_probe_on_chat(False):
                try:
                    probe_result = await run_in_threadpool(
                        _run_agent_runtime_probe,
                        server_url=target,
                        selected_model=selected_model,
                        runtime=selected_runtime,
                        force=False,
                    )
                except Exception as probe_exc:
                    probe_warning = _truncate_error_text(
                        str(probe_exc) or "tool-call probe failed",
                        limit=320,
                    )
                    logger.warning(
                        "[agent_probe] runtime=%s model=%s probe exception: %s",
                        selected_runtime,
                        selected_model,
                        probe_warning,
                    )
                    probe_result = {"ok": False, "reason": probe_warning}
                if not bool(probe_result.get("ok")):
                    probe_warning = _truncate_error_text(
                        str(probe_result.get("reason") or "tool-call probe failed"),
                        limit=320,
                    )
                    logger.warning(
                        "[agent_probe] advisory failure for runtime=%s model=%s: %s",
                        selected_runtime,
                        selected_model,
                        probe_warning,
                    )

            def stream_openclaw_events():
                yield _stream_json_line(
                    {
                        "type": "start",
                        "request_model": selected_model,
                        "model_capability": model_capability,
                        "effective_params": {
                            "max_tokens": max_tokens,
                            "temperature": temperature,
                        },
                    }
                )
                runtime_stage = _agent_runtime_stage(selected_runtime)
                runtime_label = _agent_runtime_label(selected_runtime)

                def emit_agent_error(reason: str):
                    detail = _build_agent_runtime_error_detail(
                        runtime=selected_runtime,
                        selected_model=selected_model,
                        reason=reason,
                        probe_warning=probe_warning,
                    )
                    yield _stream_json_line(
                        {
                            "type": "error",
                            "message": str(
                                detail.get("message") or "Agent runtime failed."
                            ),
                            "error_type": detail.get("error_type"),
                            "runtime": detail.get("runtime"),
                            "model": detail.get("model"),
                            "reason": detail.get("reason"),
                            "retryable": bool(detail.get("retryable")),
                            "probe_warning": detail.get("probe_warning"),
                        }
                    )

                yield _stream_json_line(
                    {
                        "type": "thinking_delta",
                        "text": f"[{runtime_stage}] starting agent process...\n",
                    }
                )
                yield _stream_json_line(
                    {
                        "type": "thinking_delta",
                        "text": f"[{runtime_stage}] running...\n",
                    }
                )
                if probe_warning:
                    yield _stream_json_line(
                        {
                            "type": "thinking_delta",
                            "text": (
                                f"[{runtime_stage}] probe warning: "
                                f"{_truncate_error_text(probe_warning, limit=220)}\n"
                            ),
                        }
                    )

                runner = (
                    _run_hermes_turn
                    if selected_runtime == "hermes"
                    else _run_openclaw_turn
                )
                openclaw_result: dict[str, Any] | None = None
                cancel_event = threading.Event()
                executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                future: concurrent.futures.Future[dict[str, Any]] | None = None
                try:
                    runner_kwargs: dict[str, Any] = {
                        "server_url": target,
                        "selected_model": selected_model,
                        "system_prompt": system_prompt,
                        "user_message": user_message,
                        "user_content": user_content,
                        "conversation_id": payload.conversation_id,
                        "agent_profile": agent_profile,
                        "generation_overrides": {
                            "max_tokens": max_tokens,
                            "temperature": temperature,
                        },
                    }
                    if selected_runtime == "openclaw":
                        session_id = str(
                            payload.session_id or ""
                        ).strip() or _normalize_openclaw_session_id(
                            payload.conversation_id,
                            selected_model=selected_model,
                            runtime="openclaw",
                        )
                        run_id = str(
                            payload.run_id or ""
                        ).strip() or _make_openclaw_run_id(session_id)
                        request_options = _resolve_openclaw_turn_request_options(
                            user_message=user_message,
                            tool_mode=payload.tool_mode,
                            allowed_tools=payload.allowed_tools,
                            model_first_chunk_timeout_ms=payload.model_first_chunk_timeout_ms,
                            server_url=target,
                            selected_model=selected_model,
                            agent_runtime=selected_runtime,
                            agent_profile=agent_profile,
                        )
                        runner_kwargs["cancel_event"] = cancel_event
                        runner_kwargs["session_id"] = session_id
                        runner_kwargs["run_id"] = run_id
                        runner_kwargs["tool_mode"] = request_options["tool_mode"]
                        runner_kwargs["allowed_tools"] = request_options[
                            "allowed_tools"
                        ]
                        runner_kwargs["model_first_chunk_timeout_ms"] = request_options[
                            "model_first_chunk_timeout_ms"
                        ]
                    future = executor.submit(runner, **runner_kwargs)
                    last_progress_ping = time.perf_counter()
                    while True:
                        try:
                            openclaw_result = future.result(timeout=1.0)
                            break
                        except concurrent.futures.TimeoutError:
                            now = time.perf_counter()
                            if (now - last_progress_ping) >= 12.0:
                                last_progress_ping = now
                                yield _stream_json_line(
                                    {
                                        "type": "thinking_delta",
                                        "text": f"[{runtime_stage}] still working...\n",
                                    }
                                )
                            continue
                        except ValueError as exc:
                            yield from emit_agent_error(str(exc))
                            return
                        except RuntimeError as exc:
                            yield from emit_agent_error(
                                f"{runtime_label} failed: {str(exc)}"
                            )
                            return
                        except Exception as exc:
                            yield from emit_agent_error(
                                f"{runtime_label} runtime error: {str(exc)}"
                            )
                            return
                finally:
                    if future is not None and not future.done():
                        cancel_event.set()
                        future.cancel()
                    executor.shutdown(wait=False, cancel_futures=True)

                if openclaw_result is None:
                    yield from emit_agent_error(
                        f"{runtime_label} failed without a response."
                    )
                    return

                reply_text = str(openclaw_result.get("reply", "")).strip()
                if _is_openclaw_progress_only_reply(
                    reply_text=reply_text,
                    user_message=user_message,
                ):
                    if reply_text:
                        if not reply_text.endswith("\n"):
                            reply_text += "\n"
                        yield _stream_json_line(
                            {"type": "thinking_delta", "text": reply_text}
                        )
                    yield from emit_agent_error(
                        f"{runtime_label} returned intermediate tool-progress text without a final answer."
                    )
                    return
                thinking_text, answer_text = _split_openclaw_reasoning_and_answer(
                    reply_text
                )
                if thinking_text:
                    if not thinking_text.endswith("\n"):
                        thinking_text += "\n"
                    yield _stream_json_line(
                        {"type": "thinking_delta", "text": thinking_text}
                    )
                final_answer_text = _sanitize_assistant_output_text(
                    str(answer_text or "").strip()
                )
                if not final_answer_text:
                    yield from emit_agent_error(
                        f"{runtime_label} returned no final answer. "
                        "Try increasing max tokens or ask for a shorter thinking process."
                    )
                    return
                yield _stream_json_line(
                    {"type": "answer_delta", "text": final_answer_text}
                )
                yield _stream_json_line(
                    {
                        "type": "done",
                        "ok": True,
                        "model": selected_model,
                        "request_model": selected_model,
                        "finish_reason": str(openclaw_result.get("stop_reason") or "")
                        .strip()
                        .lower()
                        or None,
                        "stop_reason": str(openclaw_result.get("stop_reason") or "")
                        .strip()
                        .lower()
                        or None,
                        "model_capability": model_capability,
                        "effective_params": {
                            "max_tokens": max_tokens,
                            "temperature": temperature,
                        },
                        "metrics": openclaw_result.get("metrics", {}),
                        "reply": final_answer_text,
                        "answer": final_answer_text,
                        "thinking": thinking_text,
                        "openclaw": {
                            "enabled": True,
                            "runtime": selected_runtime,
                            "session_id": openclaw_result.get("session_id"),
                            "stop_reason": openclaw_result.get("stop_reason"),
                            "agent_model": openclaw_result.get("model", selected_model),
                            "probe_warning": probe_warning or None,
                        },
                        "cypherclaw": {
                            "enabled": True,
                            "runtime": selected_runtime,
                            "session_id": openclaw_result.get("session_id"),
                            "stop_reason": openclaw_result.get("stop_reason"),
                            "agent_model": openclaw_result.get("model", selected_model),
                            "probe_warning": probe_warning or None,
                        },
                    }
                )

            return StreamingResponse(
                stream_openclaw_events(),
                media_type="application/x-ndjson",
            )

        try:
            messages = _build_messages(
                payload.history,
                system_prompt,
                user_message,
                user_content=user_content,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        def stream_events():
            request_start = time.perf_counter()
            prompt_tokens: int | None = None
            completion_tokens: int | None = None
            total_tokens: int | None = None
            connect_attempts = 1
            finish_reason = ""

            deep_thinking = bool(model_capability.get("deep_thinking"))
            section_mode = "auto" if deep_thinking else "answer"
            pending_content = ""
            explicit_reasoning_seen = False
            thinking_header_stripped = False
            answer_header_stripped = False

            yield _stream_json_line(
                {
                    "type": "start",
                    "request_model": selected_model,
                    "model_capability": model_capability,
                    "effective_params": {
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                    },
                }
            )

            try:
                response, connect_attempts = (
                    _open_stream_chat_completions_with_connect_retry(
                        target=target,
                        payload={
                            "model": selected_model,
                            "messages": messages,
                            "max_tokens": max_tokens,
                            "temperature": temperature,
                            "stream": True,
                            "stream_options": {"include_usage": True},
                            **_build_generation_guard_sampling_overrides(),
                        },
                        read_timeout_seconds=600,
                    )
                )
                with response:
                    if not response.ok:
                        detail = f"Upstream error: HTTP {response.status_code}"
                        try:
                            body = response.text
                            if body:
                                detail = f"{detail} | {body[:500]}"
                        except Exception:
                            pass
                        yield _stream_json_line({"type": "error", "message": detail})
                        return

                    for raw_line in response.iter_lines(
                        chunk_size=1, decode_unicode=True
                    ):
                        if raw_line is None:
                            continue
                        line = raw_line.strip()
                        if not line or not line.startswith("data:"):
                            continue

                        data_line = line[5:].strip()
                        if not data_line:
                            continue
                        if data_line == "[DONE]":
                            break

                        try:
                            chunk = json.loads(data_line)
                        except json.JSONDecodeError:
                            continue

                        c_prompt, c_completion, c_total = _extract_token_usage(chunk)
                        if c_prompt is not None:
                            prompt_tokens = c_prompt
                        if c_completion is not None:
                            completion_tokens = c_completion
                        if c_total is not None:
                            total_tokens = c_total

                        choices = (
                            chunk.get("choices") if isinstance(chunk, dict) else None
                        )
                        if not isinstance(choices, list) or not choices:
                            continue
                        first = choices[0]
                        if not isinstance(first, dict):
                            continue
                        raw_finish_reason = first.get("finish_reason")
                        if isinstance(raw_finish_reason, str):
                            normalized_finish_reason = raw_finish_reason.strip().lower()
                            if normalized_finish_reason:
                                finish_reason = normalized_finish_reason
                        delta = first.get("delta")
                        if not isinstance(delta, dict):
                            continue

                        reasoning_delta = delta.get("reasoning")
                        if isinstance(reasoning_delta, str) and reasoning_delta:
                            explicit_reasoning_seen = True
                            section_mode = "answer"
                            emit = reasoning_delta
                            if not thinking_header_stripped:
                                cleaned = THINKING_HEADER_RE.sub("", emit, count=1)
                                if cleaned != emit:
                                    thinking_header_stripped = True
                                emit = cleaned
                            if emit:
                                yield _stream_json_line(
                                    {"type": "thinking_delta", "text": emit}
                                )

                        content_delta = delta.get("content")
                        if not isinstance(content_delta, str) or not content_delta:
                            continue

                        if deep_thinking and not explicit_reasoning_seen:
                            pending_content += content_delta

                            if section_mode == "auto":
                                probe = pending_content.lstrip()
                                has_think_open = bool(
                                    THINK_OPEN_TAG_RE.search(pending_content)
                                )
                                has_thinking_header = bool(
                                    THINKING_HEADER_RE.match(probe)
                                )
                                has_final_header = bool(
                                    FINAL_ANSWER_MARKER_RE.search(pending_content)
                                )

                                if has_think_open or has_thinking_header:
                                    section_mode = "thinking"
                                elif has_final_header or len(pending_content) >= 56:
                                    # Auto mode fallback: when no explicit reasoning markers
                                    # appear, treat the output as final answer stream.
                                    section_mode = "answer"

                            if section_mode == "thinking":
                                # Handle Qwen/Reasoner style think tags first.
                                pending_content = THINK_OPEN_TAG_RE.sub(
                                    "",
                                    pending_content,
                                )
                                close_tag = THINK_CLOSE_TAG_RE.search(pending_content)
                                if close_tag:
                                    thinking_part = pending_content[: close_tag.start()]
                                    answer_part = pending_content[close_tag.end() :]
                                    pending_content = ""

                                    if thinking_part:
                                        emit = thinking_part
                                        if not thinking_header_stripped:
                                            cleaned = THINKING_HEADER_RE.sub(
                                                "",
                                                emit,
                                                count=1,
                                            )
                                            if cleaned != emit:
                                                thinking_header_stripped = True
                                            emit = cleaned
                                        if emit:
                                            yield _stream_json_line(
                                                {"type": "thinking_delta", "text": emit}
                                            )

                                    section_mode = "answer"
                                    if answer_part:
                                        emit = answer_part
                                        if not answer_header_stripped:
                                            cleaned = LEADING_FINAL_ANSWER_RE.sub(
                                                "",
                                                emit,
                                                count=1,
                                            )
                                            if cleaned != emit:
                                                answer_header_stripped = True
                                            emit = cleaned
                                        if emit:
                                            yield _stream_json_line(
                                                {"type": "answer_delta", "text": emit}
                                            )
                                    continue

                                marker = FINAL_ANSWER_MARKER_RE.search(pending_content)
                                if marker:
                                    thinking_part = pending_content[: marker.start()]
                                    answer_part = pending_content[marker.end() :]
                                    pending_content = ""

                                    if thinking_part:
                                        emit = thinking_part
                                        if not thinking_header_stripped:
                                            cleaned = THINKING_HEADER_RE.sub(
                                                "", emit, count=1
                                            )
                                            if cleaned != emit:
                                                thinking_header_stripped = True
                                            emit = cleaned
                                        if emit:
                                            yield _stream_json_line(
                                                {"type": "thinking_delta", "text": emit}
                                            )

                                    section_mode = "answer"

                                    if answer_part:
                                        emit = answer_part
                                        if not answer_header_stripped:
                                            cleaned = LEADING_FINAL_ANSWER_RE.sub(
                                                "", emit, count=1
                                            )
                                            if cleaned != emit:
                                                answer_header_stripped = True
                                            emit = cleaned
                                        if emit:
                                            yield _stream_json_line(
                                                {"type": "answer_delta", "text": emit}
                                            )
                                else:
                                    # Keep a small tail so split markers across chunks are handled.
                                    tail_len = 72
                                    if len(pending_content) > tail_len:
                                        emit = pending_content[:-tail_len]
                                        pending_content = pending_content[-tail_len:]
                                        if not thinking_header_stripped:
                                            cleaned = THINKING_HEADER_RE.sub(
                                                "", emit, count=1
                                            )
                                            if cleaned != emit:
                                                thinking_header_stripped = True
                                            emit = cleaned
                                        if emit:
                                            yield _stream_json_line(
                                                {"type": "thinking_delta", "text": emit}
                                            )
                            else:
                                emit = pending_content
                                pending_content = ""
                                if not answer_header_stripped:
                                    cleaned = LEADING_FINAL_ANSWER_RE.sub(
                                        "", emit, count=1
                                    )
                                    if cleaned != emit:
                                        answer_header_stripped = True
                                    emit = cleaned
                                if emit:
                                    yield _stream_json_line(
                                        {"type": "answer_delta", "text": emit}
                                    )
                        else:
                            emit = content_delta
                            if deep_thinking and not answer_header_stripped:
                                cleaned = LEADING_FINAL_ANSWER_RE.sub("", emit, count=1)
                                if cleaned != emit:
                                    answer_header_stripped = True
                                emit = cleaned
                            if emit:
                                yield _stream_json_line(
                                    {"type": "answer_delta", "text": emit}
                                )

                    if pending_content:
                        if section_mode == "thinking":
                            emit = pending_content
                            if not thinking_header_stripped:
                                cleaned = THINKING_HEADER_RE.sub("", emit, count=1)
                                if cleaned != emit:
                                    thinking_header_stripped = True
                                emit = cleaned
                            if emit:
                                yield _stream_json_line(
                                    {"type": "thinking_delta", "text": emit}
                                )
                        else:
                            emit = pending_content
                            if not answer_header_stripped:
                                cleaned = LEADING_FINAL_ANSWER_RE.sub("", emit, count=1)
                                if cleaned != emit:
                                    answer_header_stripped = True
                                emit = cleaned
                            if emit:
                                yield _stream_json_line(
                                    {"type": "answer_delta", "text": emit}
                                )

            except _BackendConnectRetryError as exc:
                yield _stream_json_line(
                    {
                        "type": "error",
                        "message": (
                            "Cannot connect to token-workshed backend server. "
                            f"(retried {exc.attempts}x)"
                        ),
                    }
                )
                return
            except requests.exceptions.ConnectionError:
                yield _stream_json_line(
                    {
                        "type": "error",
                        "message": (
                            "Cannot connect to token-workshed backend server. "
                            "Make sure it is running and reachable."
                        ),
                    }
                )
                return
            except requests.exceptions.Timeout:
                yield _stream_json_line(
                    {
                        "type": "error",
                        "message": "Model response timed out.",
                    }
                )
                return
            except requests.RequestException as exc:
                yield _stream_json_line({"type": "error", "message": str(exc)})
                return
            except Exception as exc:
                yield _stream_json_line({"type": "error", "message": str(exc)})
                return

            latency_ms = (time.perf_counter() - request_start) * 1000.0
            tokens_per_second: float | None = None
            if completion_tokens is not None and latency_ms > 0:
                tokens_per_second = completion_tokens / (latency_ms / 1000.0)

            if total_tokens is None and (
                prompt_tokens is not None or completion_tokens is not None
            ):
                total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)

            yield _stream_json_line(
                {
                    "type": "done",
                    "ok": True,
                    "model": selected_model,
                    "request_model": selected_model,
                    "finish_reason": finish_reason or None,
                    "stop_reason": finish_reason or None,
                    "model_capability": model_capability,
                    "effective_params": {
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                    },
                    "metrics": {
                        "latency_ms": round(latency_ms, 2),
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens,
                        "connect_retry_attempts": int(connect_attempts),
                        "tokens_per_second": (
                            round(tokens_per_second, 2)
                            if isinstance(tokens_per_second, float)
                            else None
                        ),
                    },
                }
            )

        return StreamingResponse(stream_events(), media_type="application/x-ndjson")

    return app


def main() -> None:
    """Run the token-workshed UI API server."""
    parser = argparse.ArgumentParser(
        description="token-workshed UI API server",
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default="http://localhost:8000",
        help="vllm-mlx server URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host for this UI server (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7862,
        help="Port for this UI server (default: 7862)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Default max tokens in UI (default: 1024)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Default temperature in UI (default: 0.7)",
    )
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="Open browser automatically after startup",
    )
    args = parser.parse_args()

    app = create_app(
        default_server_url=args.server_url,
        default_max_tokens=args.max_tokens,
        default_temperature=args.temperature,
    )

    if args.open_browser:
        url = f"http://{args.host}:{args.port}"
        threading.Timer(0.9, lambda: webbrowser.open(url)).start()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
