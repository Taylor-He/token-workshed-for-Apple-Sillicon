"""Built-in local and web tools owned exclusively by the NextCore app chat."""

from __future__ import annotations

import fnmatch
import html
import ipaddress
import json
import os
import re
import signal
import socket
import subprocess
import threading
import time
from contextlib import suppress
from html.parser import HTMLParser
from itertools import chain
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote_plus, unquote, urljoin, urlparse

import requests

from .driver_manager import DriverUnavailableError, ensure_driver
from .environment import sanitized_environment
from .store import AgentStateStore
from .macos_permissions import (
    MacOSPermissionRequiredError,
    request_computer_use_permissions,
)

MAX_WEB_BYTES = 2 * 1024 * 1024
MAX_WEB_TEXT = 24_000
MAX_FILE_BYTES = 4 * 1024 * 1024
MAX_TOOL_TEXT = 32_000
_WEB_REQUEST_LOCK = threading.Lock()
_WEB_REQUESTS: dict[threading.Event, set[requests.Response]] = {}


def _check_web_cancelled(cancel_event) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise InterruptedError("Web request stopped by the user.")


def _web_request(
    url: str,
    *,
    headers: dict[str, str],
    timeout: tuple[int, int],
    cancel_event,
) -> requests.Response:
    _check_web_cancelled(cancel_event)
    response = requests.get(
        url,
        headers=headers,
        timeout=timeout,
        stream=True,
        allow_redirects=False,
    )
    if cancel_event is not None:
        with _WEB_REQUEST_LOCK:
            _WEB_REQUESTS.setdefault(cancel_event, set()).add(response)
    try:
        _check_web_cancelled(cancel_event)
    except InterruptedError:
        response.close()
        _release_web_response(cancel_event, response)
        raise
    return response


def _release_web_response(cancel_event, response: requests.Response) -> None:
    if cancel_event is None:
        return
    with _WEB_REQUEST_LOCK:
        active = _WEB_REQUESTS.get(cancel_event)
        if active is None:
            return
        active.discard(response)
        if not active:
            _WEB_REQUESTS.pop(cancel_event, None)


def interrupt_web_requests(cancel_event) -> None:
    """Close only HTTP response streams belonging to a cancelled Agent run."""
    with _WEB_REQUEST_LOCK:
        active = list(_WEB_REQUESTS.get(cancel_event, ()))
    for response in active:
        with suppress(Exception):
            response.close()


def _read_web_response(
    response: requests.Response, *, limit: int, cancel_event
) -> bytes:
    body = bytearray()
    for chunk in response.iter_content(32 * 1024):
        _check_web_cancelled(cancel_event)
        if not chunk:
            continue
        remaining = limit - len(body)
        body.extend(chunk[:remaining])
        if len(chunk) > remaining or len(body) >= limit:
            break
    _check_web_cancelled(cancel_event)
    return bytes(body)


TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the public web and return a short list of relevant result titles, URLs, and snippets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer", "minimum": 1, "maximum": 8},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": "Fetch and read a public HTTP or HTTPS webpage. Private network and local addresses are blocked.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "terminal",
            "description": "Run a shell command as the current macOS user. No sudo or privilege escalation. Long commands can be stopped from NextCore chat.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout_seconds": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 300,
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "file",
            "description": "Read, write, list, search, or delete a file in a path accessible to Token Workshed NextCore under macOS privacy permissions. File deletion removes a single file only.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["read", "write", "list", "search", "delete"],
                    },
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                    "pattern": {"type": "string"},
                },
                "required": ["action", "path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "computer_use",
            "description": "Inspect and operate macOS applications. For a read-only screen or window visibility check, call this with action='capture' exactly once, mode='ax' for text-only inspection or mode='vision' for visual inspection, and app set to the requested application when named. Capture does not click or type. NextCore manages and prestarts the CUA Driver when the desktop app opens. macOS Accessibility and Screen Recording permissions are checked before computer use.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": [
                            "capture",
                            "list_apps",
                            "launch_app",
                            "focus_app",
                            "click",
                            "double_click",
                            "right_click",
                            "middle_click",
                            "drag",
                            "scroll",
                            "type",
                            "key",
                            "set_value",
                            "wait",
                        ],
                    },
                    "app": {"type": "string"},
                    "mode": {"type": "string", "enum": ["ax", "som", "vision"]},
                    "element": {"type": "integer"},
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                    "from_element": {"type": "integer"},
                    "to_element": {"type": "integer"},
                    "from_x": {"type": "integer"},
                    "from_y": {"type": "integer"},
                    "to_x": {"type": "integer"},
                    "to_y": {"type": "integer"},
                    "value": {"type": "string"},
                    "text": {"type": "string"},
                    "keys": {"type": "string"},
                    "direction": {
                        "type": "string",
                        "enum": ["up", "down", "left", "right"],
                    },
                    "amount": {"type": "integer"},
                    "seconds": {"type": "number"},
                },
                "required": ["action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser",
            "description": "Inspect and operate Chromium pages using NextCore's CUA Driver. Use an isolated profile by default. An existing signed-in profile can be attached only when the user has explicitly enabled that exact PID/window in NextCore; then call prepare_existing with its profile_id, pid, and window_id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": [
                            "list_windows",
                            "prepare_isolated",
                            "prepare_existing",
                            "inspect",
                            "navigate",
                            "click",
                            "type",
                        ],
                    },
                    "pid": {"type": "integer"},
                    "window_id": {"type": "integer"},
                    "profile_id": {"type": "string"},
                    "target_id": {"type": "string"},
                    "tab_id": {"type": "string"},
                    "url": {"type": "string"},
                    "ref": {"type": "string"},
                    "text": {"type": "string"},
                    "app": {"type": "string"},
                },
                "required": ["action"],
            },
        },
    },
]

# Shared schemas are read-only after import, avoiding a deep copy per Agent run.
def _function_tool(name: str, description: str, parameters: dict[str, Any]) -> dict[str, Any]:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": parameters}}


MANAGED_TOOL_DEFINITIONS: tuple[dict[str, Any], ...] = (
    _function_tool("memory", "Search, recall, save, update, roll back, or delete persistent private memories in this NextCore installation. Never save credentials, secrets, or private keys.", {
        "type": "object", "properties": {
            "action": {"type": "string", "enum": ["search", "list", "save", "revisions", "rollback", "delete"]},
            "query": {"type": "string"}, "id": {"type": "string"}, "revision": {"type": "integer"},
            "title": {"type": "string"}, "body": {"type": "string"}, "tags": {"type": "array", "items": {"type": "string"}},
        }, "required": ["action"],
    }),
    _function_tool("skills", "Create or update a Markdown Skill, search and read existing Skills, list revisions, roll back, enable, or delete a Skill stored by NextCore.", {
        "type": "object", "properties": {
            "action": {"type": "string", "enum": ["search", "list", "save", "revisions", "rollback", "enable", "delete"]},
            "query": {"type": "string"}, "id": {"type": "string"}, "revision": {"type": "integer"},
            "name": {"type": "string"}, "description": {"type": "string"}, "body": {"type": "string"}, "enabled": {"type": "boolean"},
        }, "required": ["action"],
    }),
    _function_tool("session_search", "Search indexed messages from the user's past NextCore conversations.", {
        "type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer", "minimum": 1, "maximum": 20}}, "required": ["query"],
    }),
    _function_tool("delegate", "Delegate up to two independent subtasks to parallel child Agents. They share the current SGLang service and can only use tools in the parent's allowed scope.", {
        "type": "object", "properties": {"tasks": {"type": "array", "minItems": 1, "maxItems": 2, "items": {"type": "object", "properties": {"prompt": {"type": "string"}, "tools": {"type": "array", "items": {"type": "string"}}}, "required": ["prompt"]}}}, "required": ["tasks"],
    }),
    _function_tool("code_execution", "Run short Python analysis inside a macOS sandbox with temporary-workspace-only file access and no network. Refuses to run if OS sandboxing is unavailable. This does not use the terminal tool.", {
        "type": "object", "properties": {"code": {"type": "string"}, "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 30}}, "required": ["code"],
    }),
)
_STATIC_TOOL_DEFINITIONS = (*TOOL_DEFINITIONS, *MANAGED_TOOL_DEFINITIONS)
SUPPORTED_TOOL_NAMES = frozenset(
    item["function"]["name"] for item in _STATIC_TOOL_DEFINITIONS
)


def tool_definitions(
    mcp_tools: list[dict[str, Any]] | None = None,
    allowed: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Return the schemas granted to this run; returned schema objects are shared and read-only."""
    if allowed is None:
        selected = list(_STATIC_TOOL_DEFINITIONS)
        if mcp_tools:
            selected.extend(mcp_tools)
        return selected
    return [
        item
        for item in chain(_STATIC_TOOL_DEFINITIONS, mcp_tools or ())
        if str(item.get("function", {}).get("name") or "") in allowed
    ]


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in {"script", "style", "noscript", "svg", "head"}:
            self._skip += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "noscript", "svg", "head"} and self._skip:
            self._skip -= 1
        elif tag.lower() in {"p", "div", "li", "br", "h1", "h2", "h3", "tr"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip and data.strip():
            self.parts.append(data.strip())


class _SearchParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[dict[str, str]] = []
        self._current: dict[str, str] | None = None
        self._capture_field = ""
        self._capture_tag = ""

    def _finish_current(self) -> None:
        if self._current is not None:
            self.results.append(self._current)
            self._current = None
        self._capture_field = ""
        self._capture_tag = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        classes = set((values.get("class", "") or "").split())
        if tag == "a" and "result__a" in classes:
            self._finish_current()
            self._current = {
                "url": values.get("href", "") or "",
                "title": "",
                "snippet": "",
            }
            self._capture_field = "title"
            self._capture_tag = tag
        elif self._current is not None and "result__snippet" in classes:
            self._capture_field = "snippet"
            self._capture_tag = tag

    def handle_endtag(self, tag: str) -> None:
        if tag == self._capture_tag:
            self._capture_field = ""
            self._capture_tag = ""

    def handle_data(self, data: str) -> None:
        if self._current is not None and self._capture_field:
            self._current[self._capture_field] += data

    def close(self) -> None:
        super().close()
        self._finish_current()


def _public_http_url(url: str) -> str:
    parsed = urlparse(url.strip())
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
    ):
        raise ValueError(
            "Only public HTTP/HTTPS URLs without embedded credentials are allowed."
        )
    host = parsed.hostname.rstrip(".").lower()
    if (
        host in {"localhost", "localhost.localdomain"}
        or host.endswith(".localhost")
        or host.endswith(".local")
    ):
        raise ValueError("Local and private-network webpages are blocked.")
    try:
        addresses = {ipaddress.ip_address(host)}
    except ValueError:
        try:
            addresses = {
                ipaddress.ip_address(row[4][0])
                for row in socket.getaddrinfo(
                    host,
                    parsed.port or (443 if parsed.scheme == "https" else 80),
                    type=socket.SOCK_STREAM,
                )
            }
        except OSError as exc:
            raise ValueError(f"Could not resolve webpage host: {exc}") from exc
    if not addresses or any(not address.is_global for address in addresses):
        raise ValueError("Local and private-network webpages are blocked.")
    return parsed.geturl()


def _web_search(query: str, max_results: int = 5, *, cancel_event=None) -> str:
    query = query.strip()
    if not query or len(query) > 500:
        raise ValueError("Search query must contain 1 to 500 characters.")
    response = _web_request(
        "https://html.duckduckgo.com/html/?q=" + quote_plus(query),
        headers={"User-Agent": "Mozilla/5.0 (compatible; TokenWorkshedNextCore/1.0)"},
        timeout=(8, 18),
        cancel_event=cancel_event,
    )
    try:
        response.raise_for_status()
        raw = _read_web_response(
            response, limit=MAX_WEB_BYTES, cancel_event=cancel_event
        )
        parser = _SearchParser()
        parser.feed(raw.decode(response.encoding or "utf-8", errors="replace"))
        parser.close()
        rows = []
        for item in parser.results:
            _check_web_cancelled(cancel_event)
            href = item["url"]
            if href.startswith("//duckduckgo.com/l/"):
                href = "https:" + href
            if "duckduckgo.com/l/" in href:
                href = parse_qs(urlparse(href).query).get("uddg", [href])[0]
            try:
                href = _public_http_url(unquote(href))
            except ValueError:
                continue
            rows.append(
                {
                    "title": re.sub(r"\s+", " ", html.unescape(item["title"])).strip(),
                    "url": href,
                    "snippet": re.sub(r"\s+", " ", html.unescape(item.get("snippet", ""))).strip()[:500],
                }
            )
            if len(rows) >= max(1, min(int(max_results), 8)):
                break
        return json.dumps({"query": query, "results": rows}, ensure_ascii=False)
    finally:
        response.close()
        _release_web_response(cancel_event, response)


def _web_fetch(url: str, *, cancel_event=None) -> str:
    target = _public_http_url(url)
    response = _web_request(
        target,
        headers={"User-Agent": "Mozilla/5.0 (compatible; TokenWorkshedNextCore/1.0)"},
        timeout=(8, 20),
        cancel_event=cancel_event,
    )
    try:
        for redirect_count in range(7):
            _check_web_cancelled(cancel_event)
            if response.is_redirect or response.is_permanent_redirect:
                location = response.headers.get("Location")
                if not location:
                    break
                if redirect_count >= 6:
                    raise ValueError(
                        "The webpage exceeded the six-redirect safety limit."
                    )
                next_target = _public_http_url(urljoin(target, location))
                response.close()
                _release_web_response(cancel_event, response)
                target = next_target
                response = _web_request(
                    target,
                    headers={
                        "User-Agent": "Mozilla/5.0 (compatible; TokenWorkshedNextCore/1.0)"
                    },
                    timeout=(8, 20),
                    cancel_event=cancel_event,
                )
                continue
            break
        if response.is_redirect or response.is_permanent_redirect:
            raise ValueError("The webpage exceeded the six-redirect safety limit.")
        response.raise_for_status()
        raw = _read_web_response(
            response, limit=MAX_WEB_BYTES, cancel_event=cancel_event
        ).decode(response.encoding or "utf-8", errors="replace")
        parser = _TextExtractor()
        parser.feed(raw)
        text = re.sub(r"\n[ \t]+", "\n", "\n".join(parser.parts))
        text = re.sub(r"\n{3,}", "\n\n", html.unescape(text)).strip()
        return json.dumps(
            {
                "url": target,
                "status": response.status_code,
                "text": text[:MAX_WEB_TEXT],
            },
            ensure_ascii=False,
        )
    finally:
        response.close()
        _release_web_response(cancel_event, response)


def _safe_child_env() -> dict[str, str]:
    """Pass normal shell settings while keeping inherited credentials private."""
    safe = sanitized_environment()
    safe["HOME"] = str(Path.home())
    safe.setdefault(
        "PATH",
        "/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:/usr/local/sbin:"
        "/usr/bin:/bin:/usr/sbin:/sbin",
    )
    return safe


def _terminal_shell_path() -> str:
    """Prefer zsh on macOS and fall back to an available POSIX shell."""
    for candidate in ("/bin/zsh", "/bin/bash", "/bin/sh"):
        if Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return candidate
    return "/bin/sh"


def _terminal(args: dict[str, Any], cancel_event) -> str:
    command = str(args.get("command") or "")
    if not command.strip() or len(command) > 10_000:
        raise ValueError("Command must contain 1 to 10,000 characters.")
    if re.search(
        r"(?i)(?<![\w.-])(sudo|doas|su|pkexec)(?![\w.-])", command
    ) or re.search(r"(?is)with\s+administrator\s+privileges", command):
        raise ValueError("Privilege escalation is disabled in NextCore Agent commands.")
    timeout = max(1, min(int(args.get("timeout_seconds", 120)), 300))
    process = subprocess.Popen(
        # `-f` avoids loading user/global startup files that could reintroduce
        # credentials filtered from the inherited environment.
        [_terminal_shell_path(), "-f", "-c", command],
        cwd=str(Path.home()),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=_safe_child_env(),
        start_new_session=True,
    )
    deadline = time.monotonic() + timeout
    output_queue: list[tuple[str, str | None]] = []

    def collect_output() -> None:
        try:
            stdout, _ = process.communicate()
            output_queue.append((stdout or "", None))
        except Exception as exc:
            output_queue.append(("", str(exc)))

    reader = threading.Thread(
        target=collect_output, name="nextcore-agent-command-output", daemon=True
    )
    reader.start()
    try:
        while reader.is_alive():
            if cancel_event.is_set():
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except OSError:
                    process.terminate()
                reader.join(timeout=2)
                if reader.is_alive():
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except OSError:
                        process.kill()
                raise InterruptedError("Command stopped by the user.")
            if time.monotonic() >= deadline:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except OSError:
                    process.terminate()
                reader.join(timeout=2)
                if reader.is_alive():
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except OSError:
                        process.kill()
                raise TimeoutError(
                    f"Command exceeded {timeout} seconds and was stopped."
                )
            time.sleep(0.1)
        reader.join(timeout=1)
        if not output_queue:
            raise RuntimeError("Command output was not collected.")
        output, output_error = output_queue[0]
        if output_error:
            raise RuntimeError(f"Could not collect command output: {output_error}")
        clipped = output[:MAX_TOOL_TEXT]
        if len(output) > MAX_TOOL_TEXT:
            clipped += "\n[output clipped]"
        return json.dumps(
            {"exit_code": process.returncode, "output": clipped}, ensure_ascii=False
        )
    finally:
        if process.stdout:
            process.stdout.close()


def _resolve_file_path(value: str) -> Path:
    if not value.strip():
        raise ValueError("A file path is required.")
    return Path(value).expanduser().resolve(strict=False)


def _file(args: dict[str, Any], *, cancel_event=None) -> str:
    if cancel_event is not None and cancel_event.is_set():
        raise InterruptedError("File operation stopped by the user.")
    action = str(args.get("action") or "").lower()
    path = _resolve_file_path(str(args.get("path") or ""))
    if action == "read":
        if path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError(
                f"File exceeds the {MAX_FILE_BYTES // (1024 * 1024)} MiB read limit."
            )
        data = path.read_bytes()
        return data.decode("utf-8", errors="replace")[:MAX_TOOL_TEXT]
    if action == "write":
        content = str(args.get("content") or "")
        if len(content.encode("utf-8")) > MAX_FILE_BYTES:
            raise ValueError("File content exceeds the 4 MiB write limit.")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return json.dumps(
            {"written": True, "path": str(path), "bytes": len(content.encode())}
        )
    if action == "list":
        if not path.is_dir():
            raise ValueError("The requested path is not a directory.")
        items = []
        with os.scandir(path) as entries:
            for entry in entries:
                try:
                    info = entry.stat(follow_symlinks=False)
                    items.append(
                        {
                            "name": entry.name,
                            "directory": entry.is_dir(follow_symlinks=False),
                            "size": info.st_size,
                        }
                    )
                except OSError:
                    items.append(
                        {"name": entry.name, "directory": False, "error": "unavailable"}
                    )
                if len(items) >= 500:
                    break
        return json.dumps({"path": str(path), "entries": items}, ensure_ascii=False)
    if action == "search":
        pattern = str(args.get("pattern") or "*")
        if not path.is_dir():
            raise ValueError("Search path must be a directory.")
        matches: list[str] = []
        inaccessible: list[str] = []

        def record_walk_error(error: OSError) -> None:
            if cancel_event is not None and cancel_event.is_set():
                raise InterruptedError("File search stopped by the user.")
            if error.filename and len(inaccessible) < 30:
                inaccessible.append(str(error.filename))

        for root, dirs, files in os.walk(
            path, onerror=record_walk_error, followlinks=False
        ):
            if cancel_event is not None and cancel_event.is_set():
                raise InterruptedError("File search stopped by the user.")
            dirs[:] = [name for name in dirs if not name.startswith(".")]
            for name in files:
                if cancel_event is not None and cancel_event.is_set():
                    raise InterruptedError("File search stopped by the user.")
                if fnmatch.fnmatch(name, pattern):
                    matches.append(str(Path(root) / name))
                    if len(matches) >= 300:
                        break
            if len(matches) >= 300:
                break
        return json.dumps(
            {
                "matches": matches,
                "truncated": len(matches) >= 300,
                "inaccessible_paths": inaccessible,
            },
            ensure_ascii=False,
        )
    if action == "delete":
        path.unlink()
        return json.dumps({"deleted": True, "path": str(path)})
    raise ValueError("Unsupported file action.")


_CUA_BACKEND = None
_CUA_PERMISSION_ERROR: str | None = None
_CUA_PERMISSION_CHECKED = False
_CUA_PERMISSION_CAPTURE_PROBE_USED = False
_CUA_LOCK = __import__("threading").Lock()
_COMPUTER_ACTION_LOCK = __import__("threading").Lock()


def _action_result_json(result: Any) -> str:
    """Serialize an action summary without attaching screenshot payloads."""
    payload = {
        "ok": bool(getattr(result, "ok", False)),
        "action": str(getattr(result, "action", "")),
        "message": str(getattr(result, "message", ""))[:2000],
        "meta": getattr(result, "meta", {})
        if isinstance(getattr(result, "meta", {}), dict)
        else {},
    }
    if not payload["ok"] and re.search(
        r"permission|accessibility|screen.?record|privacy|denied|not authorized",
        payload["message"],
        re.I,
    ):
        payload["message"] = (
            "macOS blocked computer control. Enable Accessibility and Screen Recording "
            "for Token Workshed NextCore or the CUA Driver in System Settings > "
            "Privacy & Security, then retry. NextCore does not bypass macOS privacy controls."
        )
    return json.dumps(payload, ensure_ascii=False, default=str)


def _computer_use(args: dict[str, Any], app_data_dir: Path, cancel_event) -> str:
    global _CUA_PERMISSION_CAPTURE_PROBE_USED
    action = str(args.get("action") or "")
    if _CUA_PERMISSION_ERROR and action != "capture":
        raise MacOSPermissionRequiredError(_CUA_PERMISSION_ERROR)
    if _CUA_PERMISSION_ERROR and _CUA_PERMISSION_CAPTURE_PROBE_USED:
        raise MacOSPermissionRequiredError(
            f"{_CUA_PERMISSION_ERROR} NextCore already attempted its one read-only "
            "capture. Grant the permissions, quit and reopen NextCore, then retry."
        )

    backend = _get_computer_backend(
        app_data_dir,
        cancel_event=cancel_event,
        allow_missing_permissions=action == "capture",
    )
    if cancel_event.is_set():
        raise InterruptedError("Computer control stopped by the user.")
    if _CUA_PERMISSION_ERROR:
        if action != "capture":
            raise MacOSPermissionRequiredError(_CUA_PERMISSION_ERROR)
        if _CUA_PERMISSION_CAPTURE_PROBE_USED:
            raise MacOSPermissionRequiredError(
                f"{_CUA_PERMISSION_ERROR} NextCore already attempted its one "
                "read-only capture. Grant the permissions, quit and reopen "
                "NextCore, then retry."
            )
        _CUA_PERMISSION_CAPTURE_PROBE_USED = True

    if action == "capture":
        capture_mode = str(args.get("mode") or "ax").strip().lower()
        try:
            result = backend.capture(
                mode=capture_mode, app=args.get("app")
            )
        except Exception as exc:
            if _CUA_PERMISSION_ERROR:
                raise MacOSPermissionRequiredError(
                    f"{_CUA_PERMISSION_ERROR} NextCore tried one read-only capture "
                    "to register the missing permission with macOS. Grant it in "
                    "System Settings, quit and reopen NextCore, then retry."
                ) from exc
            raise
        if result.width <= 0 and not result.elements:
            if _CUA_PERMISSION_ERROR:
                raise MacOSPermissionRequiredError(
                    f"{_CUA_PERMISSION_ERROR} NextCore tried one read-only capture "
                    "to register the missing permission with macOS. Grant it in "
                    "System Settings, quit and reopen NextCore, then retry."
                )
            raise RuntimeError(
                "No accessible window state was returned. Make sure the app is open and enable Accessibility and Screen Recording for Token Workshed NextCore or the CUA Driver in System Settings > Privacy & Security."
            )
        if capture_mode == "vision" and not result.png_b64:
            if _CUA_PERMISSION_ERROR:
                raise MacOSPermissionRequiredError(
                    f"{_CUA_PERMISSION_ERROR} Screen Recording returned no image. "
                    "Grant it in System Settings, quit and reopen NextCore, then retry."
                )
            raise RuntimeError(
                "The screen capture returned no image pixels, so it cannot be shown in chat."
            )
        # Vision-mode pixels become a private UI attachment; AX capture remains
        # text-only. The Agent runtime removes image bytes before model context.
        image_mime_type = str(result.image_mime_type or "").strip().lower()
        if result.png_b64 and not image_mime_type:
            if result.png_b64.startswith("/9j/"):
                image_mime_type = "image/jpeg"
            elif result.png_b64.startswith("UklGR"):
                image_mime_type = "image/webp"
            else:
                image_mime_type = "image/png"
        include_screenshot = bool(result.png_b64) and (
            str(args.get("mode") or "ax").strip().lower() == "vision"
        )
        return json.dumps(
            {
                "app": result.app,
                "window_title": result.window_title,
                "elements": [
                    {
                        "index": item.index,
                        "role": item.role,
                        "label": item.label,
                        "bounds": item.bounds,
                    }
                    for item in result.elements[:300]
                ],
                "width": result.width,
                "height": result.height,
                **(
                    {
                        "screenshot_b64": result.png_b64,
                        "screenshot_mime_type": image_mime_type,
                    }
                    if include_screenshot
                    else {}
                ),
            },
            ensure_ascii=False,
        )
    if action == "list_apps":
        return json.dumps(backend.list_apps(), ensure_ascii=False)
    if action == "focus_app":
        return _action_result_json(
            backend.focus_app(str(args.get("app") or ""), raise_window=True)
        )
    if action == "launch_app":
        name = str(args.get("app") or "").strip()
        if not name:
            raise ValueError("An app name is required to launch an application.")
        return json.dumps(
            backend.launch_app(name=name), ensure_ascii=False, default=str
        )
    if action == "click":
        return _action_result_json(
            backend.click(element=args.get("element"), x=args.get("x"), y=args.get("y"))
        )
    if action == "double_click":
        return _action_result_json(
            backend.click(
                element=args.get("element"),
                x=args.get("x"),
                y=args.get("y"),
                click_count=2,
            )
        )
    if action == "right_click":
        return _action_result_json(
            backend.click(
                element=args.get("element"),
                x=args.get("x"),
                y=args.get("y"),
                button="right",
            )
        )
    if action == "middle_click":
        return _action_result_json(
            backend.click(
                element=args.get("element"),
                x=args.get("x"),
                y=args.get("y"),
                button="middle",
            )
        )
    if action == "drag":
        from_x, from_y = args.get("from_x"), args.get("from_y")
        to_x, to_y = args.get("to_x"), args.get("to_y")
        from_xy = (
            (int(from_x), int(from_y))
            if from_x is not None and from_y is not None
            else None
        )
        to_xy = (
            (int(to_x), int(to_y)) if to_x is not None and to_y is not None else None
        )
        return _action_result_json(
            backend.drag(
                from_element=args.get("from_element"),
                to_element=args.get("to_element"),
                from_xy=from_xy,
                to_xy=to_xy,
            )
        )
    if action == "scroll":
        return _action_result_json(
            backend.scroll(
                direction=str(args.get("direction") or "down"),
                amount=int(args.get("amount", 3)),
                element=args.get("element"),
                x=args.get("x"),
                y=args.get("y"),
            )
        )
    if action == "type":
        return _action_result_json(backend.type_text(str(args.get("text") or "")))
    if action == "key":
        return _action_result_json(backend.key(str(args.get("keys") or "")))
    if action == "set_value":
        return _action_result_json(
            backend.set_value(str(args.get("value") or ""), element=args.get("element"))
        )
    if action == "wait":
        return _action_result_json(backend.wait(float(args.get("seconds", 1.0))))
    raise ValueError("Unsupported computer action.")


def _get_computer_backend(
    app_data_dir: Path, *, cancel_event=None, allow_missing_permissions: bool = False
):
    global _CUA_BACKEND, _CUA_PERMISSION_ERROR, _CUA_PERMISSION_CHECKED
    while not _CUA_LOCK.acquire(timeout=0.1):
        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError("Computer control stopped by the user.")
    try:
        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError("Computer control stopped by the user.")
        if not _CUA_PERMISSION_CHECKED:
            try:
                request_computer_use_permissions()
            except MacOSPermissionRequiredError as exc:
                if not allow_missing_permissions:
                    raise
                # A read-only capture may make one attempt before access is
                # granted. Keep normal browser/control requests permission-gated.
                _CUA_PERMISSION_ERROR = str(exc)
                _CUA_PERMISSION_CHECKED = True
            else:
                _CUA_PERMISSION_ERROR = None
                _CUA_PERMISSION_CHECKED = True
        elif _CUA_PERMISSION_ERROR and not allow_missing_permissions:
            raise MacOSPermissionRequiredError(_CUA_PERMISSION_ERROR)

        if _CUA_BACKEND is None:
            binary = ensure_driver(app_data_dir, cancel_event=cancel_event)
            try:
                from .computer_use.cua_backend import CuaDriverBackend
            except ImportError as exc:
                raise DriverUnavailableError(
                    "Computer control dependencies are missing."
                ) from exc
            backend = CuaDriverBackend(
                driver_command=str(binary),
                allow_existing_profile=AgentStateStore(app_data_dir).has_browser_profile_grants(),
            )
            try:
                backend.start()
            except Exception as exc:
                with suppress(Exception):
                    backend.stop()
                raise RuntimeError(
                    f"Could not start the computer-control driver ({type(exc).__name__})."
                ) from exc
            _CUA_BACKEND = backend
        return _CUA_BACKEND
    finally:
        _CUA_LOCK.release()


def warm_computer_backend(app_data_dir: Path, *, cancel_event=None) -> None:
    """Install and connect the managed driver without requesting OS permissions.

    Startup warmup only launches the driver and establishes its MCP session; it
    does not capture the screen or operate other applications. Permissions are
    still checked when a user actually requests browser or computer use.
    """
    global _CUA_BACKEND
    while not _CUA_LOCK.acquire(timeout=0.1):
        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError("Agent warmup stopped during shutdown.")
    try:
        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError("Agent warmup stopped during shutdown.")
        if _CUA_BACKEND is not None:
            return
        binary = ensure_driver(app_data_dir, cancel_event=cancel_event)
        try:
            from .computer_use.cua_backend import CuaDriverBackend
        except ImportError as exc:
            raise DriverUnavailableError(
                "Computer-control dependencies are missing."
            ) from exc
        backend = CuaDriverBackend(
            driver_command=str(binary),
            allow_existing_profile=AgentStateStore(app_data_dir).has_browser_profile_grants(),
        )
        try:
            backend.start()
        except Exception as exc:
            with suppress(Exception):
                backend.stop()
            raise RuntimeError(
                f"Could not warm the computer-control driver ({type(exc).__name__})."
            ) from exc
        if cancel_event is not None and cancel_event.is_set():
            with suppress(Exception):
                backend.stop()
            raise InterruptedError("Agent warmup stopped during shutdown.")
        _CUA_BACKEND = backend
    finally:
        _CUA_LOCK.release()


_BROWSER_OMIT_KEYS = {
    "cookies",
    "local_storage",
    "localstorage",
    "session_storage",
    "sessionstorage",
    "authorization",
    "headers",
    "screenshot",
    "screenshot_png_b64",
    "png_b64",
    "image",
    "images",
    "network_log",
    "request_headers",
    "response_headers",
}


def _safe_browser_payload(value: Any, *, depth: int = 0) -> Any:
    if depth > 8:
        return "[nested content omitted]"
    if isinstance(value, dict):
        result = {}
        for key, item in list(value.items())[:120]:
            key_text = str(key)
            normalized = key_text.casefold().replace("-", "_")
            if normalized in _BROWSER_OMIT_KEYS or normalized.endswith("_png_b64"):
                continue
            result[key_text] = _safe_browser_payload(item, depth=depth + 1)
        return result
    if isinstance(value, (list, tuple)):
        return [_safe_browser_payload(item, depth=depth + 1) for item in value[:100]]
    if isinstance(value, str):
        return value[:6000]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)[:1000]


def _browser_result_json(result: dict[str, Any]) -> str:
    if result.get("isError"):
        message = result.get("data") or "The browser driver rejected the operation."
        raise RuntimeError(str(message)[:1200])
    payload = result.get("structuredContent")
    if not isinstance(payload, (dict, list, str)):
        payload = result.get("data")
    if isinstance(payload, str):
        with suppress(ValueError):
            payload = json.loads(payload)
    if (
        isinstance(payload, dict)
        and str(payload.get("status") or "").lower() == "refused"
    ):
        refusal = payload.get("refusal") or {}
        if isinstance(refusal, dict):
            message = (
                refusal.get("message")
                or refusal.get("code")
                or "The browser driver refused this operation."
            )
        else:
            message = refusal or "The browser driver refused this operation."
        raise RuntimeError(str(message)[:1200])
    safe = _safe_browser_payload(payload)
    encoded = json.dumps(safe, ensure_ascii=False, default=str)
    if len(encoded) > MAX_WEB_TEXT:
        encoded = json.dumps(
            {"truncated": True, "result": encoded[: MAX_WEB_TEXT - 64]},
            ensure_ascii=False,
        )
    return encoded


def _browser(args: dict[str, Any], app_data_dir: Path, cancel_event, authorization_scope: dict[str, Any] | None = None) -> str:
    action = str(args.get("action") or "")
    backend = _get_computer_backend(app_data_dir, cancel_event=cancel_event)
    domains = [str(item).strip().casefold().lstrip(".") for item in (authorization_scope or {}).get("allowed_domains", []) if isinstance(item, str) and item.strip()]

    def url_is_allowed(value: str) -> bool:
        parsed = urlparse(value)
        host = (parsed.hostname or "").casefold().rstrip(".")
        return bool(parsed.scheme in {"http", "https"} and host and domains and any(host == domain or host.endswith("." + domain) for domain in domains))

    def validate_snapshot(encoded: str) -> None:
        try:
            payload = json.loads(encoded)
        except ValueError as exc:
            raise PermissionError("Could not verify the scheduled browser page's origin.") from exc
        urls: list[str] = []
        def collect(value: Any, key: str = "") -> None:
            if isinstance(value, dict):
                for child_key, child in value.items():
                    normalized = str(child_key).casefold().replace("-", "_")
                    if normalized in {"url", "page_url", "current_url", "active_url"} and isinstance(child, str):
                        urls.append(child)
                    else:
                        collect(child, normalized)
            elif isinstance(value, list):
                for child in value:
                    collect(child, key)
        collect(payload)
        if not urls or not all(url_is_allowed(url) for url in urls):
            raise PermissionError("The scheduled browser page is outside the task's authorized domains or its origin could not be confirmed.")

    def check_scope_snapshot(target: str, tab: str) -> str:
        snapshot = backend.call_tool(
            "get_browser_state",
            {"target_id": target, "tab_id": tab, "session": str(getattr(backend, "session_id", ""))},
            timeout=45,
        )
        encoded = _browser_result_json(snapshot)
        validate_snapshot(encoded)
        return encoded

    if cancel_event.is_set():
        raise InterruptedError("Browser operation stopped by the user.")
    if action == "list_windows":
        result = backend.call_tool("list_windows", {"on_screen_only": True}, timeout=30)
        if result.get("isError"):
            return _browser_result_json(result)
        raw = (result.get("structuredContent") or {}).get("windows") or []
        browsers = (
            "chrome",
            "chromium",
            "microsoft edge",
            "brave",
            "arc",
            "opera",
            "vivaldi",
        )
        windows = []
        for window in raw:
            if not isinstance(window, dict):
                continue
            name = str(window.get("app_name") or "")
            bundle = str(window.get("bundle_id") or "")
            if not any(
                marker in (name + " " + bundle).casefold() for marker in browsers
            ):
                continue
            pid, window_id = window.get("pid"), window.get("window_id")
            if pid is None or window_id is None:
                continue
            windows.append(
                {
                    "app_name": name,
                    "pid": int(pid),
                    "window_id": int(window_id),
                    "title": str(window.get("title") or "")[:500],
                }
            )
        return json.dumps({"windows": windows[:50]}, ensure_ascii=False)
    if action == "prepare_isolated":
        pid = int(args.get("pid") or 0)
        if pid <= 0:
            raise ValueError(
                "Select a Chromium browser process from list_windows first."
            )
        result = backend.call_tool(
            "browser_prepare",
            {"pid": pid, "allow_launch": True, "profile": {"mode": "isolated_new"}, "session": str(getattr(backend, "session_id", ""))},
            timeout=60,
        )
        return _browser_result_json(result)
    if action == "prepare_existing":
        profile_id = str(args.get("profile_id") or "").strip()
        profile = AgentStateStore(app_data_dir).get_browser_profile(profile_id)
        pid = int(args.get("pid") or 0)
        window_id = int(args.get("window_id") or 0)
        if profile is None or int(profile["pid"]) != pid or int(profile["window_id"]) != window_id:
            raise PermissionError("This exact Chromium window is not enabled for signed-in profile access in NextCore.")
        if not bool(getattr(backend, "allow_existing_profile", False)):
            raise RuntimeError("Restart the NextCore browser session after enabling this profile, then retry.")
        result = backend.call_tool(
            "browser_prepare",
            {
                "pid": pid,
                "window_id": window_id,
                "session": str(getattr(backend, "session_id", "")),
                "strategy": {"kind": "existing_profile"},
            },
            timeout=75,
        )
        return _browser_result_json(result)
    if action == "inspect":
        target_id = str(args.get("target_id") or "").strip()
        tab_id = str(args.get("tab_id") or "").strip()
        if target_id and tab_id:
            result = backend.call_tool(
                "get_browser_state",
                {"target_id": target_id, "tab_id": tab_id, "session": str(getattr(backend, "session_id", ""))},
                timeout=45,
            )
        else:
            pid = int(args.get("pid") or 0)
            window_id = int(args.get("window_id") or 0)
            if pid <= 0 or window_id <= 0:
                raise ValueError(
                    "Browser inspection requires a selected pid and window_id, or target_id and tab_id."
                )
            result = backend.call_tool(
                "get_browser_state", {"pid": pid, "window_id": window_id, "session": str(getattr(backend, "session_id", ""))}, timeout=45
            )
        encoded = _browser_result_json(result)
        if domains:
            validate_snapshot(encoded)
        return encoded
    target_id = str(args.get("target_id") or "").strip()
    tab_id = str(args.get("tab_id") or "").strip()
    if not target_id or not tab_id:
        raise ValueError(
            "Inspect and bind the exact browser window before acting on a page."
        )
    if action == "navigate":
        url = str(args.get("url") or "").strip()
        parsed = urlparse(url)
        if (
            parsed.scheme not in {"https", "http"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
        ):
            raise ValueError(
                "Browser navigation requires a public HTTP/HTTPS URL without embedded credentials."
            )
        if domains and not url_is_allowed(url):
            raise PermissionError("This scheduled task may navigate only to its authorized browser domains.")
        if domains:
            check_scope_snapshot(target_id, tab_id)
        result = backend.call_tool(
            "browser_navigate",
            {"target_id": target_id, "tab_id": tab_id, "url": url, "session": str(getattr(backend, "session_id", ""))},
            timeout=45,
        )
        encoded = _browser_result_json(result)
        if domains:
            return check_scope_snapshot(target_id, tab_id)
        return encoded
    if action == "click":
        ref = str(args.get("ref") or "").strip()
        if not ref:
            raise ValueError(
                "Take a fresh browser snapshot and provide one of its current element refs."
            )
        if domains:
            check_scope_snapshot(target_id, tab_id)
        result = backend.call_tool(
            "browser_click",
            {"target_id": target_id, "tab_id": tab_id, "ref": ref, "session": str(getattr(backend, "session_id", ""))},
            timeout=30,
        )
        encoded = _browser_result_json(result)
        if domains:
            return check_scope_snapshot(target_id, tab_id)
        return encoded
    if action == "type":
        ref = str(args.get("ref") or "").strip()
        if not ref:
            raise ValueError(
                "Take a fresh browser snapshot and provide one of its current element refs."
            )
        text = str(args.get("text") or "")
        if len(text) > 10_000:
            raise ValueError("Browser text entry is limited to 10,000 characters.")
        if domains:
            check_scope_snapshot(target_id, tab_id)
        result = backend.call_tool(
            "browser_type",
            {"target_id": target_id, "tab_id": tab_id, "ref": ref, "text": text, "session": str(getattr(backend, "session_id", ""))},
            timeout=30,
        )
        encoded = _browser_result_json(result)
        if domains:
            return check_scope_snapshot(target_id, tab_id)
        return encoded
    raise ValueError("Unsupported browser action.")


def execute_tool(
    name: str, arguments: dict[str, Any], *, app_data_dir: Path, cancel_event,
    authorization_scope: dict[str, Any] | None = None,
) -> str:
    if cancel_event.is_set():
        raise InterruptedError("Agent run stopped by the user.")
    if name == "web_search":
        return _web_search(
            str(arguments.get("query") or ""),
            int(arguments.get("max_results", 5)),
            cancel_event=cancel_event,
        )
    if name == "web_fetch":
        return _web_fetch(str(arguments.get("url") or ""), cancel_event=cancel_event)
    if name == "terminal":
        return _terminal(arguments, cancel_event)
    if name == "file":
        try:
            return _file(arguments, cancel_event=cancel_event)
        except PermissionError as exc:
            raise RuntimeError(
                "macOS denied file access. For protected locations, add Token Workshed NextCore to System Settings > Privacy & Security > Full Disk Access, then retry. NextCore does not bypass macOS privacy controls."
            ) from exc
    if name == "computer_use":
        try:
            return _run_computer_action(
                lambda: _computer_use(arguments, app_data_dir, cancel_event),
                cancel_event,
            )
        except PermissionError as exc:
            raise RuntimeError(
                "macOS blocked computer control. Enable Accessibility and Screen Recording "
                "for Token Workshed NextCore or the CUA Driver in System Settings > "
                "Privacy & Security, then retry. NextCore does not bypass macOS privacy controls."
            ) from exc
    if name == "browser":
        try:
            return _run_computer_action(
                lambda: _browser(arguments, app_data_dir, cancel_event, authorization_scope),
                cancel_event,
            )
        except PermissionError as exc:
            raise RuntimeError(
                "macOS blocked browser or computer control. Enable the requested permissions in System Settings > Privacy & Security. NextCore does not bypass macOS privacy controls."
            ) from exc
    raise ValueError(f"Unknown NextCore Agent tool: {name}")


def _run_computer_action(action, cancel_event) -> str:
    while not _COMPUTER_ACTION_LOCK.acquire(timeout=0.1):
        if cancel_event.is_set():
            raise InterruptedError("Computer control stopped by the user.")
    try:
        if cancel_event.is_set():
            raise InterruptedError("Computer control stopped by the user.")
        return action()
    finally:
        _COMPUTER_ACTION_LOCK.release()


def close_computer_backend() -> None:
    global _CUA_BACKEND, _CUA_PERMISSION_ERROR, _CUA_PERMISSION_CHECKED
    global _CUA_PERMISSION_CAPTURE_PROBE_USED
    with _CUA_LOCK:
        backend, _CUA_BACKEND = _CUA_BACKEND, None
        _CUA_PERMISSION_ERROR = None
        _CUA_PERMISSION_CHECKED = False
        _CUA_PERMISSION_CAPTURE_PROBE_USED = False
        if backend is not None:
            try:
                backend.stop()
            except Exception:
                return


def interrupt_computer_use() -> None:
    """Close the active driver session to stop an in-flight desktop action."""
    close_computer_backend()
