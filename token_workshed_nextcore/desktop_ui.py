# SPDX-License-Identifier: Apache-2.0
"""One-click desktop launcher for SGLang + native Rust UI.

This launcher starts:
1) SGLang OpenAI-compatible server
2) local FastAPI UI/backend API server
3) the Rust Iced/libcosmic desktop UI

It also exposes manager endpoints so the UI can switch models directly.
"""

from __future__ import annotations

import argparse
import inspect
import ipaddress
import json
import os
import re
import secrets
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from collections import deque
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal, NoReturn

import requests
import uvicorn
from fastapi import HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict, Field

from . import css_svg_ui
from .css_svg_ui import create_app
from .desktop_state import (
    clear_legacy_agent_settings,
    desktop_state_file,
    mutate_desktop_state,
    read_desktop_state,
)
from .ide_bridge import IDEBridge, register_ide_bridge_routes
from .jetbrains_integration import register_jetbrains_integration_routes
from .model_cache import (
    invalidate_local_snapshot_cache,
    local_huggingface_snapshot,
)
from .sglang_runtime import server_command
from .server_control import RemoteTarget, ServerConnectionController
from .workshed import WorkshedService

# Keep the mapped developer terminal rooted at this checkout without importing
# the desktop launcher's paths into the backend module.
TOKEN_WORKSHED_ROOT = Path(__file__).resolve().parent.parent

try:
    import fcntl
except Exception:
    fcntl = None  # type: ignore[assignment]


class _DeveloperLogBuffer:
    """Timestamped diagnostic log tail for desktop-manager diagnostics."""

    def __init__(self, max_entries: int = 300) -> None:
        self._max_entries = max_entries
        self._lock = threading.RLock()
        self._entries: list[str] = []

    def add(self, message: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {message.strip()}"
        with self._lock:
            self._entries.append(line)
            if len(self._entries) > self._max_entries:
                self._entries = self._entries[-self._max_entries :]

    def tail(self, limit: int = 80) -> list[str]:
        limit = max(1, min(200, int(limit or 80)))
        with self._lock:
            return list(self._entries[-limit:])


_DEVELOPER_LOGS = _DeveloperLogBuffer()


def _developer_log(message: str) -> None:
    _DEVELOPER_LOGS.add(message)


class _DeveloperTerminalBuffer:
    """Raw command-line output kept separate from application diagnostics."""

    def __init__(self, max_entries: int = 400) -> None:
        self._max_entries = max_entries
        self._lock = threading.RLock()
        self._entries: list[str] = []

    def add(self, message: str) -> None:
        # Keep terminal output faithful: do not prepend timestamps or strip
        # indentation from compiler/server output.
        line = str(message).rstrip("\r\n")
        with self._lock:
            self._entries.append(line)
            if len(self._entries) > self._max_entries:
                self._entries = self._entries[-self._max_entries :]

    def tail(self, limit: int = 120) -> list[str]:
        limit = max(1, min(300, int(limit or 120)))
        with self._lock:
            return list(self._entries[-limit:])


_DEVELOPER_TERMINAL_OUTPUT = _DeveloperTerminalBuffer()


def _terminal_output(message: str) -> None:
    _DEVELOPER_TERMINAL_OUTPUT.add(message)


class _DeveloperTerminalSession:
    """Keep one local shell alive so NextCore command-line work is stateful.

    The native UI polls the raw terminal buffer. Keeping the shell in a
    separate reader thread means stdout/stderr appears incrementally instead
    of waiting for a blocking subprocess call to finish, while the shell still
    retains working-directory and environment changes between commands.
    """

    def __init__(self, cwd: Path) -> None:
        self.cwd = Path(cwd).resolve()
        self._lock = threading.RLock()
        self._process: subprocess.Popen[str] | None = None
        self._reader: threading.Thread | None = None
        self._stop_requested = threading.Event()

    def _shell_path(self) -> str:
        configured = str(os.environ.get("SHELL", "") or "").strip()
        if configured and Path(configured).is_file() and os.access(configured, os.X_OK):
            return configured
        for candidate in ("/bin/zsh", "/bin/bash", "/bin/sh"):
            if Path(candidate).is_file() and os.access(candidate, os.X_OK):
                return candidate
        return "/bin/sh"

    def _start_locked(self) -> subprocess.Popen[str]:
        process = self._process
        if process is not None and process.poll() is None:
            return process

        self._stop_requested.clear()
        shell = self._shell_path()
        environment = os.environ.copy()
        # A non-interactive shell avoids prompts/control sequences in the
        # mapped terminal while still preserving shell state for commands.
        environment.update({"TERM": "dumb", "PS1": "", "PROMPT_COMMAND": ""})
        try:
            process = subprocess.Popen(
                [shell, "-s"],
                cwd=str(self.cwd),
                env=environment,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
        except OSError as exc:
            raise RuntimeError(f"Could not start terminal shell: {exc}") from exc

        self._process = process
        self._reader = threading.Thread(
            target=self._read_output,
            args=(process,),
            daemon=True,
            name="token-workshed-terminal-reader",
        )
        self._reader.start()
        return process

    def _read_output(self, process: subprocess.Popen[str]) -> None:
        stream = process.stdout
        if stream is not None:
            try:
                for line in stream:
                    if self._stop_requested.is_set():
                        break
                    text = line.rstrip("\r\n")
                    _terminal_output(text)
            except (OSError, ValueError) as exc:
                if not self._stop_requested.is_set():
                    _terminal_output(f"nextcore: output reader stopped: {exc}")
        return_code = process.poll()
        if return_code is None:
            try:
                return_code = process.wait(timeout=0.2)
            except subprocess.TimeoutExpired:
                return_code = None
        if not self._stop_requested.is_set() and return_code is not None:
            _terminal_output(f"nextcore: shell exited with code {return_code}")

    def send(self, command: str) -> str:
        value = str(command or "").strip()
        if not value:
            raise ValueError("Terminal command cannot be empty.")
        if len(value) > 8_000:
            raise ValueError("Terminal command is limited to 8000 characters.")

        with self._lock:
            process = self._start_locked()
            stdin = process.stdin
            if stdin is None or process.poll() is not None:
                self._process = None
                process = self._start_locked()
                stdin = process.stdin
            if stdin is None:
                raise RuntimeError("Terminal shell stdin is unavailable.")
            display_value = value.replace("\r", "\\r").replace("\n", "\\n")
            _terminal_output(f"nextcore $ {display_value}")
            try:
                stdin.write(value + "\n")
                stdin.flush()
            except (BrokenPipeError, OSError, ValueError) as exc:
                self._process = None
                _terminal_output(f"nextcore: command failed: {exc}")
                try:
                    _terminate_process(process, timeout_s=1.0)
                except Exception:
                    pass
                raise RuntimeError(
                    f"Could not send command to terminal: {exc}"
                ) from exc
        return "Command sent to the NextCore terminal."

    def stop(self) -> None:
        with self._lock:
            self._stop_requested.set()
            process = self._process
            self._process = None
            if process is None:
                return
            try:
                if process.stdin is not None:
                    process.stdin.close()
            except (OSError, ValueError):
                pass
            try:
                _terminate_process(process)
            except Exception:
                try:
                    process.kill()
                except Exception:
                    pass


def _wait_http_ok(url: str, timeout_s: float, interval_s: float = 0.6) -> bool:
    """Poll URL until it returns HTTP 200 or timeout."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            response = requests.get(url, timeout=3)
            if response.ok:
                return True
        except requests.RequestException:
            pass
        time.sleep(interval_s)
    return False


def _native_ui_binary_candidates() -> list[Path]:
    """Return possible native UI binary locations for source and app bundles."""
    binary_name = "token-workshed-native-ui"
    candidates: list[Path] = []

    explicit = os.environ.get("TOKEN_WORKSHED_NATIVE_UI_BIN", "").strip()
    if explicit:
        candidates.append(Path(explicit).expanduser())

    module_root = Path(__file__).resolve().parent.parent
    candidates.extend(
        [
            module_root / "native-ui" / "target" / "release" / binary_name,
            module_root / "native-ui" / "target" / "debug" / binary_name,
        ]
    )

    executable_dir = Path(sys.executable).resolve().parent
    candidates.extend(
        [
            executable_dir / binary_name,
            executable_dir.parent / "Resources" / binary_name,
        ]
    )

    pyinstaller_tmp = getattr(sys, "_MEIPASS", "")
    if pyinstaller_tmp:
        candidates.append(Path(str(pyinstaller_tmp)) / binary_name)

    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            deduped.append(candidate)
    return deduped


def _find_native_ui_binary() -> Path | None:
    for candidate in _native_ui_binary_candidates():
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None


def _format_native_ui_candidates() -> str:
    return "\n".join(f"  - {candidate}" for candidate in _native_ui_binary_candidates())


def _query_served_model_ids(server_url: str) -> list[str]:
    """Return model IDs exposed by OpenAI-compatible `/v1/models`."""
    try:
        response = requests.get(f"{server_url}/v1/models", timeout=3)
        if not response.ok:
            return []
        payload = response.json()
    except (requests.RequestException, ValueError):
        return []

    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return []

    model_ids: list[str] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        model_id = str(item.get("id", "") or "").strip()
        if model_id:
            model_ids.append(model_id)
    return model_ids


def _is_hidden_model(model_id: str) -> bool:
    """Hide internal models and MLX-only community checkpoints from NextCore."""
    normalized = str(model_id or "").strip().lower()
    if not normalized:
        return False
    if normalized.startswith("mlx-community/"):
        return True

    candidates = {
        str(item).strip().lower()
        for item in getattr(css_svg_ui, "TITLE_MODEL_CANDIDATES", ())
        if isinstance(item, str) and item.strip()
    }
    aliases = {
        "functiongemma-270m-int4",
        "functiongemma-270m-it-4bit",
        "mlx-community/functiongemma-270m-int4",
        "mlx-community/functiongemma-270m-it-4bit",
    }
    return normalized in candidates or normalized in aliases


def _canonicalize_model_id(model_id: str) -> str:
    """Normalize known typo-prone model IDs to canonical Hugging Face IDs."""
    value = str(model_id or "").strip()
    if not value:
        return ""

    lowered = value.lower()
    typo_prefixes = ("imstudio-community/", "1mstudio-community/")
    for prefix in typo_prefixes:
        if lowered.startswith(prefix):
            suffix = value.split("/", 1)[1].strip() if "/" in value else ""
            if suffix:
                return f"lmstudio-community/{suffix}"
            return "lmstudio-community"
    return value


def _terminate_process(proc: subprocess.Popen[bytes], timeout_s: float = 12) -> None:
    """Gracefully terminate subprocess and force-kill if needed."""
    if proc.poll() is not None:
        return

    used_group_signal = False
    try:
        # Only terminate process group when child was started in a new session.
        # This avoids killing the current shell process group for legacy children.
        pgid = os.getpgid(proc.pid)
        if pgid == proc.pid:
            os.killpg(proc.pid, signal.SIGTERM)
            used_group_signal = True
    except Exception:
        used_group_signal = False

    if not used_group_signal:
        proc.terminate()
    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        try:
            if used_group_signal:
                os.killpg(proc.pid, signal.SIGKILL)
            else:
                proc.kill()
        except Exception:
            proc.kill()
        proc.wait(timeout=5)


def _fatal(message: str, code: int = 1) -> NoReturn:
    print(f"Error: {message}")
    raise SystemExit(code)


def _is_loopback_host(host: str) -> bool:
    """Return True when host resolves to localhost/loopback."""
    value = host.strip().lower()
    if value in {"localhost"}:
        return True
    try:
        return ipaddress.ip_address(value).is_loopback
    except ValueError:
        return False


_HF_TOKEN_KEYCHAIN_SERVICE = "token-workshed-nextcore.huggingface-token"
_HF_TOKEN_KEYCHAIN_ACCOUNT = "default"


def _load_hf_token_from_keychain() -> str:
    """Read HF token from macOS Keychain (best effort)."""
    if sys.platform != "darwin":
        return ""
    try:
        result = subprocess.run(
            [
                "security",
                "find-generic-password",
                "-a",
                _HF_TOKEN_KEYCHAIN_ACCOUNT,
                "-s",
                _HF_TOKEN_KEYCHAIN_SERVICE,
                "-w",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _save_hf_token_to_keychain(token: str) -> bool:
    """Store HF token in macOS Keychain (best effort)."""
    if sys.platform != "darwin":
        return False
    clean_token = token.strip()
    if not clean_token:
        return True
    try:
        result = subprocess.run(
            [
                "security",
                "add-generic-password",
                "-U",
                "-a",
                _HF_TOKEN_KEYCHAIN_ACCOUNT,
                "-s",
                _HF_TOKEN_KEYCHAIN_SERVICE,
                "-w",
                clean_token,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return False
    return result.returncode == 0


def _delete_hf_token_from_keychain() -> None:
    """Delete HF token from macOS Keychain (best effort)."""
    if sys.platform != "darwin":
        return
    try:
        subprocess.run(
            [
                "security",
                "delete-generic-password",
                "-a",
                _HF_TOKEN_KEYCHAIN_ACCOUNT,
                "-s",
                _HF_TOKEN_KEYCHAIN_SERVICE,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        pass


def _enforce_manager_auth(request: Request, expected_token: str | None) -> None:
    """Reject manager API request when token is required and missing/invalid."""
    if not expected_token:
        return
    provided = str(
        request.headers.get("x-token-workshed-manager-token", "") or ""
    ).strip()
    if not provided or not secrets.compare_digest(provided, expected_token):
        raise HTTPException(status_code=401, detail="Unauthorized manager request.")


def _performance_snapshot() -> dict[str, Any]:
    """Return measured host-wide CPU and unified-memory utilization."""
    try:
        import psutil

        cpu_percent = float(psutil.cpu_percent(interval=0.15))
        memory = psutil.virtual_memory()
        total_bytes = int(memory.total)
        used_bytes = int(memory.used)
        return {
            "ok": True,
            "cpu_utilization_percent": round(min(100.0, max(0.0, cpu_percent)), 1),
            "logical_cpu_count": int(psutil.cpu_count(logical=True) or 0),
            "unified_memory_used_gb": round(used_bytes / (1024.0**3), 2),
            "unified_memory_total_gb": round(total_bytes / (1024.0**3), 2),
            "unified_memory_percent": round(
                min(100.0, max(0.0, float(memory.percent))), 1
            ),
        }
    except Exception as exc:
        return {"ok": False, "error": f"Resource metrics unavailable: {exc}"}


def _desktop_state_file() -> Path:
    """Return state file path for desktop launcher user preferences."""
    return desktop_state_file()


def _read_desktop_state() -> dict[str, Any]:
    """Read desktop state from disk."""
    try:
        return read_desktop_state()
    except Exception:
        # Preference loading is best-effort.
        return {}


def _mutate_desktop_state(
    mutator: Callable[[dict[str, Any]], bool],
) -> bool:
    """Apply a shared desktop-state transaction (best effort)."""
    try:
        return mutate_desktop_state(mutator)
    except Exception:
        # Preference save is best-effort.
        return False


def _is_pid_running(pid: int) -> bool:
    """Return True when a process ID is currently alive."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def _read_background_state() -> dict[str, Any]:
    """Read persisted background launcher state."""
    payload = _read_desktop_state()
    bg = payload.get("background", {})
    if not isinstance(bg, dict):
        return {}
    pid_raw = bg.get("pid", 0)
    try:
        pid = int(pid_raw)
    except (TypeError, ValueError):
        pid = 0
    return {
        "pid": pid,
        "server_url": str(bg.get("server_url", "") or "").strip(),
        "ui_url": str(bg.get("ui_url", "") or "").strip(),
        "model": str(bg.get("model", "") or "").strip(),
        "started_at": int(bg.get("started_at", 0) or 0),
    }


def _save_background_state(
    *,
    pid: int,
    server_url: str,
    ui_url: str,
    model: str,
) -> None:
    """Persist background process metadata."""

    def store(payload: dict[str, Any]) -> bool:
        payload["background"] = {
            "pid": int(pid),
            "server_url": server_url.strip(),
            "ui_url": ui_url.strip(),
            "model": model.strip(),
            "started_at": int(time.time()),
        }
        return True

    _mutate_desktop_state(store)


def _clear_background_state(expected_pid: int | None = None) -> None:
    """Clear persisted background metadata.

    When ``expected_pid`` is provided, state is cleared only if PID matches.
    """

    def clear(payload: dict[str, Any]) -> bool:
        bg = payload.get("background")
        if not isinstance(bg, dict):
            return False
        if expected_pid is not None:
            try:
                current_pid = int(bg.get("pid", 0) or 0)
            except (TypeError, ValueError):
                current_pid = 0
            if current_pid != int(expected_pid):
                return False
        payload.pop("background", None)
        return True

    _mutate_desktop_state(clear)


def _load_server_mode_preference(default_enabled: bool = True) -> bool:
    """Load persisted server mode preference for window-close behavior."""
    payload = _read_desktop_state()
    raw = payload.get("server_mode", {})
    if isinstance(raw, dict):
        value = raw.get("keep_running_on_window_close", default_enabled)
    else:
        value = default_enabled

    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        return normalized not in {"0", "false", "off", "no"}
    return bool(default_enabled)


def _save_server_mode_preference(enabled: bool) -> None:
    """Persist server mode preference for future launches."""

    def store(payload: dict[str, Any]) -> bool:
        payload["server_mode"] = {
            "keep_running_on_window_close": bool(enabled),
            "updated_at": int(time.time()),
        }
        return True

    _mutate_desktop_state(store)


def _stop_pid(pid: int, timeout_s: float = 12.0) -> tuple[bool, str]:
    """Stop a process ID gracefully, then force kill if needed."""
    if pid <= 0:
        return False, "Invalid PID."
    if not _is_pid_running(pid):
        return True, "Background process is not running."

    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return True, "Background process already exited."
    except Exception as exc:
        return False, f"Failed to send SIGTERM: {exc}"

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if not _is_pid_running(pid):
            return True, "Background process stopped."
        time.sleep(0.2)

    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return True, "Background process already exited."
    except Exception as exc:
        return False, f"Failed to send SIGKILL: {exc}"

    for _ in range(10):
        if not _is_pid_running(pid):
            return True, "Background process stopped (forced)."
        time.sleep(0.1)
    return False, "Background process did not stop."


def _list_listening_pids(port: int) -> list[int]:
    """Best-effort lookup for PIDs listening on a local TCP port."""
    if port <= 0:
        return []

    pids: set[int] = set()
    if os.name == "nt":
        try:
            output = subprocess.check_output(
                ["netstat", "-ano", "-p", "tcp"],
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="ignore",
            )
        except Exception:
            return []

        suffix = f":{int(port)}"
        for line in output.splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            local_addr = parts[1]
            state = parts[3].upper()
            pid_raw = parts[4]
            if state != "LISTENING" or not local_addr.endswith(suffix):
                continue
            try:
                pid = int(pid_raw)
            except (TypeError, ValueError):
                continue
            if pid > 0:
                pids.add(pid)
        return sorted(pids)

    try:
        output = subprocess.check_output(
            ["lsof", "-nP", "-t", f"-iTCP:{int(port)}", "-sTCP:LISTEN"],
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
    except Exception:
        return []

    for line in output.splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            pid = int(text)
        except (TypeError, ValueError):
            continue
        if pid > 0:
            pids.add(pid)
    return sorted(pids)


def _read_pid_command(pid: int) -> str:
    """Read full command line for a PID (best effort)."""
    if pid <= 0:
        return ""
    try:
        output = subprocess.check_output(
            ["ps", "-p", str(pid), "-o", "command="],
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
    except Exception:
        return ""
    return output.strip()


def _read_pid_parent(pid: int) -> int:
    """Read parent PID for a process (best effort)."""
    if pid <= 0:
        return 0
    try:
        output = subprocess.check_output(
            ["ps", "-p", str(pid), "-o", "ppid="],
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
    except Exception:
        return 0
    try:
        return int(str(output).strip() or "0")
    except (TypeError, ValueError):
        return 0


def _looks_like_nextcore_serve_process(command: str) -> bool:
    """Recognize only NextCore-owned serving processes."""
    normalized = str(command or "").strip().lower()
    if not normalized:
        return False
    return "token_workshed_nextcore.sglang_runtime" in normalized or (
        "token_workshed_nextcore.cli" in normalized and "serve" in normalized
    )


def _redact_model_server_log(value: str) -> str:
    """Keep useful model startup diagnostics while removing common credentials."""

    text = str(value or "")
    text = re.sub(r"(?i)bearer\s+[A-Za-z0-9._~+/-]+", "Bearer [redacted]", text)
    text = re.sub(r"(?i)\bhf_[A-Za-z0-9]{16,}\b", "[redacted HF token]", text)
    text = re.sub(
        r"(?i)(token|secret|password|api[_-]?key)\s*[:=]\s*[^\s,;]+",
        r"\1=[redacted]",
        text,
    )
    return text[-1200:]


def _looks_like_desktop_manager_process(command: str) -> bool:
    """Return True for token-workshed desktop manager processes."""
    normalized = str(command or "").strip().lower()
    if not normalized or "--token-workshed-serve" in normalized:
        return False
    if "token_workshed_nextcore.desktop_ui" in normalized:
        return True
    return (
        "token-workshed-nextcore" in normalized
        and "--server-port" in normalized
        and "--ui-port" in normalized
    )


def _command_flag_int(command: str, flag: str) -> int | None:
    """Extract integer value from `--flag value` or `--flag=value` command forms."""
    normalized_flag = str(flag or "").strip()
    if not normalized_flag:
        return None
    try:
        argv = shlex.split(str(command or ""))
    except Exception:
        argv = str(command or "").split()
    if not argv:
        return None
    for idx, token in enumerate(argv):
        if token == normalized_flag and idx + 1 < len(argv):
            try:
                return int(str(argv[idx + 1]).strip())
            except (TypeError, ValueError):
                return None
        if token.startswith(f"{normalized_flag}="):
            raw = token.split("=", 1)[1].strip()
            try:
                return int(raw)
            except (TypeError, ValueError):
                return None
    return None


def _cleanup_stale_desktop_ui_port_owners(
    ui_port: int,
    *,
    keep_pid: int = 0,
) -> tuple[list[int], list[tuple[int, str]]]:
    """Stop stale desktop manager processes that still occupy UI port."""
    reclaimed: list[int] = []
    blocked: list[tuple[int, str]] = []
    for pid in _list_listening_pids(ui_port):
        if pid <= 0 or (keep_pid > 0 and pid == keep_pid):
            continue
        command = _read_pid_command(pid)
        if not _looks_like_desktop_manager_process(command):
            blocked.append((pid, command))
            continue
        ok, _ = _stop_pid(pid, timeout_s=6.0)
        if ok:
            reclaimed.append(pid)
        else:
            blocked.append((pid, command))
    return reclaimed, blocked


@contextmanager
def _acquire_backend_start_lock(timeout_s: float = 60.0):
    """Serialize backend start attempts across multiple desktop manager processes."""
    if fcntl is None:
        yield
        return

    lock_path = _desktop_state_file().parent / "backend_start.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+", encoding="utf-8")
    deadline = time.time() + max(0.5, float(timeout_s))

    try:
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.time() >= deadline:
                    raise TimeoutError(
                        "Timed out waiting for backend start lock. "
                        "Another token-workshed process may still be starting."
                    )
                time.sleep(0.15)

        try:
            handle.seek(0)
            handle.truncate()
            handle.write(f"{os.getpid()} {int(time.time())}\n")
            handle.flush()
        except Exception:
            pass

        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        try:
            handle.close()
        except Exception:
            pass


def _can_connect_tcp(host: str, port: int, timeout_s: float = 0.35) -> bool:
    """Return True when a TCP endpoint accepts a connection."""
    if not host.strip() or port <= 0:
        return False
    try:
        with socket.create_connection((host, int(port)), timeout=timeout_s):
            return True
    except OSError:
        return False


def _build_background_launch_cmd(args: argparse.Namespace) -> list[str]:
    """Build detached subprocess command for background mode."""
    if getattr(sys, "frozen", False):
        cmd = [sys.executable]
    else:
        cmd = [sys.executable, "-m", "token_workshed_nextcore.desktop_ui"]

    model = str(args.model or "").strip()
    if model:
        cmd.append(model)

    cmd.extend(
        [
            "--server-host",
            str(args.server_host),
            "--server-port",
            str(args.server_port),
            "--ui-host",
            str(args.ui_host),
            "--ui-port",
            str(args.ui_port),
            "--max-tokens",
            str(args.max_tokens),
            "--temperature",
            str(args.temperature),
            "--serve-only",
        ]
    )
    if args.allow_remote_ui:
        cmd.append("--allow-remote-ui")
    return cmd


def _load_last_model_preference() -> str:
    """Load last successfully started model from local state."""
    payload = _read_desktop_state()
    if not isinstance(payload, dict):
        return ""
    value = str(payload.get("last_model", "") or "").strip()
    return value


def _save_last_model_preference(model: str) -> None:
    """Persist last successfully started model for next app launch."""
    value = model.strip()
    if not value:
        return

    def store(payload: dict[str, Any]) -> bool:
        payload["last_model"] = value
        return True

    _mutate_desktop_state(store)


def _load_hf_binding_state() -> dict[str, Any]:
    """Load persisted Hugging Face binding metadata."""
    payload = _read_desktop_state()
    hf = payload.get("huggingface", {})
    if not isinstance(hf, dict):
        hf = {}

    username = str(hf.get("username", "") or "").strip()
    legacy_token = str(hf.get("token", "") or "").strip()
    token = ""
    token_storage = str(hf.get("token_storage", "") or "").strip().lower()
    if sys.platform == "darwin":
        token = _load_hf_token_from_keychain()
        if not token and legacy_token:
            # One-time migration from legacy plaintext storage.
            if _save_hf_token_to_keychain(legacy_token):

                def clear_legacy_token(current: dict[str, Any]) -> bool:
                    current_hf = current.get("huggingface")
                    if not isinstance(current_hf, dict):
                        return False
                    if str(current_hf.get("token", "") or "").strip() != legacy_token:
                        return False
                    updated_hf = dict(current_hf)
                    updated_hf["token"] = ""
                    updated_hf["token_storage"] = "keychain"
                    current["huggingface"] = updated_hf
                    return True

                _mutate_desktop_state(clear_legacy_token)
                hf["token"] = ""
                hf["token_storage"] = "keychain"
                token_storage = "keychain"
            token = legacy_token
    else:
        token = legacy_token

    seen = bool(hf.get("seen", False))
    if username or token:
        seen = True

    return {
        "seen": seen,
        "username": username,
        "token": token,
        "token_storage": token_storage
        or ("keychain" if sys.platform == "darwin" else "file"),
    }


def _save_hf_binding_state(
    username: str, token: str, seen: bool = True
) -> dict[str, Any]:
    """Persist Hugging Face binding metadata and return sanitized state."""
    clean_username = username.strip()
    clean_token = token.strip()
    persisted_token = clean_token
    token_storage = "file"

    if sys.platform == "darwin":
        if clean_token:
            if _save_hf_token_to_keychain(clean_token):
                persisted_token = ""
                token_storage = "keychain"
            else:
                token_storage = "file"
        else:
            _delete_hf_token_from_keychain()
            persisted_token = ""
            token_storage = "none"

    def store(payload: dict[str, Any]) -> bool:
        payload["huggingface"] = {
            "seen": bool(seen),
            "username": clean_username,
            "token": persisted_token,
            "token_storage": token_storage,
            "has_token": bool(clean_token),
            "updated_at": int(time.time()),
        }
        return True

    _mutate_desktop_state(store)
    return {
        "seen": bool(seen),
        "username": clean_username,
        "token": clean_token,
        "token_storage": token_storage,
    }


def _detect_reasoning_parser(
    model_id: str,
    model_path: str | None = None,
) -> str | None:
    """Infer the SGLang reasoning parser from the ID or checkpoint config."""
    normalized = model_id.strip().lower()
    if not normalized and not model_path:
        return None
    if "qwen3" in normalized or "qwq" in normalized:
        return "qwen3"
    if "deepseek" in normalized and "r1" in normalized:
        return "deepseek_r1"
    if "gpt-oss" in normalized:
        return "gpt_oss"
    if "harmony" in normalized:
        return "harmony"

    candidate = Path(model_path or model_id).expanduser()
    if not candidate.is_dir():
        return None

    config: dict[str, Any] = {}
    try:
        config_path = candidate / "config.json"
        if config_path.stat().st_size <= 2 * 1024 * 1024:
            parsed = json.loads(config_path.read_text(encoding="utf-8"))
            if isinstance(parsed, dict):
                config = parsed
    except (OSError, TypeError, ValueError):
        pass

    model_types = [config.get("model_type"), config.get("_name_or_path")]
    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        model_types.append(text_config.get("model_type"))
    architectures = config.get("architectures")
    if isinstance(architectures, str):
        model_types.append(architectures)
    elif isinstance(architectures, (list, tuple)):
        model_types.extend(architectures)
    model_signature = " ".join(
        value.lower()
        for value in model_types
        if isinstance(value, str)
    )
    if "qwen3" in model_signature or "qwq" in model_signature:
        return "qwen3"
    if "deepseek" in model_signature and "r1" in model_signature:
        return "deepseek_r1"
    if "deepseekv3" in model_signature or "deepseek_v3" in model_signature:
        return "deepseek_v3"
    if "gpt_oss" in model_signature or "gptoss" in model_signature:
        return "gpt_oss"

    try:
        tokenizer_path = candidate / "tokenizer_config.json"
        if tokenizer_path.stat().st_size <= 2 * 1024 * 1024:
            tokenizer_config = json.loads(tokenizer_path.read_text(encoding="utf-8"))
            chat_template = (
                tokenizer_config.get("chat_template", "")
                if isinstance(tokenizer_config, dict)
                else ""
            )
            if (
                isinstance(chat_template, str)
                and "enable_thinking" in chat_template
                and "</think>" in chat_template
            ):
                return "qwen3"
    except (OSError, TypeError, ValueError):
        pass
    return None


def _detect_tool_call_parser(model_id: str) -> str | None:
    """Infer best tool-call parser for OpenAI-compatible local serving."""
    normalized = model_id.strip().lower()
    if not normalized:
        return None

    if "granite" in normalized:
        # Granite 4.0's bundled chat template emits the Hermes
        # <tool_call>{...}</tool_call> format. SGLang 0.5.22 has no Granite
        # parser, while its Hermes parser consumes that native template output.
        return "hermes"
    if "qwen3" in normalized and "coder" in normalized:
        return "qwen3_coder"
    if "qwen" in normalized:
        return "qwen"
    if "gemma" in normalized:
        return "gemma4" if "gemma 4" in normalized or "gemma-4" in normalized else "auto"
    if "mistral" in normalized:
        return "mistral"
    if "deepseek" in normalized:
        return "deepseekv3"
    if "kimi" in normalized:
        return "kimi_k2"
    if "xlam" in normalized:
        return "auto"
    if "functionary" in normalized:
        return "auto"
    if "glm-4.7" in normalized or "glm47" in normalized:
        return "glm47"
    if "llama" in normalized or "nemotron" in normalized:
        return "llama3"
    return "auto"


def _is_env_enabled(name: str, default_enabled: bool = True) -> bool:
    """Parse common boolean env values."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return bool(default_enabled)
    return raw.lower() not in {"0", "false", "off", "no", "n"}


def _model_capability(model_id: str) -> dict[str, Any]:
    """Infer UI capability metadata for a model."""
    normalized = model_id.strip().lower()
    reasoning_parser = _detect_reasoning_parser(model_id)
    deep_thinking = bool(reasoning_parser) or any(
        token in normalized for token in ("nemotron", "reasoner", "thinking")
    )
    return {
        "deep_thinking": deep_thinking,
        "reasoning_parser": reasoning_parser,
        "image": False,
        "video": False,
        "asr": False,
        "tts": False,
        "multimodal_reason": "Image, video, ASR, and TTS inputs are not supported by this NextCore build.",
    }


def _model_capability_map(models: list[str]) -> dict[str, dict[str, Any]]:
    """Build capability map keyed by model id."""
    return {model_id: _model_capability(model_id) for model_id in models}


def _model_source_label(model_id: str, managed_model_path: str | None = None) -> str:
    """Return a compact user-facing source label for model details."""
    model_id = _canonicalize_model_id(model_id)
    if not model_id:
        return "Unknown"
    if managed_model_path:
        return (
            "Workshed managed" if model_id.startswith("workshed/") else "Managed model"
        )
    if Path(model_id).expanduser().is_dir():
        return "Local cache"
    if "/" in model_id:
        return "Hugging Face"
    return "Local cache"


def _model_config_quantization_label(model_path: str | None) -> str | None:
    """Read weight precision from an existing local model config, best effort."""
    if not model_path:
        return None
    try:
        config_path = Path(model_path).expanduser() / "config.json"
        if config_path.stat().st_size > 2 * 1024 * 1024:
            return None
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return None
    if not isinstance(config, dict):
        return None
    for key in ("quantization", "quantization_config"):
        quantization = config.get(key)
        if not isinstance(quantization, dict):
            continue
        bits = quantization.get("bits")
        if isinstance(bits, str) and len(bits.strip()) <= 2 and bits.strip().isdecimal():
            bits = int(bits.strip())
        if isinstance(bits, int) and not isinstance(bits, bool) and 0 < bits <= 32:
            return f"{bits}-bit"
    return None


def _model_quantization_label(
    model_id: str,
    runtime_config: dict[str, Any] | None = None,
    model_path: str | None = None,
) -> str:
    """Return model quantization plus relevant runtime cache quantization state."""
    label = _model_config_quantization_label(model_path) or _infer_quantization_label(
        model_id
    )
    if not label or label == "-":
        label = "Unknown"
    elif label.lower() == "4bit":
        label = "4-bit"
    elif label.lower() == "8bit":
        label = "8-bit"
    if bool((runtime_config or {}).get("kv_cache_quantization")):
        if label == "Unknown":
            return "KV cache quantization"
        return f"{label} + KV cache quantization"
    return label


def _build_model_details(
    models: list[str],
    *,
    active_model: str,
    running_models: list[dict[str, Any]],
    runtime_config: dict[str, Any] | None,
    managed_model_resolver: Callable[[str], str | None] | None = None,
) -> list[dict[str, Any]]:
    """Build stable, UI-oriented model metadata while preserving legacy fields."""
    active = _canonicalize_model_id(active_model)
    running_by_model: dict[str, dict[str, Any]] = {}
    for item in running_models:
        model_id = _canonicalize_model_id(str(item.get("model", "") or ""))
        if model_id:
            running_by_model[model_id] = item

    ordered: list[str] = []
    seen: set[str] = set()
    for model in [active_model, *models, *running_by_model.keys()]:
        model_id = _canonicalize_model_id(str(model or ""))
        if not model_id or _is_hidden_model(model_id) or model_id in seen:
            continue
        seen.add(model_id)
        ordered.append(model_id)

    details: list[dict[str, Any]] = []
    for model_id in ordered:
        managed_model_path = None
        if managed_model_resolver is not None:
            try:
                managed_model_path = managed_model_resolver(model_id)
            except Exception:
                # Registry metadata must not hide an otherwise available model.
                pass
        model_path = managed_model_path
        if not model_path and Path(model_id).expanduser().is_dir():
            model_path = model_id
        running = running_by_model.get(model_id, {})
        is_active = model_id == active or bool(running.get("active"))
        is_running = bool(running)
        status = "active" if is_active else "running" if is_running else "local"
        capabilities = _model_capability(model_id)
        reasoning_parser = (
            str(running.get("reasoning_parser") or "").strip()
            or capabilities.get("reasoning_parser")
            or _detect_reasoning_parser(model_id)
            or ""
        )
        tool_call_parser = (
            str(running.get("tool_call_parser") or "").strip()
            or _detect_tool_call_parser(model_id)
            or ""
        )
        kind = str(running.get("kind") or "").strip()
        role = str(running.get("role") or "").strip() or kind
        details.append(
            {
                "id": model_id,
                "display_name": model_id,
                "status": status,
                "source": _model_source_label(model_id, managed_model_path),
                "backend": "sglang",
                "quantization": _model_quantization_label(
                    model_id, runtime_config, model_path
                ),
                "active": is_active,
                "running": is_running,
                "server_url": str(running.get("server_url") or "").strip(),
                "role": role,
                "kind": kind,
                "reasoning_parser": reasoning_parser,
                "tool_call_parser": tool_call_parser,
            }
        )
    return details


OFFICIAL_DEVELOPER_MAP = {
    "openai": "OpenAI",
    "google": "Google",
    "deepseek-ai": "Deepseek",
    "deepseek": "Deepseek",
    "mistralai": "Mistral",
    "mistral-ai": "Mistral",
    "mlx-community": "mlx-community",
    "meta-llama": "Meta",
    "qwen": "Qwen",
    "ibm-granite": "IBM",
}
OFFICIAL_DEVELOPER_ALLOWLIST = set(OFFICIAL_DEVELOPER_MAP.keys())
SIZE_TOKEN_RE = re.compile(r"(^|[-_])(\d+(?:\.\d+)?)([bkm])(?=$|[-_])", re.IGNORECASE)
SIZE_WORD_TOKEN_RE = re.compile(
    r"(^|[-_])(nano|tiny|mini|micro|small|medium|med|large|xl|xxl)(?=$|[-_])",
    re.IGNORECASE,
)
QUANT_TOKEN_RE = re.compile(
    r"(?:(?:^|[-_])((?:q\d(?:_[a-z0-9]+)*)|(?:int\d+)|(?:fp\d+)|(?:bf16)|(?:f16)|(?:\d+bit)|(?:gptq)|(?:awq)|(?:gguf))(?:$|[-_]))",
    re.IGNORECASE,
)
QUANT_FAMILY_STRIP_RE = re.compile(
    r"(^|[-_])(?:(?:q\d(?:_[a-z0-9]+)*)|(?:int\d+)|(?:fp\d+)|(?:bf16)|(?:f16)|(?:\d+bit)|(?:gptq)|(?:awq)|(?:gguf))(?=$|[-_])",
    re.IGNORECASE,
)
def _normalize_developer_name(repo_id: str) -> tuple[str, str]:
    owner = repo_id.split("/", 1)[0].strip().lower()
    canonical = OFFICIAL_DEVELOPER_MAP.get(owner, owner)
    return owner, canonical


def _is_official_developer(repo_id: str) -> bool:
    owner, _ = _normalize_developer_name(repo_id)
    return owner in OFFICIAL_DEVELOPER_ALLOWLIST


def _infer_model_family_and_size(repo_id: str) -> tuple[str, str]:
    """Extract model family key and size label from HF repo id."""
    name = repo_id.split("/", 1)[-1].strip()
    if not name:
        return repo_id.strip().lower(), "-"

    matches = list(SIZE_TOKEN_RE.finditer(name))
    word_matches = list(SIZE_WORD_TOKEN_RE.finditer(name))

    picked = None
    size_label = "-"
    if matches:
        picked = matches[-1]
        number = picked.group(2)
        suffix = picked.group(3).upper()
        size_label = f"{number}{suffix}"
    elif word_matches:
        picked = word_matches[-1]
        size_label = picked.group(2).lower()

    if picked is None:
        family = re.sub(r"[-_]{2,}", "-", name).strip("-_")
        return family.lower(), size_label

    left = name[: picked.start()]
    right = name[picked.end() :]
    family = f"{left}{right}"
    family = QUANT_FAMILY_STRIP_RE.sub(r"\1", family)
    family = re.sub(r"[-_]{2,}", "-", family).strip("-_")
    if not family:
        family = name
    return family.lower(), size_label


def _infer_quantization_label(repo_id: str, tags: list[str] | None = None) -> str:
    """Infer quantization/format label from repo id + HF tags."""
    candidates: list[str] = [repo_id.split("/", 1)[-1].strip().lower()]
    for tag in tags or []:
        if isinstance(tag, str):
            candidates.append(tag.strip().lower())

    seen: set[str] = set()
    for text in candidates:
        if not text:
            continue
        for match in QUANT_TOKEN_RE.finditer(text):
            token = (match.group(1) or "").strip().lower()
            if not token or token in seen:
                continue
            seen.add(token)
            if token == "f16":
                return "FP16"
            if token == "bf16":
                return "BF16"
            if token.startswith("fp"):
                return token.upper()
            if token.startswith("int"):
                return token.upper()
            if token.endswith("bit"):
                return token
            if token.startswith("q"):
                return token.upper()
            if token in {"gptq", "awq", "gguf"}:
                return token.upper()
            return token
    return "-"


def _query_match_score(query: str, *texts: str) -> int:
    normalized = query.strip().lower()
    if not normalized:
        return 0
    tokens = [token for token in re.split(r"\s+", normalized) if token]
    best = 0
    for text in texts:
        candidate = str(text or "").strip().lower()
        if not candidate:
            continue
        score = 0
        if candidate == normalized:
            score += 180
        if normalized in candidate:
            score += 90
        if tokens:
            score += sum(16 for token in tokens if token in candidate)
        best = max(best, score)
    return best


def _build_local_search_item(model_id: str, query: str) -> dict[str, Any] | None:
    canonical = _canonicalize_model_id(model_id.strip())
    if not canonical or _is_hidden_model(canonical):
        return None

    family_key, size_label = _infer_model_family_and_size(canonical)
    quant_label = _infer_quantization_label(canonical)
    capability = _model_capability(canonical)
    query_score = _query_match_score(
        query, canonical, family_key, size_label, quant_label, "local"
    )
    if query.strip() and query_score <= 0:
        return None

    return {
        "id": canonical,
        "developer": "Local",
        "developer_slug": "local",
        "family_key": f"local/{family_key}",
        "family_name": family_key,
        "size_label": size_label,
        "param_size_label": (
            size_label if re.match(r"^\d+(?:\.\d+)?[BKM]$", size_label) else "-"
        ),
        "variant_size_label": size_label,
        "quantization_label": quant_label,
        "downloads": 0,
        "likes": 0,
        "pipeline_tag": "text-generation",
        "last_modified": None,
        "private": False,
        "gated": False,
        "local": True,
        "deep_thinking": bool(capability.get("deep_thinking")),
        "reasoning_parser": capability.get("reasoning_parser"),
        "source": "local",
        "source_priority": 3,
        "query_score": query_score,
    }


def _merge_and_rank_search_results(
    local_results: list[dict[str, Any]],
    community_results: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    ordered: list[dict[str, Any]] = []

    for item in [*local_results, *community_results]:
        model_id = _canonicalize_model_id(str(item.get("id", "")).strip())
        if not model_id:
            continue

        existing = merged.get(model_id)
        if existing is None:
            payload = dict(item)
            payload["id"] = model_id
            merged[model_id] = payload
            ordered.append(payload)
            continue

        existing["downloads"] = max(
            int(existing.get("downloads", 0) or 0),
            int(item.get("downloads", 0) or 0),
        )
        existing["likes"] = max(
            int(existing.get("likes", 0) or 0),
            int(item.get("likes", 0) or 0),
        )
        existing["local"] = bool(existing.get("local")) or bool(item.get("local"))
        existing["gated"] = bool(existing.get("gated")) or bool(item.get("gated"))
        existing["deep_thinking"] = bool(existing.get("deep_thinking")) or bool(
            item.get("deep_thinking")
        )
        existing["source_priority"] = max(
            int(existing.get("source_priority", 0) or 0),
            int(item.get("source_priority", 0) or 0),
        )
        existing["query_score"] = max(
            int(existing.get("query_score", 0) or 0),
            int(item.get("query_score", 0) or 0),
        )
        existing_source = str(existing.get("source", "") or "").strip().lower()
        item_source = str(item.get("source", "") or "").strip().lower()
        if existing_source != item_source and item_source:
            existing["source"] = "hybrid"

    def search_sort_key(item: dict[str, Any]) -> tuple[Any, ...]:
        query_score = int(item.get("query_score", 0) or 0)
        local = int(bool(item.get("local")))
        downloads = int(item.get("downloads", 0) or 0)
        likes = int(item.get("likes", 0) or 0)
        model_id = str(item.get("id", "") or "").lower()

        if query_score > 0:
            return (-query_score, -local, -downloads, -likes, model_id)
        return (-downloads, -likes, -local, model_id)

    ordered.sort(key=search_sort_key)
    return ordered[: max(1, int(limit))]


class SwitchModelPayload(BaseModel):
    """Request body for switching serve model."""

    model: str = Field(min_length=1)


class RuntimeConfigPayload(BaseModel):
    """Request body for runtime scheduler/performance options."""

    continuous_batching: bool | None = None
    use_paged_cache: bool | None = None
    kv_cache_quantization: bool | None = None
    chunked_prefill_tokens: int | None = None
    enable_mtp: bool | None = None
    mtp_num_draft_tokens: int | None = None


class DeveloperPromptPayload(BaseModel):
    """Request body for the native developer terminal system prompt."""

    prompt: str = ""


class DeveloperTerminalPayload(BaseModel):
    """Request body for one command sent to the mapped local shell."""

    command: str = Field(default="", max_length=8_000)


class CommunityDownloadPayload(BaseModel):
    """Request body for a direct Hugging Face Community download."""

    model: str = Field(min_length=1)


class DeleteModelPayload(BaseModel):
    """Request body for deleting local cached model."""

    model: str = Field(min_length=1)


class ModelCleanupPayload(BaseModel):
    """Request body for confirming model cache cleanup."""

    confirm: bool = False


class CommunityJobActionPayload(BaseModel):
    """Request body for controlling an active Community model load job."""

    action: Literal["pause", "resume", "cancel"]


class HfBindingPayload(BaseModel):
    """Request body for Hugging Face account binding."""

    username: str = ""
    token: str = ""
    seen: bool = True


class ServerModePayload(BaseModel):
    """Request body for server mode preference."""

    enabled: bool = False


class AgentDataTransferPayload(BaseModel):
    transfer: dict[str, Any]
    categories: list[str] = Field(default_factory=list)


class AgentDataExportPayload(BaseModel):
    categories: list[str] = Field(default_factory=list)


class RemoteServerTargetPayload(BaseModel):
    """SSH target for an Apple Silicon RoboServer host."""

    host: str = Field(min_length=1, max_length=253)
    username: str = Field(min_length=1, max_length=64)
    port: int = Field(default=22, ge=1, le=65535)
    identity_file: str = ""
    app_path: str = "~/Applications/token-workshed-roboserver.app"


class RemoteServerConnectPayload(RemoteServerTargetPayload):
    """Target plus the host fingerprint the user explicitly confirmed."""

    fingerprint: str = Field(min_length=1, max_length=100)


class ConcurrentModelsPayload(BaseModel):
    """Request body for configuring concurrently running auxiliary models."""

    models: list[str] = Field(default_factory=list)


class WorkshedWorkOrderPayload(BaseModel):
    """Versioned block work order submitted by the native Workshed canvas."""

    work_order: dict[str, Any] = Field(default_factory=dict)


class WorkshedUpdatePayload(BaseModel):
    """Optimistic-concurrency update for a saved work order."""

    work_order: dict[str, Any] = Field(default_factory=dict)
    expected_revision: int = Field(ge=1)


class WorkshedWorkOrderActionPayload(BaseModel):
    action: Literal["duplicate", "archive", "restore"]


class WorkshedPreflightPayload(BaseModel):
    work_order: dict[str, Any] | None = None
    work_order_id: str = Field(default="", max_length=96)


class WorkshedRunPayload(BaseModel):
    preflight_id: str = Field(min_length=1, max_length=96)
    accepted_consent_ids: list[str] = Field(default_factory=list)


class WorkshedRunActionPayload(BaseModel):
    action: Literal["cancel", "pause", "resume", "retry"]


class WorkshedArtifactActionPayload(BaseModel):
    action: Literal["register", "reveal", "export", "trash"]


class WorkshedToolchainPreparePayload(BaseModel):
    digests: list[str] = Field(default_factory=list, max_length=8)


WorkshedOptionProvider = Literal[
    "models.local",
    "models.managed",
    "models.hub",
    "datasets.hub",
    "lm_eval.tasks",
]


class WorkshedOptionsPayload(BaseModel):
    """Allowlisted dynamic option lookup for schema-driven Workshed fields."""

    model_config = ConfigDict(extra="forbid")

    provider: WorkshedOptionProvider
    query: str = Field(default="", max_length=256)
    limit: int = Field(default=24, ge=1, le=80)
    context: dict[str, Any] = Field(default_factory=dict, max_length=24)


class WorkshedOptionItem(BaseModel):
    """One stable option consumed by the native schema renderer."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(max_length=512)
    label: str = Field(max_length=512)
    description: str = Field(default="", max_length=1200)
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkshedOptionsResponse(BaseModel):
    """Uniform response for every Workshed option provider."""

    model_config = ConfigDict(extra="forbid")

    ok: bool = True
    provider: WorkshedOptionProvider
    items: list[WorkshedOptionItem] = Field(default_factory=list)
    next_cursor: str = ""
    message: str = Field(default="", max_length=1200)


class WorkshedReceiptResponse(BaseModel):
    """Envelope for an immutable or in-progress Workshed run receipt."""

    model_config = ConfigDict(extra="forbid")

    ok: bool = True
    receipt: dict[str, Any] = Field(default_factory=dict)


class _WorkshedModelResolver:
    """Expose only artifacts registered by SGLang Workshed runs."""

    def __init__(self, workshed: WorkshedService) -> None:
        self.workshed = workshed

    def __call__(self, model_id: str) -> str | None:
        return self.workshed.resolve_model(model_id)

    def managed_model_ids(self) -> list[str]:
        return self.workshed.managed_model_ids()

    def unregister_model(self, model_id: str) -> dict[str, Any]:
        return self.workshed.unregister_model(model_id)


# Release-scoped allowlist: Workshed records the exact task id in the RunPlan,
# so the UI must not offer newly installed lm-eval tasks that 0.2.0 has never
# exercised.  Expanding this set is an explicit release change.
_WORKSHED_LM_EVAL_TASK_ALLOWLIST: tuple[str, ...] = (
    "arc_challenge",
    "gsm8k",
    "hellaswag",
    "mmlu",
    "truthfulqa_mc2",
    "winogrande",
)


def _workshed_option_item(
    option_id: Any,
    label: Any,
    description: Any = "",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one bounded, JSON-safe option without forwarding credentials."""

    clean_id = str(option_id or "").strip()[:512]
    clean_label = str(label or clean_id).strip()[:512]
    clean_description = str(description or "").strip()[:1200]
    return {
        "id": clean_id,
        "label": clean_label or clean_id,
        "description": clean_description,
        "metadata": dict(metadata or {}),
    }


def _workshed_option_matches(query: str, *values: Any) -> bool:
    needle = str(query or "").strip().casefold()
    if not needle:
        return True
    return any(needle in str(value or "").casefold() for value in values)


def _workshed_options_envelope(
    provider: WorkshedOptionProvider,
    items: list[dict[str, Any]],
    *,
    message: str = "",
) -> dict[str, Any]:
    return {
        "ok": True,
        "provider": provider,
        "items": items,
        "next_cursor": "",
        "message": str(message or "").strip()[:1200],
    }


def _workshed_managed_model_options(
    query: str,
    limit: int,
    workshed_service: WorkshedService,
) -> tuple[list[dict[str, Any]], str]:
    options: dict[str, dict[str, Any]] = {}
    unavailable: list[str] = []
    try:
        model_ids = workshed_service.managed_model_ids()
    except Exception:
        unavailable.append("workshed")
        model_ids = []
    for model_id in model_ids:
        identifier = str(model_id or "").strip()
        if not identifier or not _workshed_option_matches(query, identifier, "workshed"):
            continue
        options.setdefault(
            identifier,
            _workshed_option_item(
                identifier,
                identifier,
                "Managed Workshed model",
                {"source": "managed", "registry": "workshed", "ref": identifier},
            ),
        )
    items = sorted(options.values(), key=lambda item: item["label"].casefold())[:limit]
    message = ""
    if unavailable:
        message = "Some managed model registries are temporarily unavailable."
    elif not items:
        message = "No managed models matched the query."
    return items, message


def _workshed_hub_dataset_options(
    query: str,
    limit: int,
) -> tuple[list[dict[str, Any]], str]:
    try:
        from huggingface_hub import HfApi
    except Exception:
        return [], "Hugging Face dataset search is unavailable in this installation."

    binding = _load_hf_binding_state()
    token = str(binding.get("token", "") or "").strip()
    try:
        api = HfApi(token=token) if token else HfApi()
    except TypeError:
        api = HfApi()

    base: dict[str, Any] = {
        "search": query or None,
        "sort": "downloads",
        "limit": min(80, max(limit * 2, limit)),
    }
    variants = [
        {"direction": -1, "full": True, **base},
        {"full": True, **base},
        {"direction": -1, **base},
        dict(base),
    ]
    datasets_iter = None
    try:
        for kwargs in variants:
            try:
                datasets_iter = api.list_datasets(**kwargs)
                break
            except TypeError:
                continue
        if datasets_iter is None:
            return [], "The installed Hugging Face client cannot search datasets."
        raw_items = list(datasets_iter)
    except Exception:
        return [], "Hugging Face dataset search is unavailable while offline."

    raw_items.sort(
        key=lambda item: int(getattr(item, "downloads", 0) or 0),
        reverse=True,
    )
    items: list[dict[str, Any]] = []
    for item in raw_items:
        dataset_id = str(getattr(item, "id", "") or "").strip()
        if not dataset_id or not _workshed_option_matches(query, dataset_id):
            continue
        description = str(getattr(item, "description", "") or "").strip()
        items.append(
            _workshed_option_item(
                dataset_id,
                dataset_id,
                description or "Hugging Face dataset",
                {
                    "source": "hub",
                    "ref": dataset_id,
                    "downloads": int(getattr(item, "downloads", 0) or 0),
                    "likes": int(getattr(item, "likes", 0) or 0),
                    "private": bool(getattr(item, "private", False)),
                    "gated": bool(getattr(item, "gated", False)),
                },
            )
        )
        if len(items) >= limit:
            break
    message = "" if items else "No Hugging Face datasets matched the query."
    return items, message


def _workshed_lm_eval_task_options(
    query: str,
    limit: int,
) -> tuple[list[dict[str, Any]], str]:
    try:
        from lm_eval.tasks import TaskManager
    except Exception:
        return [], "lm-eval tasks are unavailable until the evaluation toolchain is prepared."

    try:
        task_manager = TaskManager()
        raw_tasks = getattr(task_manager, "all_tasks", None)
        if isinstance(raw_tasks, dict):
            tasks = list(raw_tasks)
        elif isinstance(raw_tasks, (list, tuple, set)):
            tasks = [str(item) for item in raw_tasks]
        else:
            task_index = getattr(task_manager, "task_index", {})
            tasks = list(task_index) if isinstance(task_index, dict) else []
    except Exception:
        return [], "lm-eval task discovery is unavailable in the prepared environment."

    installed = {task.strip() for task in tasks if task.strip()}
    filtered = sorted(
        {
            task
            for task in _WORKSHED_LM_EVAL_TASK_ALLOWLIST
            if task in installed and _workshed_option_matches(query, task)
        },
        key=str.casefold,
    )[:limit]
    items = [
        _workshed_option_item(
            task,
            task,
            "lm-eval task",
            {
                "source": "lm_eval",
                "task": task,
                "allowlist_version": "0.2.0",
            },
        )
        for task in filtered
    ]
    message = "" if items else "No lm-eval tasks matched the query."
    return items, message


def _resolve_workshed_options(
    payload: WorkshedOptionsPayload,
    manager: Any,
    workshed_service: WorkshedService,
) -> dict[str, Any]:
    """Resolve one allowlisted provider without accepting URLs or package names."""

    provider = payload.provider
    query = payload.query.strip()
    limit = payload.limit

    if provider == "models.managed":
        items, message = _workshed_managed_model_options(
            query,
            limit,
            workshed_service,
        )
        return _workshed_options_envelope(provider, items, message=message)

    if provider == "models.local":
        try:
            managed_ids = set(workshed_service.managed_model_ids())
        except Exception:
            managed_ids = set()
        try:
            model_ids = manager.discover_models()
        except Exception:
            return _workshed_options_envelope(
                provider,
                [],
                message="The local model catalog is temporarily unavailable.",
            )
        # Workshed's local source runner needs a readable snapshot directory,
        # not the Hub repository id that the Chat model picker displays.
        snapshots: dict[str, list[tuple[bool, str, str, str]]] = {}
        try:
            from huggingface_hub import scan_cache_dir

            cache_info = scan_cache_dir()
            for repo in getattr(cache_info, "repos", []):
                repo_id = str(getattr(repo, "repo_id", "") or "").strip()
                if not repo_id or getattr(repo, "repo_type", None) not in (None, "model"):
                    continue
                for revision in getattr(repo, "revisions", []):
                    snapshot_path = Path(str(getattr(revision, "snapshot_path", "") or "")).expanduser()
                    if not snapshot_path.is_absolute():
                        continue
                    if not manager._snapshot_has_required_files(snapshot_path):
                        continue
                    refs = {str(value).casefold() for value in (getattr(revision, "refs", []) or [])}
                    snapshots.setdefault(repo_id, []).append((
                        "main" in refs,
                        str(getattr(revision, "last_modified", "") or ""),
                        str(getattr(revision, "commit_hash", "") or ""),
                        str(snapshot_path.resolve()),
                    ))
        except Exception:
            snapshots = {}

        items = []
        for model_id in sorted({str(item).strip() for item in model_ids}, key=str.casefold):
            if not model_id or model_id in managed_ids or not _workshed_option_matches(query, model_id):
                continue
            revisions = snapshots.get(model_id, [])
            if not revisions:
                continue
            revisions.sort(key=lambda item: (item[0], item[1]), reverse=True)
            _, _, commit_hash, snapshot_path = revisions[0]
            items.append(_workshed_option_item(
                snapshot_path,
                model_id,
                f"Local snapshot · {commit_hash[:8]}" if commit_hash else "Local Hugging Face snapshot",
                {"source": "local", "ref": snapshot_path, "hub_id": model_id, "commit_hash": commit_hash},
            ))
            if len(items) >= limit:
                break
        return _workshed_options_envelope(
            provider,
            items,
            message="" if items else "No local models matched the query.",
        )

    if provider == "models.hub":
        try:
            raw_items = manager.search_community_models(
                query,
                min(80, max(limit * 3, limit)),
            )
        except Exception:
            return _workshed_options_envelope(
                provider,
                [],
                message="Hugging Face model search is unavailable while offline.",
            )
        items: list[dict[str, Any]] = []
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            model_id = str(item.get("id", "") or "").strip()
            source = str(item.get("source", "") or "").strip().lower()
            if not model_id or source == "local":
                continue
            developer = str(item.get("developer", "") or "").strip()
            size = str(item.get("size_label", "") or "").strip()
            quantization = str(item.get("quantization_label", "") or "").strip()
            detail = " · ".join(value for value in (developer, size, quantization) if value and value != "-")
            items.append(
                _workshed_option_item(
                    model_id,
                    model_id,
                    detail or "Hugging Face model",
                    {
                        "source": "hub",
                        "ref": model_id,
                        "developer": developer,
                        "downloads": int(item.get("downloads", 0) or 0),
                        "likes": int(item.get("likes", 0) or 0),
                        "private": bool(item.get("private", False)),
                        "gated": bool(item.get("gated", False)),
                        "pipeline_tag": str(item.get("pipeline_tag", "") or ""),
                    },
                )
            )
            if len(items) >= limit:
                break
        return _workshed_options_envelope(
            provider,
            items,
            message=(
                ""
                if items
                else "No Hub models matched, or Hub search is unavailable offline."
            ),
        )

    if provider == "datasets.hub":
        items, message = _workshed_hub_dataset_options(query, limit)
        return _workshed_options_envelope(provider, items, message=message)

    if provider == "lm_eval.tasks":
        items, message = _workshed_lm_eval_task_options(query, limit)
        return _workshed_options_envelope(provider, items, message=message)

    raise HTTPException(
        status_code=422,
        detail="This Workshed option provider is unavailable in SGLang-only NextCore.",
    )


def _register_workshed_contract_routes(
    ui_app: Any,
    *,
    manager: Any,
    workshed_service: WorkshedService,
    manager_api_token: str | None,
) -> None:
    """Register independently testable Workshed option and receipt contracts."""

    @ui_app.post(
        "/api/manager/workshed/options",
        response_model=WorkshedOptionsResponse,
    )
    async def manager_workshed_options(
        request: Request,
        payload: WorkshedOptionsPayload,
    ) -> dict[str, Any]:
        _enforce_manager_auth(request, manager_api_token)
        return await run_in_threadpool(
            _resolve_workshed_options,
            payload,
            manager,
            workshed_service,
        )

    @ui_app.get(
        "/api/manager/workshed/runs/{run_id}/receipt",
        response_model=WorkshedReceiptResponse,
    )
    async def manager_workshed_run_receipt(
        request: Request,
        run_id: str,
    ) -> dict[str, Any]:
        _enforce_manager_auth(request, manager_api_token)
        try:
            receipt = await run_in_threadpool(workshed_service.run_receipt, run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"ok": True, "receipt": receipt}


class LocalModelServerManager:
    """Manage local SGLang servers and coordinate temporary Workshed jobs."""

    def __init__(
        self,
        python_executable: str,
        server_host: str,
        server_port: int,
        startup_timeout_s: float = 240,
    ):
        self.python_executable = python_executable
        self.server_host = server_host
        self.server_port = server_port
        self.server_url = f"http://{server_host}:{server_port}"
        self.startup_timeout_s = startup_timeout_s

        self._lock = threading.RLock()
        self._proc: subprocess.Popen[bytes] | None = None
        self._current_model = ""
        self._active_model = ""
        self._active_server_url = self.server_url
        self._active_reasoning_parser: str | None = None
        self._active_tool_call_parser: str | None = None
        self._extra_servers: dict[str, dict[str, Any]] = {}
        self._last_error = ""
        self._workbench_reserved = False
        self._runtime_config: dict[str, Any] = self._normalize_runtime_config({})
        self._managed_model_resolver: Callable[[str], str | None] | None = None
        self._community_job: dict[str, Any] | None = None
        self._community_download_candidates: dict[str, dict[str, Any]] = {}
        self._community_pause_event = threading.Event()
        self._community_cancel_event = threading.Event()
        self._integrity_checked = False
        self._integrity_report: dict[str, Any] = {
            "checked": False,
            "removed": [],
            "errors": [],
            "checked_at": None,
        }

    def set_managed_model_resolver(
        self, resolver: Callable[[str], str | None] | None
    ) -> None:
        """Attach the Workshed artifact registry without coupling the manager to it."""
        with self._lock:
            self._managed_model_resolver = resolver

    def _resolve_managed_model(self, model: str) -> str | None:
        resolver = self._managed_model_resolver
        if resolver is None:
            return None
        try:
            resolved = resolver(str(model or "").strip())
        except Exception:
            return None
        value = str(resolved or "").strip()
        return value or None

    def _normalize_runtime_config(self, raw: dict[str, Any] | None) -> dict[str, Any]:
        """Normalize runtime config and enforce safe bounds."""
        payload = dict(raw or {})

        def as_bool(key: str, default: bool = False) -> bool:
            value = payload.get(key, default)
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "on"}
            return bool(value)

        def as_int(key: str, default: int, min_value: int, max_value: int) -> int:
            value = payload.get(key, default)
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                parsed = default
            return max(min_value, min(max_value, parsed))

        cfg = {
            "continuous_batching": as_bool("continuous_batching", False),
            "use_paged_cache": as_bool("use_paged_cache", False),
            "kv_cache_quantization": as_bool("kv_cache_quantization", False),
            "chunked_prefill_tokens": as_int("chunked_prefill_tokens", 0, 0, 8192),
            "enable_mtp": as_bool("enable_mtp", False),
            "mtp_num_draft_tokens": as_int("mtp_num_draft_tokens", 1, 1, 8),
        }

        # These options are effective only in batched mode.
        if (
            cfg["use_paged_cache"]
            or cfg["kv_cache_quantization"]
            or cfg["chunked_prefill_tokens"] > 0
            or cfg["enable_mtp"]
        ):
            cfg["continuous_batching"] = True

        return cfg

    def _effective_reasoning_parser(
        self,
        model: str,
        *,
        model_path: str | None = None,
    ) -> str | None:
        """Resolve the reasoning parser from the model ID or local checkpoint."""
        canonical = _canonicalize_model_id(str(model or "").strip())
        if not canonical:
            return None
        parser = _detect_reasoning_parser(canonical)
        if parser:
            return parser

        resolved_path = str(model_path or "").strip()
        if not resolved_path:
            resolved_path = self._resolve_managed_model(canonical) or ""
        if not resolved_path and Path(canonical).expanduser().is_dir():
            resolved_path = canonical
        if not resolved_path:
            resolved_path = local_huggingface_snapshot(canonical)[0]
        return _detect_reasoning_parser(canonical, resolved_path or None)

    def _effective_tool_call_parser(self, model: str) -> str | None:
        """Infer SGLang's tool parser from the model family."""
        canonical = _canonicalize_model_id(str(model or "").strip())
        return _detect_tool_call_parser(canonical) if canonical else None

    def _purge_dead_extra_servers_locked(self) -> None:
        """Remove exited auxiliary model server records."""
        stale_models: list[str] = []
        for model_id, entry in self._extra_servers.items():
            proc = entry.get("proc")
            if not isinstance(proc, subprocess.Popen) or proc.poll() is not None:
                stale_models.append(model_id)
        for model_id in stale_models:
            self._extra_servers.pop(model_id, None)

    def _managed_pids_locked(self) -> set[int]:
        """Return process IDs currently managed by this manager."""
        pids: set[int] = set()
        if self._proc is not None and self._proc.poll() is None:
            pids.add(int(self._proc.pid))
        for entry in self._extra_servers.values():
            proc = entry.get("proc")
            if isinstance(proc, subprocess.Popen) and proc.poll() is None:
                pids.add(int(proc.pid))
        return pids

    def _is_primary_running_locked(self) -> bool:
        return (
            self._proc is not None
            and self._proc.poll() is None
            and bool(self._current_model)
        )

    def _refresh_active_target_locked(self) -> None:
        """Ensure active model pointer references a currently running server."""
        self._purge_dead_extra_servers_locked()

        if self._active_model:
            active_canonical = _canonicalize_model_id(self._active_model)
            if (
                active_canonical == _canonicalize_model_id(self._current_model)
                and self._is_primary_running_locked()
            ):
                self._active_model = self._current_model
                self._active_server_url = self.server_url
                self._active_reasoning_parser = self._effective_reasoning_parser(
                    self._current_model
                )
                self._active_tool_call_parser = self._effective_tool_call_parser(
                    self._current_model
                )
                return

            extra_entry = self._extra_servers.get(active_canonical)
            extra_proc = (
                extra_entry.get("proc") if isinstance(extra_entry, dict) else None
            )
            if isinstance(extra_proc, subprocess.Popen) and extra_proc.poll() is None:
                self._active_model = active_canonical
                self._active_server_url = str(
                    extra_entry.get("server_url", "") or ""
                ).strip()
                self._active_reasoning_parser = extra_entry.get("reasoning_parser")
                self._active_tool_call_parser = (
                    str(extra_entry.get("tool_call_parser", "")).strip() or None
                )
                return

        if self._is_primary_running_locked():
            self._active_model = self._current_model
            self._active_server_url = self.server_url
            self._active_reasoning_parser = self._effective_reasoning_parser(
                self._current_model
            )
            self._active_tool_call_parser = self._effective_tool_call_parser(
                self._current_model
            )
            return

        if self._extra_servers:
            first_model = sorted(self._extra_servers.keys(), key=lambda x: x.lower())[0]
            first_entry = self._extra_servers[first_model]
            self._active_model = first_model
            self._active_server_url = str(
                first_entry.get("server_url", "") or ""
            ).strip()
            self._active_reasoning_parser = first_entry.get("reasoning_parser")
            self._active_tool_call_parser = (
                str(first_entry.get("tool_call_parser", "")).strip() or None
            )
            return

        self._active_model = ""
        self._active_server_url = self.server_url
        self._active_reasoning_parser = None
        self._active_tool_call_parser = None

    def _find_free_extra_port_locked(self) -> int:
        """Pick an unused local port for auxiliary model servers."""
        used_ports: set[int] = {int(self.server_port)}
        for entry in self._extra_servers.values():
            try:
                port = int(entry.get("port", 0) or 0)
            except (TypeError, ValueError):
                port = 0
            if port > 0:
                used_ports.add(port)

        for candidate in range(int(self.server_port) + 1, int(self.server_port) + 400):
            if candidate in used_ports:
                continue
            if _list_listening_pids(candidate):
                continue
            if _can_connect_tcp(self.server_host, candidate):
                continue
            return candidate
        raise RuntimeError("No available port for auxiliary model server.")

    def _start_model_process_locked(
        self,
        model: str,
        *,
        port: int,
        runtime_config: dict[str, Any] | None = None,
        serve_model: str | None = None,
        backend: str = "mlx",
        cancel_event: threading.Event | None = None,
    ) -> tuple[bool, str, subprocess.Popen[bytes] | None, str | None]:
        """Start a model process on the requested port and wait for readiness."""
        canonical_model = _canonicalize_model_id(model.strip())
        if not canonical_model:
            return False, "Model name cannot be empty.", None, None
        if _is_hidden_model(canonical_model):
            return (
                False,
                "MLX-community checkpoints are not available in Token Workshed NextCore.",
                None,
                None,
            )
        if port <= 0:
            return False, "Invalid server port.", None, None

        managed_pids = self._managed_pids_locked()
        for pid in _list_listening_pids(port):
            if pid in managed_pids or pid == os.getpid():
                continue
            command = _read_pid_command(pid).strip() or "unknown process"
            if len(command) > 140:
                command = f"{command[:137]}..."
            return (
                False,
                f"Server port {port} is occupied by PID {pid} ({command}).",
                None,
                None,
            )

        if _can_connect_tcp(self.server_host, port):
            return (
                False,
                f"Server port {port} is already accepting connections.",
                None,
                None,
            )

        server_url = f"http://{self.server_host}:{port}"
        serve_target = str(serve_model or canonical_model).strip() or canonical_model
        reasoning_parser = self._effective_reasoning_parser(
            canonical_model,
            model_path=serve_target,
        )

        def _start_once(
            parser_override: str | None,
        ) -> tuple[bool, str, subprocess.Popen[bytes] | None]:
            try:
                cmd = self._build_serve_cmd(
                    serve_target, port=port, runtime_config=runtime_config,
                    tool_call_parser_override=parser_override, backend=backend,
                    served_model_name=canonical_model,
                )
            except ValueError as exc:
                return False, str(exc), None
            child_env = os.environ.copy()
            hf_binding = _load_hf_binding_state()
            hf_token = str(hf_binding.get("token", "") or "").strip()
            if hf_token:
                # SGLang's Hugging Face loader reads the standard token env var.
                # Keep the credential out of argv and limit it to this server process.
                child_env["HF_TOKEN"] = hf_token
                child_env["HUGGING_FACE_HUB_TOKEN"] = hf_token
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=child_env,
                start_new_session=True,
            )
            output_tail: deque[str] = deque(maxlen=24)
            output_lock = threading.Lock()

            def read_startup_output() -> None:
                stream = proc.stdout
                if stream is None:
                    return
                try:
                    for raw_line in iter(stream.readline, b""):
                        line = _redact_model_server_log(
                            raw_line.decode("utf-8", errors="replace").rstrip()
                        )
                        if line:
                            with output_lock:
                                output_tail.append(line)
                except (OSError, ValueError):
                    return

            output_reader = threading.Thread(
                target=read_startup_output,
                daemon=True,
                name=f"model-start-log-{canonical_model[:24]}",
            )
            output_reader.start()

            def startup_diagnostics() -> str:
                output_reader.join(timeout=0.5)
                with output_lock:
                    lines = list(output_tail)[-8:]
                return " | ".join(lines)[-2400:]

            deadline = time.time() + self.startup_timeout_s
            observed_models: list[str] = []

            while time.time() < deadline:
                if cancel_event is not None and cancel_event.is_set():
                    _terminate_process(proc)
                    return False, "Workbench operation cancelled during model startup.", None
                if proc.poll() is not None:
                    code = proc.returncode
                    detail = startup_diagnostics()
                    diagnostic_hint = f" Startup output: {detail}." if detail else " No startup output was captured."
                    return (
                        False,
                        (
                            f"Model '{canonical_model}' process exited before ready "
                            f"(exit code: {code}) on port {port}.{diagnostic_hint}"
                        ),
                        None,
                    )

                served_models = _query_served_model_ids(server_url)
                if served_models:
                    observed_models = served_models
                    served_set = {str(item).strip() for item in served_models}
                    acceptable_models = {
                        canonical_model,
                        _canonicalize_model_id(serve_target),
                        Path(serve_target).name,
                    }
                    if acceptable_models.intersection(served_set):
                        return (
                            True,
                            f"Model '{canonical_model}' is ready on port {port}.",
                            proc,
                        )
                time.sleep(0.6)

            _terminate_process(proc)
            detail = startup_diagnostics()
            observed_hint = (
                f" Observed model(s): {', '.join(observed_models[:6])}."
                if observed_models
                else ""
            )
            diagnostic_hint = f" Startup output: {detail}." if detail else ""
            return (
                False,
                (
                    f"Model '{canonical_model}' did not become ready within "
                    f"{self.startup_timeout_s}s on port {port}.{observed_hint}{diagnostic_hint}"
                ),
                None,
            )

        preferred_parser = self._effective_tool_call_parser(canonical_model)
        parser_name = (preferred_parser or "auto") if backend == "mlx" else None
        ok, message, proc = _start_once(parser_name)
        if not ok or proc is None:
            return False, message, None, None
        return True, message, proc, reasoning_parser

    def _select_running_model_locked(self, model: str) -> tuple[bool, str]:
        """Point active route to an already running model if available."""
        target = _canonicalize_model_id(model.strip())
        if not target:
            return False, "Model name cannot be empty."

        self._purge_dead_extra_servers_locked()
        if (
            target == _canonicalize_model_id(self._current_model)
            and self._is_primary_running_locked()
        ):
            self._active_model = self._current_model
            self._active_server_url = self.server_url
            self._active_reasoning_parser = self._effective_reasoning_parser(
                self._current_model
            )
            self._active_tool_call_parser = self._effective_tool_call_parser(
                self._current_model
            )
            _save_last_model_preference(self._active_model)
            self._last_error = ""
            return True, f"Model '{target}' is already active."

        entry = self._extra_servers.get(target)
        proc = entry.get("proc") if isinstance(entry, dict) else None
        if isinstance(proc, subprocess.Popen) and proc.poll() is None:
            self._active_model = target
            self._active_server_url = str(entry.get("server_url", "") or "").strip()
            self._active_reasoning_parser = entry.get("reasoning_parser")
            self._active_tool_call_parser = (
                str(entry.get("tool_call_parser", "")).strip() or None
            )
            _save_last_model_preference(self._active_model)
            self._last_error = ""
            return True, f"Model '{target}' is already running."

        return False, f"Model '{target}' is not running."

    def _cleanup_unmanaged_port_owners_locked(
        self,
    ) -> tuple[list[int], list[tuple[int, str]]]:
        """Stop orphan serve processes that keep occupying manager server port."""
        managed_pid = (
            int(self._proc.pid)
            if self._proc is not None and self._proc.poll() is None
            else -1
        )
        reclaimed: list[int] = []
        blocked: list[tuple[int, str]] = []
        for pid in _list_listening_pids(self.server_port):
            if pid <= 0 or pid == os.getpid() or pid == managed_pid:
                continue
            command = _read_pid_command(pid)
            if _looks_like_nextcore_serve_process(command):
                parent_pid = _read_pid_parent(pid)
                if (
                    parent_pid > 0
                    and parent_pid != os.getpid()
                    and _is_pid_running(parent_pid)
                ):
                    parent_command = _read_pid_command(parent_pid)
                    if _looks_like_desktop_manager_process(parent_command):
                        # Port takeover path: when another manager is still alive and
                        # owns the backend on this exact server port, stop that manager
                        # first so current UI instance can recover model switching.
                        parent_server_port = _command_flag_int(
                            parent_command,
                            "--server-port",
                        )
                        if parent_server_port == int(self.server_port):
                            _stop_pid(parent_pid, timeout_s=6.0)
                            time.sleep(0.2)
                            ok, _ = _stop_pid(pid, timeout_s=6.0)
                            if ok:
                                reclaimed.append(pid)
                                continue
                        blocked.append((pid, command))
                        continue
                ok, _ = _stop_pid(pid, timeout_s=6.0)
                if ok:
                    reclaimed.append(pid)
                    continue
            blocked.append((pid, command))
        return reclaimed, blocked

    def _verify_server_port_available_locked(self) -> tuple[bool, str]:
        """Ensure manager server port is free before spawning a model process."""
        reclaimed, _ = self._cleanup_unmanaged_port_owners_locked()
        if reclaimed:
            # Give OS a brief moment to release the port after SIGTERM/SIGKILL.
            time.sleep(0.25)

        managed_pid = (
            int(self._proc.pid)
            if self._proc is not None and self._proc.poll() is None
            else -1
        )

        remaining: list[tuple[int, str]] = []
        for pid in _list_listening_pids(self.server_port):
            if pid <= 0 or pid == os.getpid() or pid == managed_pid:
                continue
            remaining.append((pid, _read_pid_command(pid)))
        if remaining:
            blocker_pid, blocker_cmd = remaining[0]
            pretty_cmd = blocker_cmd.strip() or "unknown process"
            if len(pretty_cmd) > 140:
                pretty_cmd = f"{pretty_cmd[:137]}..."
            return (
                False,
                f"Server port {self.server_port} is occupied by PID {blocker_pid} "
                f"({pretty_cmd}).",
            )

        if _can_connect_tcp(self.server_host, self.server_port):
            return (
                False,
                f"Server port {self.server_port} is already accepting connections.",
            )
        return True, ""

    def _build_serve_cmd(
        self,
        model: str,
        *,
        port: int | None = None,
        runtime_config: dict[str, Any] | None = None,
        tool_call_parser_override: str | None = None,
        backend: str = "mlx",
        served_model_name: str | None = None,
    ) -> list[str]:
        cfg = (self._normalize_runtime_config(runtime_config)
               if runtime_config is not None else dict(self._runtime_config))
        if cfg.get("kv_cache_quantization") or cfg.get("enable_mtp"):
            raise ValueError("The selected KV quantization or MTP option is not supported by the SGLang Apple backend.")
        tool_parser = None
        reasoning_parser = None
        if backend == "mlx":
            tool_parser = (str(tool_call_parser_override or "").strip()
                           or self._effective_tool_call_parser(model) or "auto")
            reasoning_parser = self._effective_reasoning_parser(model)
        return server_command(
            self.python_executable, model, self.server_host,
            int(port) if isinstance(port, int) and port > 0 else self.server_port,
            backend=backend, served_model_name=served_model_name,
            tool_parser=tool_parser, reasoning_parser=reasoning_parser,
            chunked_prefill_size=int(cfg.get("chunked_prefill_tokens", 0) or 0),
        )

    def _start_locked(self, model: str) -> tuple[bool, str]:
        model = _canonicalize_model_id(model.strip())
        if not model:
            return False, "Model name cannot be empty."

        lock_timeout_s = max(30.0, min(float(self.startup_timeout_s) + 20.0, 420.0))
        try:
            with _acquire_backend_start_lock(timeout_s=lock_timeout_s):
                port_ok, port_message = self._verify_server_port_available_locked()
                if not port_ok:
                    self._active_reasoning_parser = None
                    self._active_tool_call_parser = None
                    self._last_error = port_message
                    return False, self._last_error

                ok, message, proc, reasoning_parser = self._start_model_process_locked(
                    model,
                    port=self.server_port,
                    runtime_config=self._runtime_config,
                    serve_model=(
                        self._resolve_managed_model(model)
                        or local_huggingface_snapshot(model)[0]
                        or model
                    ),
                )
                if not ok or proc is None:
                    self._active_reasoning_parser = None
                    self._active_tool_call_parser = None
                    self._last_error = message
                    return False, self._last_error
        except TimeoutError as exc:
            self._active_reasoning_parser = None
            self._active_tool_call_parser = None
            self._last_error = str(exc)
            return False, self._last_error

        self._proc = proc
        self._current_model = model
        self._active_model = model
        self._active_server_url = self.server_url
        self._active_reasoning_parser = reasoning_parser
        self._active_tool_call_parser = self._effective_tool_call_parser(model)
        self._last_error = ""
        _save_last_model_preference(model)
        if reasoning_parser:
            return True, f"{message} Reasoning parser: {reasoning_parser}"
        return True, message

    @contextmanager
    def workbench_runtime(
        self, model: str, *, backend: str = "mlx",
        cancel_event: threading.Event | None = None,
    ):
        """Temporarily release chat models and restore them after a Workshed job."""
        with self._lock:
            if self._workbench_reserved:
                raise RuntimeError("Another Workshed operation owns the model runtime.")
            primary = self._current_model if self._is_primary_running_locked() else ""
            extras = self.concurrent_models()
            active = self._active_model
            self._workbench_reserved = True
            self.stop()
        process = None
        try:
            ok, message, process, _ = self._start_model_process_locked(
                model, port=self.server_port, runtime_config={},
                serve_model=(
                    self._resolve_managed_model(model)
                    or local_huggingface_snapshot(model)[0]
                    or model
                ), backend=backend,
                cancel_event=cancel_event,
            )
            if not ok or process is None:
                raise RuntimeError(message)
            yield self.server_url
        finally:
            if process is not None:
                _terminate_process(process)
            with self._lock:
                self._workbench_reserved = False
                failures = []
                if primary:
                    ok, message = self._start_locked(primary)
                    if not ok:
                        failures.append(message)
                if extras:
                    ok, message = self.set_concurrent_models(extras)
                    if not ok:
                        failures.append(message)
                if active:
                    self._select_running_model_locked(active)
                if failures:
                    self._last_error = "Could not restore chat services: " + "; ".join(failures)
                    if sys.exc_info()[0] is None:
                        raise RuntimeError(self._last_error)

    def start(self, model: str) -> tuple[bool, str]:
        with self._lock:
            if self._workbench_reserved:
                return False, "The model runtime is temporarily reserved by Workshed."
            return self._start_locked(model)

    def ensure_primary_running(
        self,
        preferred_model: str | None = None,
    ) -> tuple[bool, str]:
        """Best-effort self-heal: restart primary model server if it exited unexpectedly."""
        with self._lock:
            if self._workbench_reserved:
                return False, "The model runtime is temporarily reserved by Workshed."
            if self._is_primary_running_locked():
                return True, "Primary model server is running."

            candidate = _canonicalize_model_id(
                str(
                    preferred_model
                    or self._current_model
                    or self._active_model
                    or _load_last_model_preference()
                    or ""
                ).strip()
            )
            if not candidate:
                return False, "No model target available for auto-recovery."

            ok, message = self._start_locked(candidate)
            if ok:
                return True, f"Auto-recovered model '{candidate}'."
            return False, message

    def stop(self) -> None:
        with self._lock:
            if self._proc is not None:
                _terminate_process(self._proc)
            for entry in self._extra_servers.values():
                proc = entry.get("proc")
                if isinstance(proc, subprocess.Popen):
                    _terminate_process(proc)
            self._proc = None
            self._extra_servers = {}
            self._current_model = ""
            self._active_model = ""
            self._active_server_url = self.server_url
            self._active_reasoning_parser = None
            self._active_tool_call_parser = None

    def is_running(self) -> bool:
        with self._lock:
            self._purge_dead_extra_servers_locked()
            if self._proc is not None and self._proc.poll() is None:
                return True
            return bool(self._extra_servers)

    def active_model(self) -> str:
        with self._lock:
            self._refresh_active_target_locked()
            return self._active_model

    def last_error(self) -> str:
        with self._lock:
            return self._last_error

    def active_reasoning_parser(self) -> str | None:
        with self._lock:
            self._refresh_active_target_locked()
            return self._active_reasoning_parser

    def reasoning_parser_for_model(self, model: str) -> str | None:
        """Return the parser configured for a served model or its checkpoint."""
        canonical = _canonicalize_model_id(str(model or "").strip())
        if not canonical:
            return None
        with self._lock:
            if canonical.casefold() == _canonicalize_model_id(
                self._current_model
            ).casefold():
                self._refresh_active_target_locked()
                if self._active_reasoning_parser:
                    return self._active_reasoning_parser
            for model_id, entry in self._extra_servers.items():
                if canonical.casefold() != _canonicalize_model_id(
                    model_id
                ).casefold():
                    continue
                parser = str(entry.get("reasoning_parser") or "").strip()
                if parser:
                    return parser
                break
        return self._effective_reasoning_parser(canonical)

    def active_tool_call_parser(self) -> str | None:
        with self._lock:
            self._refresh_active_target_locked()
            return self._active_tool_call_parser

    def active_server_url(self) -> str:
        with self._lock:
            self._refresh_active_target_locked()
            return self._active_server_url

    def model_server_urls(self) -> dict[str, str]:
        with self._lock:
            self._purge_dead_extra_servers_locked()
            mapping: dict[str, str] = {}
            if self._is_primary_running_locked() and self._current_model:
                mapping[_canonicalize_model_id(self._current_model)] = self.server_url
            for model_id, entry in self._extra_servers.items():
                proc = entry.get("proc")
                if not isinstance(proc, subprocess.Popen) or proc.poll() is not None:
                    continue
                url = str(entry.get("server_url", "") or "").strip()
                if url:
                    mapping[_canonicalize_model_id(model_id)] = url
            return mapping

    def running_models(self) -> list[dict[str, Any]]:
        with self._lock:
            self._refresh_active_target_locked()
            items: list[dict[str, Any]] = []
            if self._is_primary_running_locked() and self._current_model:
                items.append(
                    {
                        "model": _canonicalize_model_id(self._current_model),
                        "server_url": self.server_url,
                        "port": int(self.server_port),
                        "kind": "primary",
                        "active": (
                            _canonicalize_model_id(self._active_model)
                            == _canonicalize_model_id(self._current_model)
                            and self._active_server_url == self.server_url
                        ),
                        "reasoning_parser": self._effective_reasoning_parser(
                            self._current_model
                        ),
                        "tool_call_parser": self._effective_tool_call_parser(
                            self._current_model
                        ),
                    }
                )

            for model_id in sorted(self._extra_servers.keys(), key=lambda x: x.lower()):
                entry = self._extra_servers[model_id]
                proc = entry.get("proc")
                if not isinstance(proc, subprocess.Popen) or proc.poll() is not None:
                    continue
                items.append(
                    {
                        "model": _canonicalize_model_id(model_id),
                        "server_url": str(entry.get("server_url", "") or "").strip(),
                        "port": int(entry.get("port", 0) or 0),
                        "kind": "extra",
                        "active": (
                            _canonicalize_model_id(self._active_model)
                            == _canonicalize_model_id(model_id)
                        ),
                        "reasoning_parser": entry.get("reasoning_parser"),
                        "tool_call_parser": str(
                            entry.get("tool_call_parser", "")
                        ).strip()
                        or None,
                    }
                )
            return items

    def runtime_config(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._runtime_config)

    def concurrent_models(self) -> list[str]:
        with self._lock:
            self._purge_dead_extra_servers_locked()
            return sorted(self._extra_servers.keys(), key=lambda x: x.lower())

    def concurrent_model_state(self) -> dict[str, Any]:
        with self._lock:
            self._refresh_active_target_locked()
            return {
                "active_model": self._active_model,
                "active_server_url": self._active_server_url,
                "running_models": self.running_models(),
                "concurrent_models": self.concurrent_models(),
                "model_server_urls": self.model_server_urls(),
            }

    def set_concurrent_models(self, models: list[str]) -> tuple[bool, str]:
        """Ensure requested auxiliary models run concurrently on extra local ports."""
        with self._lock:
            if self._workbench_reserved:
                return False, "The model runtime is temporarily reserved by Workshed."
            self._refresh_active_target_locked()

            requested: list[str] = []
            seen: set[str] = set()
            for item in models:
                canonical = _canonicalize_model_id(str(item or "").strip())
                if not canonical or _is_hidden_model(canonical):
                    continue
                if canonical == _canonicalize_model_id(self._current_model):
                    continue
                if canonical in seen:
                    continue
                seen.add(canonical)
                requested.append(canonical)

            to_stop = [
                model_id for model_id in self._extra_servers if model_id not in seen
            ]
            for model_id in to_stop:
                entry = self._extra_servers.pop(model_id, None)
                if isinstance(entry, dict):
                    proc = entry.get("proc")
                    if isinstance(proc, subprocess.Popen):
                        _terminate_process(proc)

            start_failures: list[str] = []
            started: list[str] = []
            for model_id in requested:
                entry = self._extra_servers.get(model_id)
                proc = entry.get("proc") if isinstance(entry, dict) else None
                if isinstance(proc, subprocess.Popen) and proc.poll() is None:
                    continue
                if model_id in self._extra_servers:
                    self._extra_servers.pop(model_id, None)

                try:
                    port = self._find_free_extra_port_locked()
                except RuntimeError as exc:
                    start_failures.append(f"{model_id}: {exc}")
                    continue

                ok, message, proc_new, reasoning_parser = (
                    self._start_model_process_locked(
                        model_id,
                        port=port,
                        runtime_config=self._runtime_config,
                    )
                )
                if not ok or proc_new is None:
                    start_failures.append(f"{model_id}: {message}")
                    continue

                self._extra_servers[model_id] = {
                    "model": model_id,
                    "proc": proc_new,
                    "port": int(port),
                    "server_url": f"http://{self.server_host}:{port}",
                    "reasoning_parser": reasoning_parser,
                    "tool_call_parser": self._effective_tool_call_parser(model_id),
                }
                started.append(model_id)

            self._refresh_active_target_locked()
            if start_failures:
                self._last_error = " ; ".join(start_failures[:4])
                return (
                    False,
                    (
                        f"Concurrent model apply partially failed. "
                        f"Started: {len(started)}. Errors: {self._last_error}"
                    ),
                )

            self._last_error = ""
            return (
                True,
                (
                    f"Concurrent models updated. Running extras: {len(self._extra_servers)}."
                ),
            )

    def update_runtime_config(self, new_config: dict[str, Any]) -> tuple[bool, str]:
        """Apply runtime config and restart active model if running."""
        with self._lock:
            if self._workbench_reserved:
                return False, "The model runtime is temporarily reserved by Workshed."
            normalized = self._normalize_runtime_config(new_config)
            previous_config = dict(self._runtime_config)

            if normalized == previous_config:
                return True, "Runtime config unchanged."

            self._runtime_config = normalized

            if (
                self._proc is None
                or self._proc.poll() is not None
                or not self._current_model
            ):
                self._last_error = ""
                return (
                    True,
                    "Runtime config updated. It will apply on next model start.",
                )

            previous_model = self._current_model
            previous_proc = self._proc

            if previous_proc is not None:
                _terminate_process(previous_proc)
            self._proc = None
            self._current_model = ""
            self._active_reasoning_parser = None
            self._active_tool_call_parser = None

            ok, message = self._start_locked(previous_model)
            if ok:
                return True, f"{message} Runtime config applied."

            failed_message = message

            # Roll back config and try to recover previous runtime.
            self._runtime_config = previous_config
            rollback_ok, rollback_message = self._start_locked(previous_model)
            if rollback_ok:
                return (
                    False,
                    f"Apply failed: {failed_message} Restored previous runtime config.",
                )
            return (
                False,
                f"Apply failed: {failed_message} Rollback failed: {rollback_message}",
            )

    def switch_model(self, model: str) -> tuple[bool, str]:
        """Switch model by restarting serve process; rollback on failure."""
        target_model = _canonicalize_model_id(model.strip())
        if not target_model:
            return False, "Model name cannot be empty."

        with self._lock:
            if self._workbench_reserved:
                return False, "The model runtime is temporarily reserved by Workshed."
            select_ok, select_message = self._select_running_model_locked(target_model)
            if select_ok:
                return True, select_message

            if (
                self._proc is not None
                and self._proc.poll() is None
                and self._current_model == target_model
            ):
                self._active_model = self._current_model
                self._active_server_url = self.server_url
                self._active_reasoning_parser = self._effective_reasoning_parser(
                    self._current_model
                )
                self._active_tool_call_parser = self._effective_tool_call_parser(
                    self._current_model
                )
                _save_last_model_preference(self._current_model)
                self._last_error = ""
                return True, f"Model '{target_model}' is already active."

            prev_model = self._current_model
            prev_proc = self._proc

            if prev_proc is not None:
                _terminate_process(prev_proc)
            self._proc = None
            self._current_model = ""
            self._active_reasoning_parser = None
            self._active_tool_call_parser = None

            ok, message = self._start_locked(target_model)
            if ok:
                return True, message

            first_error = message

            # Retry once for transient start races (port release / process warmup).
            time.sleep(0.9)
            retry_ok, retry_message = self._start_locked(target_model)
            if retry_ok:
                return True, f"{retry_message} (Recovered on retry.)"

            # Compatibility fallback: temporarily disable advanced runtime flags
            # and try to start target model in safe defaults.
            runtime_before = dict(self._runtime_config)
            runtime_safe = self._normalize_runtime_config({})
            fallback_error = retry_message
            if runtime_before != runtime_safe:
                self._runtime_config = runtime_safe
                safe_ok, safe_message = self._start_locked(target_model)
                if safe_ok:
                    return (
                        True,
                        f"{safe_message} Runtime options auto-reset to safe defaults for compatibility.",
                    )
                fallback_error = safe_message
                self._runtime_config = runtime_before

            if prev_model:
                rollback_ok, rollback_msg = self._start_locked(prev_model)
                if rollback_ok:
                    return (
                        False,
                        f"Switch failed: {first_error} Retry failed: {retry_message} "
                        f"Fallback failed: {fallback_error} Rolled back to '{prev_model}'.",
                    )
                return (
                    False,
                    f"Switch failed: {first_error} Retry failed: {retry_message} "
                    f"Fallback failed: {fallback_error} Rollback failed: {rollback_msg}",
                )

            return (
                False,
                f"Switch failed: {first_error} Retry failed: {retry_message} Fallback failed: {fallback_error}",
            )

    def _snapshot_has_required_files(
        self, snapshot_path: str | os.PathLike[str]
    ) -> bool:
        """Heuristic check for a complete text model snapshot."""
        root = Path(snapshot_path)
        if not root.exists() or not root.is_dir():
            return False

        config_ok = (root / "config.json").exists()
        tokenizer_ok = any(
            (root / name).exists()
            for name in (
                "tokenizer.json",
                "tokenizer.model",
                "tokenizer_config.json",
                "vocab.json",
                "spiece.model",
            )
        )
        weights_ok = False
        for path in root.iterdir():
            if not path.is_file():
                continue
            name = path.name.lower()
            if (
                name.endswith(".safetensors")
                or name.endswith(".bin")
                or name.endswith(".gguf")
            ):
                weights_ok = True
                break
        return config_ok and tokenizer_ok and weights_ok

    def _cleanup_incomplete_models_locked(
        self,
        force: bool = False,
        apply_delete: bool = False,
    ) -> dict[str, Any]:
        # Reuse the last cache pass until a download/delete operation marks
        # the cache dirty.  The old condition bypassed this cache whenever
        # ``apply_delete`` was true, so normal discovery could repeatedly scan
        # the cache on every manager poll.
        if self._integrity_checked and not force:
            previous_pass_deleted = bool(self._integrity_report.get("apply_delete"))
            if not apply_delete or previous_pass_deleted:
                return dict(self._integrity_report)

        report: dict[str, Any] = {
            "checked": True,
            "removed": [],
            "pending": [],
            "errors": [],
            "checked_at": time.time(),
            "requires_confirmation": False,
            "apply_delete": bool(apply_delete),
        }

        try:
            from huggingface_hub import scan_cache_dir
        except Exception as exc:
            report["errors"].append(f"huggingface_hub unavailable: {exc}")
            self._integrity_checked = True
            self._integrity_report = report
            return dict(report)

        try:
            cache_info = scan_cache_dir()
        except Exception as exc:
            report["errors"].append(f"scan_cache_dir failed: {exc}")
            self._integrity_checked = True
            self._integrity_report = report
            return dict(report)

        candidates: list[dict[str, Any]] = []

        for repo in getattr(cache_info, "repos", []):
            repo_type = getattr(repo, "repo_type", None)
            repo_id = str(getattr(repo, "repo_id", "") or "").strip()
            if repo_type not in (None, "model") or not repo_id:
                continue

            running_models = {
                _canonicalize_model_id(self._current_model),
                *[_canonicalize_model_id(x) for x in self._extra_servers.keys()],
            }
            # Never remove running models.
            if _canonicalize_model_id(repo_id) in running_models:
                continue

            revisions = list(getattr(repo, "revisions", []) or [])
            if not revisions:
                repo_path_raw = getattr(repo, "repo_path", None)
                if repo_path_raw:
                    candidates.append(
                        {
                            "repo_id": repo_id,
                            "reason": "no_revisions",
                            "revision_hashes": [],
                            "repo_paths": [str(repo_path_raw)],
                        }
                    )
                continue

            has_complete_revision = False
            for rev in revisions:
                snapshot_path = getattr(rev, "snapshot_path", None)
                if snapshot_path and self._snapshot_has_required_files(snapshot_path):
                    has_complete_revision = True
                    break

            if has_complete_revision:
                continue

            revision_hashes: list[str] = []
            for rev in revisions:
                commit_hash = str(getattr(rev, "commit_hash", "") or "").strip()
                if commit_hash:
                    revision_hashes.append(commit_hash)

            repo_paths: list[str] = []
            if not revision_hashes:
                repo_path_raw = getattr(repo, "repo_path", None)
                if repo_path_raw:
                    repo_paths.append(str(repo_path_raw))

            if revision_hashes or repo_paths:
                candidates.append(
                    {
                        "repo_id": repo_id,
                        "reason": "incomplete",
                        "revision_hashes": sorted(set(revision_hashes)),
                        "repo_paths": repo_paths,
                    }
                )

        report["pending"] = [
            {"repo_id": item["repo_id"], "reason": item["reason"]}
            for item in candidates
        ]
        report["requires_confirmation"] = bool(candidates)

        if apply_delete:
            for item in candidates:
                repo_id = str(item.get("repo_id", "") or "").strip()
                reason = str(item.get("reason", "") or "").strip()
                revision_hashes = [
                    str(value).strip()
                    for value in item.get("revision_hashes", [])
                    if str(value).strip()
                ]
                repo_paths = [
                    Path(str(value).strip())
                    for value in item.get("repo_paths", [])
                    if str(value).strip()
                ]

                item_failed = False
                if revision_hashes:
                    try:
                        delete_strategy = cache_info.delete_revisions(
                            *sorted(set(revision_hashes))
                        )
                        delete_strategy.execute()
                    except Exception as exc:
                        item_failed = True
                        report["errors"].append(
                            f"delete_revisions failed ({repo_id}): {exc}"
                        )

                for repo_path in repo_paths:
                    try:
                        if repo_path.exists():
                            shutil.rmtree(repo_path, ignore_errors=False)
                    except Exception as exc:
                        item_failed = True
                        report["errors"].append(
                            f"delete repo path failed ({repo_path}): {exc}"
                        )

                if not item_failed:
                    report["removed"].append({"repo_id": repo_id, "reason": reason})

            # Preserve candidates whose deletion failed so the UI can explain
            # why an incomplete cache entry remains and allow a retry.
            removed_ids = {
                str(item.get("repo_id", "") or "").strip()
                for item in report["removed"]
                if str(item.get("repo_id", "") or "").strip()
            }
            report["pending"] = [
                item
                for item in report["pending"]
                if str(item.get("repo_id", "") or "").strip() not in removed_ids
            ]
            report["requires_confirmation"] = bool(report["pending"])

        self._integrity_checked = True
        self._integrity_report = report
        return dict(report)

    def validate_and_cleanup_models(
        self,
        force: bool = False,
        apply_delete: bool = False,
    ) -> dict[str, Any]:
        """Validate cache integrity, with optional confirmed deletion."""
        with self._lock:
            return self._cleanup_incomplete_models_locked(
                force=force,
                apply_delete=apply_delete,
            )

    def model_integrity_report(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._integrity_report)

    def discover_models(self, ensure_integrity: bool = True) -> list[str]:
        """Discover locally available models from HF cache + active model."""
        models: set[str] = set()

        with self._lock:
            if ensure_integrity:
                # Model discovery is the normal cache-health boundary. Any
                # incomplete, non-running snapshot is removed automatically;
                # explicit cleanup remains available for a forced retry.
                self._cleanup_incomplete_models_locked(force=False, apply_delete=True)

            if self._current_model:
                models.add(_canonicalize_model_id(self._current_model))

        try:
            from huggingface_hub import scan_cache_dir

            cache_info = scan_cache_dir()
            for repo in getattr(cache_info, "repos", []):
                repo_type = getattr(repo, "repo_type", None)
                repo_id = getattr(repo, "repo_id", None)
                if repo_type not in (None, "model"):
                    continue
                if isinstance(repo_id, str) and repo_id.strip():
                    models.add(_canonicalize_model_id(repo_id.strip()))
        except Exception:
            # Cache scan is best-effort.
            pass

        resolver = self._managed_model_resolver
        if resolver is not None:
            try:
                managed_ids = (
                    resolver
                    if callable(getattr(resolver, "managed_model_ids", None))
                    else getattr(resolver, "__self__", None)
                )
                managed_list = (
                    managed_ids.managed_model_ids()
                    if managed_ids is not None
                    and hasattr(managed_ids, "managed_model_ids")
                    else []
                )
                for model_id in managed_list:
                    if isinstance(model_id, str) and model_id.strip():
                        models.add(_canonicalize_model_id(model_id.strip()))
            except Exception:
                # Registry discovery is deliberately best effort; a broken
                # quantization artifact must not hide the HF cache.
                pass

        visible_models = [
            model_id for model_id in models if not _is_hidden_model(model_id)
        ]
        return sorted(visible_models, key=lambda x: x.lower())

    def delete_local_model(self, model: str) -> tuple[bool, str, list[str]]:
        """Unregister managed aliases or delete a local HF cache model."""
        target_raw = model.strip()
        target = _canonicalize_model_id(target_raw)
        if not target:
            return False, "Model name cannot be empty.", []

        with self._lock:
            self._purge_dead_extra_servers_locked()
            running_models = {
                _canonicalize_model_id(self._current_model),
                _canonicalize_model_id(self._active_model),
                *[_canonicalize_model_id(x) for x in self._extra_servers.keys()],
            }
            if target in running_models:
                return (
                    False,
                    "Cannot delete currently active model.",
                    self.discover_models(False),
                )

            if target.startswith("workshed/"):
                target_path = self._resolve_managed_model(target)
                if target_path and any(
                    self._resolve_managed_model(running) == target_path
                    or (Path(running).is_absolute() and str(Path(running).resolve()) == target_path)
                    for running in running_models
                    if running
                ):
                    return False, "Cannot delete a model whose artifact is currently running.", self.discover_models(False)
                resolver = self._managed_model_resolver
                owner = (
                    resolver
                    if callable(getattr(resolver, "unregister_model", None))
                    else getattr(resolver, "__self__", None)
                )
                unregister = getattr(owner, "unregister_model", None)
                if not callable(unregister):
                    return False, "Managed model removal is unavailable in this manager.", self.discover_models(False)
                try:
                    unregister(target)
                except (KeyError, ValueError, PermissionError, OSError) as exc:
                    return False, f"Failed to unregister '{target}': {exc}", self.discover_models(False)
                return (
                    True,
                    f"Unregistered local model '{target}'. Shared artifact content is retained until explicitly moved to Trash.",
                    self.discover_models(False),
                )

            try:
                from huggingface_hub import scan_cache_dir
            except Exception as exc:
                return (
                    False,
                    f"huggingface_hub unavailable: {exc}",
                    self.discover_models(False),
                )

            try:
                cache_info = scan_cache_dir()
            except Exception as exc:
                return (
                    False,
                    f"scan_cache_dir failed: {exc}",
                    self.discover_models(False),
                )

            target_repos = []
            for repo in getattr(cache_info, "repos", []):
                repo_id = str(getattr(repo, "repo_id", "") or "").strip()
                repo_id_canonical = _canonicalize_model_id(repo_id)
                repo_type = getattr(repo, "repo_type", None)
                if repo_type not in (None, "model"):
                    continue
                if (
                    repo_id == target_raw
                    or repo_id == target
                    or repo_id_canonical == target
                ):
                    target_repos.append(repo)

            if not target_repos:
                return (
                    False,
                    f"Model '{target}' not found in local cache.",
                    self.discover_models(False),
                )

            revision_hashes: set[str] = set()
            repo_paths: list[Path] = []
            for repo in target_repos:
                revisions = list(getattr(repo, "revisions", []) or [])
                for rev in revisions:
                    commit_hash = str(getattr(rev, "commit_hash", "") or "").strip()
                    if commit_hash:
                        revision_hashes.add(commit_hash)
                repo_path_raw = getattr(repo, "repo_path", None)
                if repo_path_raw:
                    repo_paths.append(Path(repo_path_raw))

            try:
                if revision_hashes:
                    cache_info.delete_revisions(*sorted(revision_hashes)).execute()
                for repo_path in repo_paths:
                    if repo_path.exists():
                        shutil.rmtree(repo_path, ignore_errors=True)
            except Exception as exc:
                return (
                    False,
                    f"Failed to delete '{target}': {exc}",
                    self.discover_models(False),
                )

            invalidate_local_snapshot_cache()
            self._integrity_checked = False
            models = self.discover_models(ensure_integrity=True)
            return True, f"Deleted local model '{target}'.", models

    def search_community_models(
        self, query: str, limit: int = 24, *, include_unready: bool = False
    ) -> list[dict[str, Any]]:
        """Search Transformer checkpoints that Hugging Face lists for SGLang."""
        q = query.strip()
        bounded_limit = max(1, min(80, int(limit)))
        with self._lock:
            # Only models from the latest visible search can be loaded with SGLang.
            self._community_download_candidates = {}

        try:
            from huggingface_hub import HfApi
        except Exception as exc:
            raise RuntimeError(
                "Hugging Face search is unavailable because huggingface_hub is not installed."
            ) from exc

        hf_binding = _load_hf_binding_state()
        hf_token = str(hf_binding.get("token", "") or "").strip()
        try:
            api = HfApi(token=hf_token) if hf_token else HfApi()
        except TypeError:
            # Older huggingface_hub may not accept token in constructor.
            api = HfApi()

        fetch_limit = min(400, max(bounded_limit * 8, 160))
        list_kwargs: dict[str, Any] = {
            "search": q or None,
            # `apps` comes from Hugging Face's Local Apps listing, so Community
            # only offers repositories that explicitly list SGLang.
            "apps": "sglang",
            # `filter` is the Hub API's library/tag filter. HfApi.list_models
            # does not accept a `library=` keyword in current Hub releases.
            "filter": "transformers",
            "sort": "downloads",
            "limit": fetch_limit,
            # Use full metadata so we can reliably detect gated/private models.
            "full": True,
        }

        signature_variants: list[dict[str, Any]] = [
            {"direction": -1, **list_kwargs},
            dict(list_kwargs),
            {"direction": -1, **{k: v for k, v in list_kwargs.items() if k != "full"}},
            {k: v for k, v in list_kwargs.items() if k != "full"},
        ]
        models_iter = None
        last_signature_error: Exception | None = None
        for kwargs in signature_variants:
            try:
                models_iter = api.list_models(**kwargs)
                break
            except TypeError as exc:
                last_signature_error = exc
                continue

        if models_iter is None:
            if last_signature_error is not None:
                raise last_signature_error
            raise RuntimeError("Failed to call Hugging Face list_models API.")

        raw_items = list(models_iter)
        raw_items.sort(
            key=lambda x: int(getattr(x, "downloads", 0) or 0),
            reverse=True,
        )

        try:
            cached_model_ids = set(self.discover_models(ensure_integrity=False))
        except Exception:
            cached_model_ids = set()
        community_results: list[dict[str, Any]] = []
        for item in raw_items:
            model_id = getattr(item, "id", None)
            if not isinstance(model_id, str) or not model_id.strip():
                continue
            model_id = _canonicalize_model_id(model_id.strip())
            pipeline_tag = getattr(item, "pipeline_tag", None)
            if pipeline_tag and pipeline_tag not in {
                "text-generation",
                "text2text-generation",
            }:
                continue

            tags_raw = getattr(item, "tags", None)
            tags = (
                [x for x in tags_raw if isinstance(x, str)]
                if isinstance(tags_raw, list)
                else []
            )
            library_name = str(getattr(item, "library_name", "") or "").strip().lower()
            developer_slug, developer = _normalize_developer_name(model_id)
            family_key, size_label = _infer_model_family_and_size(model_id)
            quant_label = _infer_quantization_label(model_id, tags)
            pipeline_supported = not pipeline_tag or pipeline_tag == "text-generation"
            format_hints = " ".join((model_id, library_name, *tags)).casefold()
            incompatible_format_hint = next(
                (
                    marker
                    for marker in ("gguf", "gptq", "awq", "exl2", "bitsandbytes")
                    if marker in format_hints
                ),
                "",
            )
            unsupported_format = (
                quant_label.strip().upper() in {"GGUF", "GPTQ", "AWQ", "EXL2"}
                or bool(incompatible_format_hint)
            )
            standard_hf_format = library_name == "transformers"
            if not standard_hf_format:
                continue
            pullable = (
                pipeline_supported
                and standard_hf_format
                and not unsupported_format
            )
            if not include_unready and not pullable:
                continue
            capability = _model_capability(model_id)
            query_score = _query_match_score(
                q,
                model_id,
                family_key,
                size_label,
                quant_label,
                developer_slug,
            )
            community_results.append(
                {
                    "id": model_id,
                    "developer": developer,
                    "developer_slug": developer_slug,
                    "family_key": f"{developer_slug}/{family_key}",
                    "family_name": family_key,
                    "size_label": size_label,
                    "param_size_label": (
                        size_label
                        if re.match(r"^\d+(?:\.\d+)?[BKM]$", size_label)
                        else "-"
                    ),
                    "variant_size_label": size_label,
                    "quantization_label": quant_label,
                    "downloads": int(getattr(item, "downloads", 0) or 0),
                    "likes": int(getattr(item, "likes", 0) or 0),
                    "pipeline_tag": pipeline_tag,
                    "library_name": library_name,
                    "pullable": pullable,
                    "downloadable": pullable,
                    "sglang_backend": "SGLang Runtime",
                    "model_format": "Hugging Face Transformers checkpoint",
                    "last_modified": (
                        str(getattr(item, "last_modified", ""))
                        if getattr(item, "last_modified", None) is not None
                        else None
                    ),
                    "private": bool(getattr(item, "private", False)),
                    "gated": bool(getattr(item, "gated", False)),
                    "local": model_id in cached_model_ids,
                    "deep_thinking": bool(capability.get("deep_thinking")),
                    "reasoning_parser": capability.get("reasoning_parser"),
                    "source": "community",
                    "source_priority": 1,
                    "query_score": query_score,
                }
            )
            if len(community_results) >= max(bounded_limit * 2, 60):
                break

        ranked_results = _merge_and_rank_search_results(
            [],
            community_results,
            max(len(community_results), bounded_limit),
        )
        if include_unready:
            ranked_results.sort(
                key=lambda result: (
                    not bool(result.get("pullable")),
                    -int(result.get("query_score", 0) or 0),
                    -int(result.get("downloads", 0) or 0),
                    -int(result.get("likes", 0) or 0),
                    str(result.get("id", "")).casefold(),
                )
            )
        visible_results = ranked_results[:bounded_limit]
        with self._lock:
            self._community_download_candidates = {
                str(result.get("id", "")): {
                    "pullable": bool(result.get("pullable")),
                    "local": bool(result.get("local")),
                }
                for result in visible_results
                if str(result.get("id", "")).strip()
            }
        return visible_results

    def _set_community_job_locked(self, **values: Any) -> dict[str, Any]:
        payload = dict(self._community_job or {})
        payload.update(values)
        now = time.time()
        payload.setdefault("updated_at", now)
        if "created_at" not in payload:
            payload["created_at"] = now
        if "updated_at" not in values:
            payload["updated_at"] = now
        self._community_job = payload
        return dict(self._community_job)

    def _run_hf_model_download_job(self, job_id: str, model_id: str) -> None:
        """Download a standard Hugging Face checkpoint without starting SGLang."""
        model = _canonicalize_model_id(str(model_id or "").strip())
        cancel_marker = "__community_job_canceled__"
        if not model:
            with self._lock:
                self._set_community_job_locked(
                    id=job_id,
                    model=model_id,
                    status="failed",
                    message="Model name cannot be empty.",
                    done=True,
                )
            return

        try:
            from huggingface_hub import snapshot_download
            from huggingface_hub.utils import tqdm as hf_tqdm

            manager = self

            class CommunityDownloadTqdm(hf_tqdm):
                """Report aggregate byte progress and honor download cancellation."""

                def __init__(self, *args: Any, **kwargs: Any) -> None:
                    unit = str(kwargs.get("unit", "")).strip().lower()
                    super().__init__(*args, **kwargs)
                    # snapshot_download passes this class to its aggregate
                    # byte bar, whose description changes across Hub versions
                    # (for example, "Downloading (incomplete total...)" in
                    # huggingface_hub 1.8). Unit is the stable discriminator.
                    self._tracks_download_bytes = unit in {"b", "byte", "bytes"}
                    self._tracks_files = unit in {"it", "file", "files"}
                    initial = kwargs.get("initial", 0) or 0
                    try:
                        self._workshed_progress_n = max(0, int(initial))
                    except (TypeError, ValueError, OverflowError):
                        self._workshed_progress_n = 0

                def update(self, n: int = 1) -> Any:
                    if manager._community_cancel_event.is_set():
                        raise RuntimeError(cancel_marker)
                    try:
                        self._workshed_progress_n = max(
                            0,
                            self._workshed_progress_n + int(n),
                        )
                    except (TypeError, ValueError, OverflowError):
                        pass
                    result = super().update(n)
                    if self._tracks_download_bytes:
                        # Hugging Face disables terminal bars in desktop mode;
                        # tqdm.update then returns before mutating self.n. Keep
                        # our own counter so the API still receives live bytes.
                        downloaded = self._workshed_progress_n
                        total = planned_total_bytes
                        percent = (
                            min(99.9, downloaded * 100.0 / total)
                            if total > 0
                            else 0.0
                        )
                        elapsed = max(0.0, time.monotonic() - download_started_at)
                        eta = None
                        if total > downloaded > 0 and elapsed > 0:
                            rate = downloaded / elapsed
                            if rate > 0:
                                eta = max(0, int((total - downloaded) / rate))
                        with manager._lock:
                            job = manager._community_job or {}
                            if (
                                job.get("id") != job_id
                                or job.get("status") in {
                                    "canceling",
                                    "canceled",
                                    "completed",
                                    "failed",
                                }
                            ):
                                return result
                            manager._set_community_job_locked(
                                progress_downloaded_bytes=downloaded,
                                progress_total_bytes=total,
                                progress_percent=percent,
                                progress_message=(
                                    f"Downloaded {downloaded:,} bytes"
                                    + (f" of {total:,} bytes." if total else ". Total size unavailable.")
                                ),
                                progress_eta_seconds=eta,
                            )
                    elif self._tracks_files and self.total:
                        completed = min(self._workshed_progress_n, int(self.total))
                        total = max(1, int(self.total))
                        percent = min(100.0, completed * 100.0 / total)
                        with manager._lock:
                            job = manager._community_job or {}
                            if job.get("id") == job_id and job.get("status") not in {
                                "canceling",
                                "canceled",
                                "completed",
                                "failed",
                            }:
                                manager._set_community_job_locked(
                                    progress_percent=percent,
                                    progress_message=(
                                        f"Downloaded {completed} of {total} Hugging Face files."
                                    ),
                                )
                    return result

            with self._lock:
                self._set_community_job_locked(
                    id=job_id,
                    model=model,
                    status="downloading",
                    message=f"Downloading the Hugging Face Transformers checkpoint for {model}.",
                    done=False,
                    progress_downloaded_bytes=0,
                    progress_total_bytes=0,
                    progress_percent=0.0,
                    progress_message="Preparing Hugging Face download...",
                    progress_eta_seconds=None,
                )

            if self._community_cancel_event.is_set():
                raise RuntimeError(cancel_marker)
            hf_token = str(_load_hf_binding_state().get("token", "") or "").strip()
            planned_total_bytes = 0
            try:
                dry_run_supported = (
                    "dry_run" in inspect.signature(snapshot_download).parameters
                )
            except (TypeError, ValueError):
                dry_run_supported = False
            if dry_run_supported:
                try:
                    plan = snapshot_download(
                        repo_id=model,
                        token=hf_token or None,
                        dry_run=True,
                    )
                    if isinstance(plan, list):
                        planned_total_bytes = sum(
                            max(0, int(item.file_size or 0))
                            for item in plan
                            if bool(getattr(item, "will_download", False))
                        )
                    with self._lock:
                        job = self._community_job or {}
                        if job.get("id") == job_id and job.get("status") == "downloading":
                            self._set_community_job_locked(
                                progress_total_bytes=planned_total_bytes,
                                progress_message=(
                                    "All checkpoint files are already cached."
                                    if planned_total_bytes == 0
                                    else f"Download size: {planned_total_bytes:,} bytes."
                                ),
                            )
                except Exception:
                    # Progress preflight is optional; let snapshot_download report
                    # the authoritative download error if the Hub cannot be reached.
                    planned_total_bytes = 0
                    with self._lock:
                        job = self._community_job or {}
                        if job.get("id") == job_id and job.get("status") == "downloading":
                            self._set_community_job_locked(
                                progress_total_bytes=0,
                                progress_message=(
                                    "Download total unavailable; reporting bytes received."
                                ),
                            )
            download_started_at = time.monotonic()
            if self._community_cancel_event.is_set():
                raise RuntimeError(cancel_marker)
            snapshot_download(
                repo_id=model,
                token=hf_token or None,
                tqdm_class=CommunityDownloadTqdm,
                max_workers=4,
            )
            if self._community_cancel_event.is_set():
                raise RuntimeError(cancel_marker)
            invalidate_local_snapshot_cache()
            with self._lock:
                completed_job = self._community_job or {}
                completed_bytes = planned_total_bytes or max(
                    0,
                    int(completed_job.get("progress_downloaded_bytes", 0) or 0),
                )
                self._set_community_job_locked(
                    id=job_id,
                    model=model,
                    status="completed",
                    message=(
                        f"Downloaded the Hugging Face checkpoint for {model}. "
                        "NextCore will start it directly with SGLang when selected."
                    ),
                    done=True,
                    progress_downloaded_bytes=completed_bytes,
                    progress_percent=100.0,
                    progress_message="Hugging Face download complete.",
                    progress_eta_seconds=0,
                )
        except Exception as exc:
            message = str(exc)
            lowered = message.lower()
            if cancel_marker in lowered or self._community_cancel_event.is_set():
                with self._lock:
                    self._set_community_job_locked(
                        id=job_id,
                        model=model,
                        status="canceled",
                        message=f"Canceled Hugging Face download for {model}.",
                        done=True,
                        progress_eta_seconds=None,
                    )
                return
            if "gated" in lowered or "401" in lowered or "unauthorized" in lowered:
                message = (
                    f"Access denied for '{model}'. This model is gated/private. "
                    f"Bind a Hugging Face token in Token Workshed, then request access at "
                    f"https://huggingface.co/{model}."
                )
            with self._lock:
                self._set_community_job_locked(
                    id=job_id,
                    model=model,
                    status="failed",
                    message=message,
                    done=True,
                    progress_eta_seconds=None,
                )
        finally:
            self._community_pause_event.clear()
            self._community_cancel_event.clear()
            # Hugging Face may have populated the shared cache while downloading.
            with self._lock:
                self._integrity_checked = False

    def start_community_model_download(
        self, model_id: str
    ) -> tuple[bool, str, dict[str, Any] | None]:
        """Download a searched Transformers checkpoint directly from Hugging Face."""
        model = _canonicalize_model_id(model_id.strip())
        if not model:
            return False, "Model name cannot be empty.", None

        with self._lock:
            candidate = dict(self._community_download_candidates.get(model) or {})
        if not candidate:
            return (
                False,
                "Search for this checkpoint in Community before downloading it.",
                None,
            )
        if not bool(candidate.get("pullable")):
            return (
                False,
                "This checkpoint is not a standard Hugging Face Transformers download.",
                None,
            )
        if bool(candidate.get("local")):
            return False, "This Hugging Face checkpoint is already in the local cache.", None

        with self._lock:
            if self._community_job and not bool(self._community_job.get("done", False)):
                running_model = (
                    str(self._community_job.get("model", "")).strip() or "current"
                )
                return (
                    False,
                    f"A community job is already running for '{running_model}'.",
                    dict(self._community_job),
                )

            job_id = uuid.uuid4().hex[:10]
            self._community_pause_event.clear()
            self._community_cancel_event.clear()
            self._set_community_job_locked(
                id=job_id,
                model=model,
                status="downloading",
                message=f"Starting direct Hugging Face download for {model}...",
                done=False,
                progress_downloaded_bytes=0,
                progress_total_bytes=0,
                progress_percent=0.0,
                progress_message="Preparing Hugging Face download...",
                progress_eta_seconds=None,
            )

        worker = threading.Thread(
            target=self._run_hf_model_download_job,
            args=(job_id, model),
            daemon=True,
        )
        worker.start()

        with self._lock:
            return True, "Hugging Face model download started.", dict(self._community_job)

    def control_community_job(
        self, action: Literal["pause", "resume", "cancel"]
    ) -> tuple[bool, str, dict[str, Any] | None]:
        """Pause, resume, or cancel active community job."""
        with self._lock:
            job = dict(self._community_job or {})
            if not job:
                return False, "No active community job.", None

            done = bool(job.get("done", False))
            status = str(job.get("status", "") or "").strip().lower()
            if done:
                return False, "Current community job is already finished.", dict(job)

            if action == "pause":
                return False, "Hugging Face downloads can be canceled but not paused here.", dict(job)

            if action == "resume":
                return False, "There is no paused Hugging Face download to resume.", dict(job)

            if action == "cancel":
                self._community_pause_event.clear()
                self._community_cancel_event.set()
                if status in {"queued", "paused"}:
                    self._set_community_job_locked(
                        status="canceled",
                        message=f"Canceled Hugging Face download for {job.get('model', 'model')}.",
                        done=True,
                        progress_eta_seconds=None,
                    )
                elif status in {"downloading", "deploying", "verifying"}:
                    self._set_community_job_locked(
                        status="canceling",
                        message=f"Stopping Hugging Face download for {job.get('model', 'model')}...",
                        done=False,
                        progress_eta_seconds=None,
                    )
                return (
                    True,
                    "Cancel signal sent to community job.",
                    dict(self._community_job),
                )

            return False, f"Unsupported action: {action}", dict(job)

    def community_job_status(self) -> dict[str, Any] | None:
        """Return the latest community job status."""
        with self._lock:
            if self._community_job is None:
                return None
            return dict(self._community_job)


def main() -> None:
    if len(sys.argv) >= 2 and sys.argv[1] == "--token-workshed-serve":
        from .cli import main as cli_main

        sys.argv = ["token-workshed-nextcore", "serve", *sys.argv[2:]]
        cli_main()
        return

    # NextCore owns one in-app SGLang Agent; purge persisted configuration
    # left behind by the retired OpenClaw/Hermes runtime selection.
    try:
        clear_legacy_agent_settings()
    except Exception:
        pass

    parser = argparse.ArgumentParser(
        description="token-workshed NextCore desktop launcher with SGLang inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  token-workshed-nextcore-desktop Qwen/Qwen3.5-2B
  token-workshed-nextcore-desktop --server-port 8010 --ui-port 7872
        """,
    )
    default_model = (
        os.environ.get(
            "TOKEN_WORKSHED_DEFAULT_MODEL",
            "ibm-granite/granite-4.0-h-350m",
        ).strip()
        or "ibm-granite/granite-4.0-h-350m"
    )
    parser.add_argument(
        "model",
        nargs="?",
        type=str,
        default="",
        help=(
            "Model name for the pinned SGLang runtime "
            f"(default: last used model, fallback {default_model})."
        ),
    )
    parser.add_argument(
        "--server-host",
        type=str,
        default="127.0.0.1",
        help="Host for SGLang server (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=8010,
        help="Port for SGLang server (default: 8010)",
    )
    parser.add_argument(
        "--ui-host",
        type=str,
        default="127.0.0.1",
        help="Host for UI server (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--ui-port",
        type=int,
        default=7872,
        help="Port for UI server (default: 7872)",
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
        "--browser",
        action="store_true",
        help="Also open the local API status page in the system browser",
    )
    parser.add_argument(
        "--allow-remote-ui",
        action="store_true",
        help=(
            "Allow non-loopback UI host. Requires manager API token header "
            "`X-Token-Workshed-Manager-Token` for manager endpoints."
        ),
    )
    parser.add_argument(
        "--initial-page",
        type=str,
        default="chat",
        help=(
            "Initial native page. Workshed accepts legacy aliases "
            "quantize/quantization (default: chat)."
        ),
    )
    parser.add_argument(
        "--background",
        action="store_true",
        help="Run desktop manager in background (no window), then return.",
    )
    parser.add_argument(
        "--background-stop",
        action="store_true",
        help="Stop the recorded background desktop manager process.",
    )
    parser.add_argument(
        "--background-status",
        action="store_true",
        help="Show whether background desktop manager process is running.",
    )
    parser.add_argument(
        "--serve-only",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--roboserver",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-model-autostart",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    requested_model = str(args.model or "").strip()

    if args.roboserver:
        if not _is_loopback_host(args.ui_host) or not _is_loopback_host(args.server_host):
            _fatal("RoboServer APIs and inference must remain bound to loopback; remote access uses SSH tunnels.")
        if sys.platform != "darwin" or __import__("platform").machine().lower() not in {"arm64", "aarch64"}:
            _fatal("token-workshed-roboserver currently supports Apple Silicon macOS only.")

    bg_actions = sum(
        int(flag)
        for flag in (
            bool(args.background),
            bool(args.background_stop),
            bool(args.background_status),
        )
    )
    if bg_actions > 1:
        _fatal("Use only one of --background, --background-stop, --background-status.")

    if not _is_loopback_host(args.ui_host) and not args.allow_remote_ui:
        _fatal(
            "Refusing non-loopback UI host by default. "
            "Use --allow-remote-ui only in trusted networks."
        )

    server_url = f"http://{args.server_host}:{args.server_port}"
    ui_url = f"http://{args.ui_host}:{args.ui_port}"

    if args.background_status:
        bg_state = _read_background_state()
        pid = int(bg_state.get("pid", 0) or 0)
        if pid > 0 and _is_pid_running(pid):
            ui_target = str(bg_state.get("ui_url", "") or "").strip() or ui_url
            server_target = (
                str(bg_state.get("server_url", "") or "").strip() or server_url
            )
            ready = _wait_http_ok(f"{ui_target}/api/config", timeout_s=1.2)
            print("Background mode: running")
            print(f"PID: {pid}")
            print(f"UI: {ui_target} ({'ready' if ready else 'starting'})")
            print(f"Server: {server_target}")
            return
        if pid > 0:
            _clear_background_state()
        print("Background mode: stopped")
        return

    if args.background_stop:
        bg_state = _read_background_state()
        pid = int(bg_state.get("pid", 0) or 0)
        if pid <= 0:
            print("No recorded background process.")
            return
        ok, message = _stop_pid(pid)
        _clear_background_state()
        if ok:
            print(message)
            return
        _fatal(message)

    if args.background:
        bg_state = _read_background_state()
        existing_pid = int(bg_state.get("pid", 0) or 0)
        if existing_pid > 0 and _is_pid_running(existing_pid):
            ui_target = str(bg_state.get("ui_url", "") or "").strip() or ui_url
            print(f"Background process already running (PID {existing_pid}).")
            print(f"UI: {ui_target}")
            return
        if existing_pid > 0:
            _clear_background_state()

        reclaimed_pids, blocked_pids = _cleanup_stale_desktop_ui_port_owners(
            args.ui_port,
            keep_pid=existing_pid,
        )
        if reclaimed_pids:
            # Give kernel a short interval to release the port after SIGTERM/SIGKILL.
            time.sleep(0.35)
        if blocked_pids:
            pid, command = blocked_pids[0]
            hint = command or "(unknown command)"
            _fatal(
                f"UI port {args.ui_port} is occupied by PID {pid} ({hint}). "
                "Stop that process or choose another --ui-port."
            )

        log_path = _desktop_state_file().parent / "desktop_background.log"
        cmd = _build_background_launch_cmd(args)
        with log_path.open("ab") as log_file:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )

        time.sleep(1.2)
        exit_code = proc.poll()
        if exit_code is not None:
            _fatal(
                f"Background launch failed (exit code {exit_code}). "
                f"Check log: {log_path}"
            )

        _save_background_state(
            pid=proc.pid,
            server_url=server_url,
            ui_url=ui_url,
            model=requested_model,
        )
        ready = _wait_http_ok(f"{ui_url}/api/config", timeout_s=6.0)
        print(f"Background process started (PID {proc.pid}).")
        print(f"UI: {ui_url}")
        print(f"Server: {server_url}")
        print(f"Status: {'ready' if ready else 'starting'}")
        print(f"Log: {log_path}")
        return

    last_used_model = _load_last_model_preference()
    manager_api_token: str | None = None
    if args.roboserver:
        manager_api_token = str(os.environ.get("TOKEN_WORKSHED_MANAGER_TOKEN", "") or "").strip() or secrets.token_urlsafe(48)
    elif args.allow_remote_ui:
        manager_api_token = str(
            os.environ.get("TOKEN_WORKSHED_MANAGER_TOKEN", "") or ""
        ).strip() or secrets.token_urlsafe(24)
    agent_ui_token = (
        str(os.environ.get("TOKEN_WORKSHED_AGENT_UI_TOKEN", "") or "").strip()
        if args.roboserver
        else ""
    ) or secrets.token_urlsafe(48)

    workshed_service = WorkshedService()
    manager = LocalModelServerManager(
        python_executable=sys.executable,
        server_host=args.server_host,
        server_port=args.server_port,
    )
    workshed_service.set_runtime_manager(manager)
    manager.set_managed_model_resolver(_WorkshedModelResolver(workshed_service))

    saved_server_connection = _read_desktop_state().get("server_connection", {})
    remote_selected = bool(
        not args.roboserver
        and isinstance(saved_server_connection, dict)
        and str(saved_server_connection.get("mode", "local")).strip().lower() == "remote"
    )
    if remote_selected:
        # A disconnected remote selection must never leak local cache entries or
        # silently fall back to a local model server.
        manager.stop()

    # Startup integrity pass: remove incomplete, non-running cache entries
    # before exposing the model list to the UI.
    try:
        manager.validate_and_cleanup_models(force=True, apply_delete=True)
    except Exception:
        pass

    try:
        local_models = manager.discover_models(ensure_integrity=False)
    except Exception:
        local_models = []

    initial_model = ""
    model_source = "No local model found"
    if remote_selected:
        model_source = "remote RoboServer selected; local inference is disabled"
    elif requested_model:
        initial_model = requested_model
        model_source = "CLI argument"
    elif last_used_model and last_used_model in local_models:
        initial_model = last_used_model
        model_source = "last used model preference"
    elif default_model and default_model in local_models:
        initial_model = default_model
        model_source = "TOKEN_WORKSHED_DEFAULT_MODEL fallback"
    elif local_models:
        initial_model = local_models[0]
        model_source = "first discovered local model"

    print("=" * 60)
    print("Launching token-workshed NextCore desktop UI")
    print(f"Initial model: {initial_model or '(none)'}")
    print(f"Model source: {model_source}")
    print(f"Local models detected: {len(local_models)}")
    print(f"Server URL: {server_url}")
    print(f"UI URL: {ui_url}")
    if manager_api_token:
        print("Remote UI mode: enabled")
        print("Manager API header: X-Token-Workshed-Manager-Token")
        print(f"Manager API token: {manager_api_token}")
    print("=" * 60)

    if initial_model and not args.no_model_autostart and not remote_selected:
        ok, msg = manager.start(initial_model)
        if ok:
            print(msg)
            _developer_log(f"initial model started model={initial_model}: {msg}")
        else:
            manager.stop()
            print(f"Warning: failed to start initial model '{initial_model}': {msg}")
            print("Fallback: launching UI without active model.")
            _developer_log(f"initial model failed model={initial_model}: {msg}")
    elif not initial_model:
        print("No local model detected. UI starts without an active model.")
        _developer_log("native UI started without active local model")
    else:
        print("RoboServer starts without loading a model. Choose one from its Dashboard.")

    ui_app = create_app(
        default_server_url=server_url,
        default_max_tokens=args.max_tokens,
        default_temperature=args.temperature,
        warm_agent_on_startup=bool(args.roboserver or not remote_selected),
    )
    ui_app.state.manager_auth_required = bool(manager_api_token)
    ui_app.state.manager_api_token = manager_api_token
    ui_app.state.agent_ui_token = agent_ui_token
    ui_app.state.roboserver_mode = bool(args.roboserver)
    ui_app.state.model_manager = manager
    server_connection = (
        None
        if args.roboserver
        else ServerConnectionController(
            TOKEN_WORKSHED_ROOT,
            local_inference_port=args.server_port,
        )
    )
    ui_app.state.server_connection_controller = server_connection
    ui_app.state.keep_running_on_window_close = bool(
        server_connection and server_connection.local_server_enabled()
    )
    ui_app.state.developer_system_prompt = ""
    terminal_session = _DeveloperTerminalSession(TOKEN_WORKSHED_ROOT)
    ui_app.state.developer_terminal = terminal_session
    ui_app.state.developer_status_line = ""
    ui_app.state.developer_status_logged_at = 0.0
    ui_app.state.workshed_service = workshed_service

    remote_reconnect_stop = threading.Event()
    remote_reconnect_thread: threading.Thread | None = None
    if server_connection is not None:
        def _remote_reconnect_loop() -> None:
            while not remote_reconnect_stop.wait(5.0):
                if server_connection.mode() != "remote":
                    continue
                if server_connection.status().get("connected"):
                    continue
                result = server_connection.restore()
                if result.get("connected"):
                    ui_app.state.server_url = str(result.get("server_url") or server_url)
                    ui_app.state.keep_running_on_window_close = False
                    _developer_log("remote RoboServer SSH tunnel restored")

        remote_reconnect_thread = threading.Thread(
            target=_remote_reconnect_loop,
            daemon=True,
            name="nextcore-roboserver-reconnect",
        )
        remote_reconnect_thread.start()

    def launch_ide_companion() -> bool:
        """Start a second, compact Rust/libcosmic workbench for the IDE."""
        companion_binary = _find_native_ui_binary()
        if companion_binary is None:
            raise RuntimeError(
                "Native UI binary was not found; build native-ui with cargo first."
            )
        companion_cmd = [
            str(companion_binary),
            "--api-base",
            ui_url,
            "--server-url",
            server_url,
            "--initial-page",
            "chat",
            "--ide-companion",
        ]
        companion_env = os.environ.copy()
        if manager_api_token:
            companion_cmd.extend(["--manager-token", manager_api_token])
            companion_env["TOKEN_WORKSHED_MANAGER_TOKEN"] = manager_api_token
        process = subprocess.Popen(
            companion_cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=companion_env,
            start_new_session=True,
            close_fds=True,
        )
        _developer_log(f"IDE Companion launched pid={process.pid}")
        return process.poll() is None

    # The IDE bridge is deliberately separate from the manager and developer
    # endpoints below. A paired JetBrains client gets scoped v1/v2 access; it
    # never receives the manager token or a local shell/Agent credential.
    ide_bridge = IDEBridge(
        get_models=manager.discover_models,
        get_active_model=manager.active_model,
        switch_model=manager.switch_model,
        get_server_url=manager.active_server_url,
        default_server_url=server_url,
        default_max_tokens=args.max_tokens,
        default_temperature=args.temperature,
        launch_companion=launch_ide_companion,
    )
    register_ide_bridge_routes(ui_app, ide_bridge)
    register_jetbrains_integration_routes(
        ui_app,
        manager=manager,
        ui_url=ui_url,
        manager_api_token=manager_api_token,
        bridge=ide_bridge,
    )
    _register_workshed_contract_routes(
        ui_app,
        manager=manager,
        workshed_service=workshed_service,
        manager_api_token=manager_api_token,
    )

    @ui_app.on_event("shutdown")
    async def _shutdown_developer_terminal() -> None:
        terminal_session.stop()
        workshed_service.shutdown()
        remote_reconnect_stop.set()
        controller = ui_app.state.server_connection_controller
        if controller is not None:
            controller.disconnect(persist=False)

    _developer_log(
        f"desktop backend ready ui={ui_url} server={server_url} initial_model={initial_model or '(none)'}"
    )

    @ui_app.get("/api/developer/logs")
    async def developer_logs(request: Request, limit: int = 80) -> dict[str, Any]:
        """Return timestamped desktop-manager diagnostics for compatibility."""
        _enforce_manager_auth(request, manager_api_token)
        active = await run_in_threadpool(manager.active_model)
        running = await run_in_threadpool(manager.is_running)
        server = await run_in_threadpool(manager.active_server_url)
        status_line = (
            f"status running={running} active_model={active or '(none)'} "
            f"server={server or manager.server_url}"
        )
        now = time.time()
        previous = str(getattr(ui_app.state, "developer_status_line", "") or "")
        previous_at = float(
            getattr(ui_app.state, "developer_status_logged_at", 0.0) or 0.0
        )
        # Polling is intentionally frequent while diagnostics are open. Only
        # emit a status line when it changes (or every five seconds) so the
        # diagnostic tail remains useful instead of filling with heartbeats.
        if status_line != previous or now - previous_at >= 5.0:
            _developer_log(status_line)
            ui_app.state.developer_status_line = status_line
            ui_app.state.developer_status_logged_at = now
        logs = _DEVELOPER_LOGS.tail(limit)
        return {
            "ok": True,
            "logs": logs[-max(1, min(200, int(limit or 80))) :],
        }

    @ui_app.get("/api/developer/terminal/output")
    async def developer_terminal_output(
        request: Request, limit: int = 120
    ) -> dict[str, Any]:
        """Return only raw command-line output for the native Terminal page."""
        _enforce_manager_auth(request, manager_api_token)
        return {
            "ok": True,
            "lines": _DEVELOPER_TERMINAL_OUTPUT.tail(limit),
        }

    @ui_app.post("/api/developer/prompt")
    async def developer_prompt(
        request: Request,
        payload: DeveloperPromptPayload,
    ) -> dict[str, Any]:
        _enforce_manager_auth(request, manager_api_token)
        prompt = str(payload.prompt or "").strip()
        ui_app.state.developer_system_prompt = prompt
        _developer_log(
            f"system prompt updated chars={len(prompt)} applies_to=future_chat_requests"
        )
        return {
            "ok": True,
            "message": "System prompt applied to future SGLang chat requests.",
            "prompt_chars": len(prompt),
        }

    @ui_app.post("/api/developer/terminal")
    async def developer_terminal(
        request: Request,
        payload: DeveloperTerminalPayload,
    ) -> dict[str, Any]:
        """Send one command to the stateful shell mirrored by the native UI."""
        _enforce_manager_auth(request, manager_api_token)
        command = str(payload.command or "").strip()
        if not command:
            raise HTTPException(
                status_code=400, detail="Terminal command cannot be empty."
            )
        try:
            message = await run_in_threadpool(terminal_session.send, command)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            _terminal_output(f"nextcore: {exc}")
            _developer_log(f"[terminal] command failed: {exc}")
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {"ok": True, "message": message}

    @ui_app.get("/api/manager/models")
    async def manager_models(request: Request) -> dict[str, Any]:
        _enforce_manager_auth(request, manager_api_token)
        models = await run_in_threadpool(manager.discover_models)
        active = await run_in_threadpool(manager.active_model)

        if active and active not in models and not _is_hidden_model(active):
            models = [active, *models]

        model_capabilities = _model_capability_map(models)
        running_models = await run_in_threadpool(manager.running_models)
        runtime_config = await run_in_threadpool(manager.runtime_config)
        model_details = await run_in_threadpool(
            _build_model_details,
            models,
            active_model=active,
            running_models=running_models,
            runtime_config=runtime_config,
            managed_model_resolver=manager._resolve_managed_model,
        )
        deep_thinking_models = [
            model_id
            for model_id, cap in model_capabilities.items()
            if bool(cap.get("deep_thinking"))
        ]

        return {
            "ok": True,
            "manager": "desktop",
            "server_url": manager.server_url,
            "active_model": active,
            "active_server_url": await run_in_threadpool(manager.active_server_url),
            "available_models": models,
            "count": len(models),
            "model_capabilities": model_capabilities,
            "deep_thinking_models": deep_thinking_models,
            "model_details": model_details,
            "running_models": running_models,
            "concurrent_models": await run_in_threadpool(manager.concurrent_models),
            "model_server_urls": await run_in_threadpool(manager.model_server_urls),
            "active_reasoning_parser": await run_in_threadpool(
                manager.active_reasoning_parser
            ),
            "active_tool_call_parser": await run_in_threadpool(
                manager.active_tool_call_parser
            ),
            "runtime_config": runtime_config,
            "community_job": await run_in_threadpool(manager.community_job_status),
            "integrity_report": await run_in_threadpool(manager.model_integrity_report),
            "running": await run_in_threadpool(manager.is_running),
            "last_error": await run_in_threadpool(manager.last_error),
        }

    @ui_app.get("/api/manager/performance")
    async def manager_performance(request: Request) -> dict[str, Any]:
        _enforce_manager_auth(request, manager_api_token)
        return await run_in_threadpool(_performance_snapshot)

    # ------------------------------------------------------------------
    # Local Workshed block compiler/coordinator.
    # ------------------------------------------------------------------
    @ui_app.get("/api/manager/workshed/capabilities")
    async def manager_workshed_capabilities(request: Request) -> dict[str, Any]:
        _enforce_manager_auth(request, manager_api_token)
        return await run_in_threadpool(workshed_service.capabilities)

    @ui_app.get("/api/manager/workshed/work-orders")
    async def manager_workshed_work_orders(request: Request) -> dict[str, Any]:
        _enforce_manager_auth(request, manager_api_token)
        return {"ok": True, "work_orders": await run_in_threadpool(workshed_service.list_work_orders)}

    @ui_app.post("/api/manager/workshed/work-orders")
    async def manager_workshed_work_order_create(
        request: Request,
        payload: WorkshedWorkOrderPayload,
    ) -> dict[str, Any]:
        _enforce_manager_auth(request, manager_api_token)
        try:
            order = await run_in_threadpool(workshed_service.create_work_order, payload.work_order)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, "work_order": order}

    @ui_app.get("/api/manager/workshed/work-orders/{work_order_id}")
    async def manager_workshed_work_order(
        request: Request,
        work_order_id: str,
    ) -> dict[str, Any]:
        _enforce_manager_auth(request, manager_api_token)
        order = await run_in_threadpool(workshed_service.get_work_order, work_order_id)
        if not order:
            raise HTTPException(status_code=404, detail="Workshed work order not found.")
        return {"ok": True, "work_order": order}

    @ui_app.put("/api/manager/workshed/work-orders/{work_order_id}")
    async def manager_workshed_work_order_update(
        request: Request,
        work_order_id: str,
        payload: WorkshedUpdatePayload,
    ) -> dict[str, Any]:
        _enforce_manager_auth(request, manager_api_token)
        try:
            order = await run_in_threadpool(
                workshed_service.update_work_order,
                work_order_id,
                payload.work_order,
                payload.expected_revision,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"ok": True, "work_order": order}

    @ui_app.post("/api/manager/workshed/work-orders/{work_order_id}/action")
    async def manager_workshed_work_order_action(
        request: Request,
        work_order_id: str,
        payload: WorkshedWorkOrderActionPayload,
    ) -> dict[str, Any]:
        _enforce_manager_auth(request, manager_api_token)
        try:
            order = await run_in_threadpool(workshed_service.action_work_order, work_order_id, payload.action)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, "work_order": order}

    @ui_app.post("/api/manager/workshed/preflight")
    async def manager_workshed_preflight(
        request: Request,
        payload: WorkshedPreflightPayload,
    ) -> dict[str, Any]:
        _enforce_manager_auth(request, manager_api_token)
        return await run_in_threadpool(workshed_service.preflight, payload.model_dump())

    @ui_app.post("/api/manager/workshed/toolchains/prepare")
    async def manager_workshed_toolchains_prepare(
        request: Request,
        payload: WorkshedToolchainPreparePayload,
    ) -> dict[str, Any]:
        _enforce_manager_auth(request, manager_api_token)
        try:
            prepared = await run_in_threadpool(workshed_service.prepare_toolchains, payload.digests)
        except (OSError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": all(item.get("status") == "ready" for item in prepared), "toolchains": prepared}

    @ui_app.post("/api/manager/workshed/runs")
    async def manager_workshed_run_start(
        request: Request,
        payload: WorkshedRunPayload,
    ) -> dict[str, Any]:
        _enforce_manager_auth(request, manager_api_token)
        try:
            run = await run_in_threadpool(workshed_service.start, payload.model_dump())
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, "run": run}

    @ui_app.get("/api/manager/workshed/runs")
    async def manager_workshed_runs(request: Request) -> dict[str, Any]:
        _enforce_manager_auth(request, manager_api_token)
        return {"ok": True, "runs": await run_in_threadpool(workshed_service.list_runs)}

    @ui_app.get("/api/manager/workshed/runs/{run_id}")
    async def manager_workshed_run(
        request: Request,
        run_id: str,
        after_event_id: int = 0,
    ) -> dict[str, Any]:
        _enforce_manager_auth(request, manager_api_token)
        run = await run_in_threadpool(workshed_service.get_run, run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Workshed run not found.")
        cursor = max(0, int(after_event_id or 0))
        if cursor:
            run["events"] = [event for event in run.get("events", []) if int(event.get("id", 0)) > cursor]
        return {"ok": True, "run": run}

    @ui_app.post("/api/manager/workshed/runs/{run_id}/action")
    async def manager_workshed_run_action(
        request: Request,
        run_id: str,
        payload: WorkshedRunActionPayload,
    ) -> dict[str, Any]:
        _enforce_manager_auth(request, manager_api_token)
        try:
            run = await run_in_threadpool(workshed_service.action, run_id, payload.action)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, "run": run}

    @ui_app.get("/api/manager/workshed/artifacts")
    async def manager_workshed_artifacts(request: Request) -> dict[str, Any]:
        _enforce_manager_auth(request, manager_api_token)
        return {"ok": True, "artifacts": await run_in_threadpool(workshed_service.artifacts)}

    @ui_app.post("/api/manager/workshed/artifacts/{artifact_id:path}/action")
    async def manager_workshed_artifact_action(
        request: Request,
        artifact_id: str,
        payload: WorkshedArtifactActionPayload,
    ) -> dict[str, Any]:
        _enforce_manager_auth(request, manager_api_token)
        try:
            artifact = await run_in_threadpool(workshed_service.artifact_action, artifact_id, payload.action)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (PermissionError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, "artifact": artifact}

    @ui_app.get("/api/manager/hf-binding")
    async def manager_hf_binding_status(request: Request) -> dict[str, Any]:
        _enforce_manager_auth(request, manager_api_token)
        binding = _load_hf_binding_state()
        username = str(binding.get("username", "") or "").strip()
        token = str(binding.get("token", "") or "").strip()
        seen = bool(binding.get("seen", False))
        if username or token:
            seen = True
        return {
            "ok": True,
            "seen": seen,
            "username": username,
            "has_token": bool(token),
        }

    @ui_app.post("/api/manager/hf-binding")
    async def manager_hf_binding_save(
        request: Request,
        payload: HfBindingPayload,
    ) -> dict[str, Any]:
        _enforce_manager_auth(request, manager_api_token)
        binding = _save_hf_binding_state(
            payload.username, payload.token, seen=payload.seen
        )
        username = str(binding.get("username", "") or "").strip()
        token = str(binding.get("token", "") or "").strip()
        seen = bool(binding.get("seen", False))
        if username or token:
            seen = True
        return {
            "ok": True,
            "seen": seen,
            "username": username,
            "has_token": bool(token),
            "message": "Hugging Face binding saved.",
        }

    @ui_app.get("/api/manager/server-connection")
    async def manager_server_connection_get(request: Request) -> dict[str, Any]:
        _enforce_manager_auth(request, manager_api_token)
        controller = ui_app.state.server_connection_controller
        if controller is None:
            return {"ok": True, "mode": "roboserver", "connected": True}
        return controller.status()

    @ui_app.post("/api/manager/agent-data/local-export")
    async def manager_local_agent_data_export(
        request: Request, payload: AgentDataExportPayload
    ) -> dict[str, Any]:
        _enforce_manager_auth(request, manager_api_token)
        try:
            return ui_app.state.agent_store.export_transfer("local", payload.categories)
        except ValueError as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc

    @ui_app.post("/api/manager/agent-data/local-preview-import")
    async def manager_local_agent_data_preview_import(
        request: Request, payload: AgentDataTransferPayload
    ) -> dict[str, Any]:
        _enforce_manager_auth(request, manager_api_token)
        try:
            return ui_app.state.agent_store.preview_transfer_import(
                payload.transfer, payload.categories
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @ui_app.post("/api/manager/agent-data/local-import")
    async def manager_local_agent_data_import(
        request: Request, payload: AgentDataTransferPayload
    ) -> dict[str, Any]:
        _enforce_manager_auth(request, manager_api_token)
        try:
            return ui_app.state.agent_store.import_transfer(
                payload.transfer, payload.categories
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @ui_app.post("/api/manager/remote/inspect")
    async def manager_remote_inspect(
        request: Request,
        payload: RemoteServerTargetPayload,
    ) -> dict[str, Any]:
        _enforce_manager_auth(request, manager_api_token)
        controller = ui_app.state.server_connection_controller
        if controller is None:
            raise HTTPException(status_code=409, detail="Remote connection is managed by the NextCore desktop app.")
        try:
            target = RemoteTarget.from_payload(payload.model_dump())
            return await run_in_threadpool(controller.inspect_host, target)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @ui_app.post("/api/manager/remote/connect")
    async def manager_remote_connect(
        request: Request,
        payload: RemoteServerConnectPayload,
    ) -> dict[str, Any]:
        _enforce_manager_auth(request, manager_api_token)
        controller = ui_app.state.server_connection_controller
        if controller is None:
            raise HTTPException(status_code=409, detail="Remote connection is managed by the NextCore desktop app.")
        previous_model = await run_in_threadpool(manager.active_model)
        previous_was_running = await run_in_threadpool(manager.is_running)
        try:
            target = RemoteTarget.from_payload(payload.model_dump())
            # The stable local OpenAI-compatible inference port becomes the
            # loopback end of the SSH tunnel while remote mode is selected.
            await run_in_threadpool(manager.stop)
            result = await run_in_threadpool(controller.connect, target, payload.fingerprint)
        except (ValueError, RuntimeError) as exc:
            if previous_was_running and previous_model:
                await run_in_threadpool(manager.start, previous_model)
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        ui_app.state.server_url = result["server_url"]
        ui_app.state.keep_running_on_window_close = False
        return result

    @ui_app.post("/api/manager/remote/disconnect")
    async def manager_remote_disconnect(request: Request) -> dict[str, Any]:
        _enforce_manager_auth(request, manager_api_token)
        controller = ui_app.state.server_connection_controller
        if controller is None:
            raise HTTPException(status_code=409, detail="RoboServer cannot disconnect a remote target.")
        result = await run_in_threadpool(controller.disconnect)
        ui_app.state.server_url = server_url
        ui_app.state.keep_running_on_window_close = controller.local_server_enabled()
        return {**result, "server_url": server_url}

    @ui_app.get("/api/manager/local-server")
    async def manager_local_server_get(request: Request) -> dict[str, Any]:
        _enforce_manager_auth(request, manager_api_token)
        controller = ui_app.state.server_connection_controller
        return {"ok": True, "enabled": bool(controller and controller.local_server_enabled())}

    @ui_app.post("/api/manager/local-server")
    async def manager_local_server_set(
        request: Request,
        payload: ServerModePayload,
    ) -> dict[str, Any]:
        _enforce_manager_auth(request, manager_api_token)
        controller = ui_app.state.server_connection_controller
        if controller is None:
            raise HTTPException(status_code=409, detail="Local server mode is controlled by NextCore.")
        enabled = bool(payload.enabled)
        controller.set_local_server_enabled(enabled)
        ui_app.state.keep_running_on_window_close = enabled
        _save_server_mode_preference(enabled)
        return {"ok": True, "enabled": enabled, "mode": controller.mode()}

    @ui_app.get("/api/manager/server-mode")
    async def manager_server_mode_get(request: Request) -> dict[str, Any]:
        _enforce_manager_auth(request, manager_api_token)
        controller = ui_app.state.server_connection_controller
        enabled = bool(controller and controller.local_server_enabled())
        return {"ok": True, "enabled": enabled}

    @ui_app.post("/api/manager/server-mode")
    async def manager_server_mode_set(
        request: Request,
        payload: ServerModePayload,
    ) -> dict[str, Any]:
        _enforce_manager_auth(request, manager_api_token)
        controller = ui_app.state.server_connection_controller
        if controller is None:
            raise HTTPException(status_code=409, detail="Server mode is controlled by NextCore.")
        enabled = bool(payload.enabled)
        controller.set_local_server_enabled(enabled)
        ui_app.state.keep_running_on_window_close = enabled
        _save_server_mode_preference(enabled)
        return {"ok": True, "enabled": enabled, "mode": controller.mode()}

    @ui_app.post("/api/manager/stop")
    async def manager_stop_server(request: Request) -> dict[str, Any]:
        _enforce_manager_auth(request, manager_api_token)
        await run_in_threadpool(manager.stop)
        return {"ok": True, "message": "SGLang server stopped.", "running": False}

    @ui_app.get("/api/manager/concurrent-models")
    async def manager_concurrent_models_get(request: Request) -> dict[str, Any]:
        _enforce_manager_auth(request, manager_api_token)
        state_payload = await run_in_threadpool(manager.concurrent_model_state)
        return {
            "ok": True,
            **state_payload,
        }

    @ui_app.post("/api/manager/concurrent-models")
    async def manager_concurrent_models_set(
        request: Request,
        payload: ConcurrentModelsPayload,
    ) -> dict[str, Any]:
        _enforce_manager_auth(request, manager_api_token)
        ok, message = await run_in_threadpool(
            manager.set_concurrent_models, payload.models
        )
        state_payload = await run_in_threadpool(manager.concurrent_model_state)
        if not ok:
            raise HTTPException(
                status_code=500,
                detail=f"{message}",
            )
        return {
            "ok": True,
            "message": message,
            **state_payload,
        }

    @ui_app.get("/api/manager/runtime-config")
    async def manager_runtime_config(request: Request) -> dict[str, Any]:
        _enforce_manager_auth(request, manager_api_token)
        return {
            "ok": True,
            "runtime_config": await run_in_threadpool(manager.runtime_config),
            "running": await run_in_threadpool(manager.is_running),
            "active_model": await run_in_threadpool(manager.active_model),
            "active_server_url": await run_in_threadpool(manager.active_server_url),
        }

    @ui_app.post("/api/manager/runtime-config")
    async def manager_runtime_config_update(
        request: Request,
        payload: RuntimeConfigPayload,
    ) -> dict[str, Any]:
        _enforce_manager_auth(request, manager_api_token)
        current_config = await run_in_threadpool(manager.runtime_config)
        patch = payload.model_dump(exclude_none=True)
        next_config = {**current_config, **patch}

        ok, message = await run_in_threadpool(
            manager.update_runtime_config, next_config
        )
        if not ok:
            _developer_log(f"runtime config update failed: {message}")
            raise HTTPException(status_code=500, detail=message)
        _developer_log(f"runtime config updated: {message}")

        return {
            "ok": True,
            "message": message,
            "runtime_config": await run_in_threadpool(manager.runtime_config),
            "running": await run_in_threadpool(manager.is_running),
            "active_model": await run_in_threadpool(manager.active_model),
            "active_server_url": await run_in_threadpool(manager.active_server_url),
        }

    @ui_app.get("/api/manager/community/search")
    async def manager_community_search(
        request: Request,
        q: str = "",
        limit: int = 24,
        include_unready: bool = False,
    ) -> dict[str, Any]:
        _enforce_manager_auth(request, manager_api_token)
        try:
            results = await run_in_threadpool(
                manager.search_community_models,
                q,
                limit,
                include_unready=include_unready,
            )
            local_count = sum(1 for item in results if bool(item.get("local")))
            community_count = sum(
                1
                for item in results
                if str(item.get("source", "")).strip().lower() != "local"
            )
            return {
                "ok": True,
                "query": q,
                "count": len(results),
                "local_count": local_count,
                "community_count": community_count,
                "results": results,
            }
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @ui_app.get("/api/manager/community/job")
    async def manager_community_job(request: Request) -> dict[str, Any]:
        _enforce_manager_auth(request, manager_api_token)
        job = await run_in_threadpool(manager.community_job_status)
        return {
            "ok": True,
            "job": job,
            "active_model": await run_in_threadpool(manager.active_model),
        }

    @ui_app.post("/api/manager/community/download-deploy", include_in_schema=False)
    @ui_app.post("/api/manager/community/serve-verify", include_in_schema=False)
    @ui_app.post("/api/manager/community/download")
    async def manager_community_download(
        request: Request,
        payload: CommunityDownloadPayload,
    ) -> dict[str, Any]:
        _enforce_manager_auth(request, manager_api_token)
        model = payload.model.strip()
        if not model:
            raise HTTPException(status_code=400, detail="Model name cannot be empty.")

        ok, message, job = await run_in_threadpool(
            manager.start_community_model_download,
            model,
        )
        if not ok:
            raise HTTPException(status_code=409, detail=message)

        return {
            "ok": True,
            "message": message,
            "job": job,
        }

    @ui_app.post("/api/manager/community/job/action")
    async def manager_community_job_action(
        request: Request,
        payload: CommunityJobActionPayload,
    ) -> dict[str, Any]:
        _enforce_manager_auth(request, manager_api_token)
        ok, message, job = await run_in_threadpool(
            manager.control_community_job,
            payload.action,
        )
        if not ok:
            raise HTTPException(status_code=409, detail=message)
        return {
            "ok": True,
            "message": message,
            "job": job,
        }

    @ui_app.post("/api/manager/switch-model")
    async def manager_switch(
        request: Request,
        payload: SwitchModelPayload,
    ) -> dict[str, Any]:
        _enforce_manager_auth(request, manager_api_token)
        model = payload.model.strip()
        if not model:
            raise HTTPException(status_code=400, detail="Model name cannot be empty.")

        # Release lightweight title model cache before switching the main model.
        try:
            await run_in_threadpool(css_svg_ui._release_title_model)
        except Exception:
            pass

        ok, message = await run_in_threadpool(manager.switch_model, model)
        if not ok:
            _developer_log(f"switch model failed model={model}: {message}")
            raise HTTPException(status_code=500, detail=message)
        _developer_log(f"switch model ok model={model}: {message}")

        models = await run_in_threadpool(manager.discover_models)
        active = await run_in_threadpool(manager.active_model)
        if active and active not in models and not _is_hidden_model(active):
            models = [active, *models]

        model_capabilities = _model_capability_map(models)
        running_models = await run_in_threadpool(manager.running_models)
        runtime_config = await run_in_threadpool(manager.runtime_config)
        deep_thinking_models = [
            model_id
            for model_id, cap in model_capabilities.items()
            if bool(cap.get("deep_thinking"))
        ]

        return {
            "ok": True,
            "message": message,
            "active_model": active,
            "active_server_url": await run_in_threadpool(manager.active_server_url),
            "available_models": models,
            "model_capabilities": model_capabilities,
            "deep_thinking_models": deep_thinking_models,
            "model_details": _build_model_details(
                models,
                active_model=active,
                running_models=running_models,
                runtime_config=runtime_config,
            ),
            "running_models": running_models,
            "concurrent_models": await run_in_threadpool(manager.concurrent_models),
            "model_server_urls": await run_in_threadpool(manager.model_server_urls),
            "runtime_config": runtime_config,
            "active_reasoning_parser": await run_in_threadpool(
                manager.active_reasoning_parser
            ),
            "active_tool_call_parser": await run_in_threadpool(
                manager.active_tool_call_parser
            ),
        }

    @ui_app.post("/api/manager/delete-model")
    async def manager_delete_model(
        request: Request,
        payload: DeleteModelPayload,
    ) -> dict[str, Any]:
        _enforce_manager_auth(request, manager_api_token)
        model = payload.model.strip()
        if not model:
            raise HTTPException(status_code=400, detail="Model name cannot be empty.")

        ok, message, models = await run_in_threadpool(manager.delete_local_model, model)
        if not ok:
            raise HTTPException(status_code=409, detail=message)

        active = await run_in_threadpool(manager.active_model)
        model_capabilities = _model_capability_map(models)
        running_models = await run_in_threadpool(manager.running_models)
        runtime_config = await run_in_threadpool(manager.runtime_config)
        return {
            "ok": True,
            "message": message,
            "active_model": active,
            "available_models": models,
            "model_capabilities": model_capabilities,
            "model_details": _build_model_details(
                models,
                active_model=active,
                running_models=running_models,
                runtime_config=runtime_config,
            ),
            "running_models": running_models,
            "runtime_config": runtime_config,
            "integrity_report": await run_in_threadpool(manager.model_integrity_report),
        }

    @ui_app.post("/api/manager/model-integrity/cleanup")
    async def manager_model_integrity_cleanup(
        request: Request,
        payload: ModelCleanupPayload,
    ) -> dict[str, Any]:
        _enforce_manager_auth(request, manager_api_token)
        if not payload.confirm:
            raise HTTPException(
                status_code=400,
                detail="Cleanup requires explicit confirmation.",
            )
        report = await run_in_threadpool(
            manager.validate_and_cleanup_models,
            True,
            True,
        )
        models = await run_in_threadpool(manager.discover_models, False)
        return {
            "ok": True,
            "message": "Model cache cleanup completed.",
            "integrity_report": report,
            "available_models": models,
        }

    ui_config = uvicorn.Config(
        ui_app, host=args.ui_host, port=args.ui_port, log_level="warning"
    )
    ui_server = uvicorn.Server(ui_config)
    ui_thread = threading.Thread(target=ui_server.run, daemon=True)
    ui_thread.start()

    if not _wait_http_ok(f"{ui_url}/api/config", timeout_s=20):
        ui_server.should_exit = True
        terminal_session.stop()
        manager.stop()
        _fatal("UI server did not start correctly.")

    print("UI server is ready.")

    shutting_down = False
    tray_controller = None
    watchdog_stop = threading.Event()
    watchdog_thread: threading.Thread | None = None

    if not remote_selected and _is_env_enabled("TOKEN_WORKSHED_MODEL_WATCHDOG", True):
        watchdog_model_hint = str(initial_model or requested_model or "").strip()

        def _watchdog_loop() -> None:
            failure_count = 0
            while not watchdog_stop.wait(2.0):
                try:
                    ok, message = manager.ensure_primary_running(watchdog_model_hint)
                except Exception as exc:
                    failure_count += 1
                    if failure_count in {1, 4, 10}:
                        print(f"Watchdog error: {exc}")
                    continue

                if ok:
                    if message.startswith("Auto-recovered model"):
                        print(message)
                    failure_count = 0
                    continue

                # No model target is expected on fresh installs before first model run.
                if "No model target available" in message:
                    failure_count = 0
                    continue

                failure_count += 1
                if failure_count in {1, 4, 10}:
                    print(f"Watchdog retry pending: {message}")

        watchdog_thread = threading.Thread(
            target=_watchdog_loop,
            daemon=True,
            name="token-workshed-watchdog",
        )
        watchdog_thread.start()

    def shutdown() -> None:
        nonlocal shutting_down, tray_controller
        if shutting_down:
            return
        shutting_down = True

        print("\nShutting down...")
        watchdog_stop.set()
        remote_reconnect_stop.set()
        if tray_controller is not None:
            try:
                tray_controller.uninstall()
            except Exception:
                pass
            tray_controller = None
        ui_server.should_exit = True
        terminal_session.stop()
        manager.stop()
        if watchdog_thread is not None:
            watchdog_thread.join(timeout=2.5)
        if remote_reconnect_thread is not None:
            remote_reconnect_thread.join(timeout=2.5)
        ui_thread.join(timeout=4)
        _clear_background_state(expected_pid=os.getpid())

    def handle_signal(signum: int, _frame: object) -> None:
        print(f"\nReceived signal {signum}.")
        shutdown()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        if args.serve_only:
            print("Running in background serve-only mode. Press Ctrl+C to stop.")
            while True:
                time.sleep(0.8)
        if args.browser:
            print(
                "--browser is deprecated for the native UI rewrite; "
                "opening the local API status page only."
            )
            webbrowser.open(ui_url)

        native_ui_binary = _find_native_ui_binary()
        if native_ui_binary is None:
            candidates = _format_native_ui_candidates()
            _fatal(
                "Native Rust UI binary was not found.\n"
                "Build it with `cd native-ui && cargo build --release`, "
                "or set TOKEN_WORKSHED_NATIVE_UI_BIN.\n"
                f"Checked:\n{candidates}"
            )

        active_model = str(
            manager.active_model() or initial_model or requested_model
        ).strip()
        _save_background_state(
            pid=os.getpid(),
            server_url=server_url,
            ui_url=ui_url,
            model=active_model,
        )

        native_cmd = [
            str(native_ui_binary),
            "--api-base",
            ui_url,
            "--server-url",
            server_url,
            "--initial-page",
            str(args.initial_page or "chat").strip() or "chat",
        ]
        ide_companion_process = os.environ.get("TOKEN_WORKSHED_IDE_COMPANION", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if ide_companion_process:
            native_cmd.append("--ide-companion")
        if manager_api_token:
            native_cmd.extend(["--manager-token", manager_api_token])

        print(f"Launching native UI: {native_ui_binary}")
        native_env = os.environ.copy()
        if manager_api_token:
            native_env["TOKEN_WORKSHED_MANAGER_TOKEN"] = manager_api_token
        if not ide_companion_process:
            native_env["TOKEN_WORKSHED_AGENT_UI_TOKEN"] = agent_ui_token
        native_process = subprocess.Popen(native_cmd, env=native_env)
        return_code = native_process.wait()
        if return_code:
            print(f"Native UI exited with status {return_code}.")
    finally:
        shutdown()


if __name__ == "__main__":
    main()
