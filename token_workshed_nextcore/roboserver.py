# SPDX-License-Identifier: Apache-2.0
"""User-level RoboServer service and bootstrap commands for macOS."""

from __future__ import annotations

import argparse
import json
import os
import platform
import secrets
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROBO_SERVER_VERSION = "0.1.0"
MANAGER_PORT = 17872
INFERENCE_PORT = 18010
LAUNCH_AGENT_LABEL = "com.taylorhe.tokenworkshed.roboserver"


def data_dir() -> Path:
    configured = str(os.environ.get("TOKEN_WORKSHED_ROBOSERVER_DATA_DIR", "")).strip()
    if configured:
        root = Path(configured).expanduser()
    elif sys.platform == "darwin":
        root = Path.home() / "Library" / "Application Support" / "token-workshed-roboserver"
    else:
        root = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "token-workshed-roboserver"
    root.mkdir(parents=True, exist_ok=True)
    try:
        root.chmod(0o700)
    except OSError:
        pass
    return root


def _secret(name: str) -> str:
    path = data_dir() / f".{name}"
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError:
        value = ""
    if value:
        return value
    value = secrets.token_urlsafe(48)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return path.read_text(encoding="utf-8").strip()
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(value + "\n")
    try:
        path.chmod(0o600)
    except OSError:
        pass
    return value


def connection_credentials() -> dict[str, Any]:
    """Return credentials only for the SSH-authenticated bootstrap channel."""
    return {
        "ok": True,
        "manager_token": _secret("manager-token"),
        "agent_token": _secret("agent-token"),
        "manager_port": MANAGER_PORT,
        "inference_port": INFERENCE_PORT,
    }


def _source_root() -> Path:
    configured = str(os.environ.get("TOKEN_WORKSHED_NEXTCORE_SOURCE", "")).strip()
    if configured:
        return Path(configured).expanduser().resolve()
    package_root = Path(__file__).resolve().parent.parent
    if (package_root / "scripts" / "install_sglang_runtime.sh").is_file():
        return package_root
    resource_root = Path(sys.argv[0]).resolve().parent.parent / "Resources" / "source"
    return resource_root


def _runtime_python() -> Path:
    configured = str(os.environ.get("TOKEN_WORKSHED_ROBOSERVER_PYTHON", "")).strip()
    if configured:
        return Path(configured).expanduser()
    local_venv = _source_root() / ".venv" / "bin" / "python"
    if local_venv.is_file():
        try:
            result = subprocess.run(
                [str(local_venv), "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.stdout.strip() == "3.12":
                return local_venv
        except (OSError, subprocess.SubprocessError):
            pass
    return data_dir() / "runtime" / ".venv" / "bin" / "python3"


def _valid_runtime(python: Path) -> bool:
    try:
        result = subprocess.run(
            [
                str(python),
                "-c",
                "from importlib.metadata import version; import sys; print('{}.{} {}'.format(sys.version_info.major, sys.version_info.minor, version('sglang')))"
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=8,
        )
        return result.stdout.strip() == "3.12 0.5.22"
    except (OSError, subprocess.SubprocessError):
        return False


def ensure_runtime() -> Path:
    """Install the pinned user-level Python/SGLang runtime when absent."""
    python = _runtime_python()
    if _valid_runtime(python):
        return python
    if sys.platform != "darwin" or platform.machine().lower() not in {"arm64", "aarch64"}:
        raise RuntimeError("RoboServer v0.1 supports Apple Silicon macOS only.")
    source = _source_root()
    installer = source / "scripts" / "install_sglang_runtime.sh"
    if not installer.is_file():
        raise RuntimeError(f"RoboServer runtime installer is missing: {installer}")
    venv_root = python.parent.parent
    venv_root.parent.mkdir(parents=True, exist_ok=True)
    base_python = str(os.environ.get("TOKEN_WORKSHED_BASE_PYTHON", "")).strip() or shutil.which("python3.12")
    if not base_python:
        raise RuntimeError("Python 3.12 is required to install RoboServer Runtime. Install Python 3.12 or uv, then start the server again.")
    if not python.exists():
        subprocess.run([base_python, "-m", "venv", str(venv_root)], check=True, timeout=120)
    env = os.environ.copy()
    env["TOKEN_WORKSHED_NEXTCORE_PYTHON"] = str(python)
    env["TOKEN_WORKSHED_NEXTCORE_SGLANG_ROOT"] = str(data_dir() / "sglang-v0.5.22")
    env["TOKEN_WORKSHED_NEXTCORE_SOURCE"] = str(source)
    # Install the app dependencies first, then restore the exact SGLang pin.
    subprocess.run([str(python), "-m", "pip", "install", "-e", str(source)], check=True, env=env, timeout=1800)
    subprocess.run(["/bin/bash", str(installer)], check=True, env=env, timeout=1800)
    if not _valid_runtime(python):
        raise RuntimeError("RoboServer runtime installation did not produce Python 3.12 + SGLang 0.5.22.")
    return python


def _launch_agent_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{LAUNCH_AGENT_LABEL}.plist"


def _runner_path() -> Path:
    configured = str(os.environ.get("TOKEN_WORKSHED_ROBOSERVER_RUNNER", "")).strip()
    if configured:
        return Path(configured).expanduser().resolve()
    root = Path(__file__).resolve().parent.parent
    bundle_candidate = root / "dist" / "token-workshed-roboserver.app" / "Contents" / "Resources" / "roboserver-runner"
    if bundle_candidate.is_file():
        return bundle_candidate
    source_candidate = root / "scripts" / "roboserver-runner"
    if source_candidate.is_file():
        return source_candidate
    raise RuntimeError("RoboServer service runner is missing; reinstall the RoboServer App.")


def install_launch_agent() -> dict[str, Any]:
    if sys.platform != "darwin" or platform.machine().lower() not in {"arm64", "aarch64"}:
        raise RuntimeError("RoboServer is currently supported only on Apple Silicon macOS.")
    runner = _runner_path()
    if not runner.is_file():
        raise RuntimeError(f"RoboServer service runner was not found: {runner}")
    root = data_dir()
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    plist_path = _launch_agent_path()
    plist_path.parent.mkdir(parents=True, exist_ok=True)
    environment = {
        "TOKEN_WORKSHED_ROBOSERVER_DATA_DIR": str(root),
        "TOKEN_WORKSHED_ROBOSERVER_RUNNER": str(runner),
        "TOKEN_WORKSHED_ROBOSERVER_BOOTSTRAPPED": "1",
        "TOKEN_WORKSHED_ROBOSERVER_MODE": "1",
        "TOKEN_WORKSHED_MANAGER_TOKEN": _secret("manager-token"),
        "TOKEN_WORKSHED_AGENT_UI_TOKEN": _secret("agent-token"),
    }
    if str(os.environ.get("TOKEN_WORKSHED_NEXTCORE_SOURCE", "")).strip():
        environment["TOKEN_WORKSHED_NEXTCORE_SOURCE"] = str(_source_root())
    plist: dict[str, Any] = {
        "Label": LAUNCH_AGENT_LABEL,
        "ProgramArguments": [str(runner), "--serve"],
        "RunAtLoad": True,
        "KeepAlive": True,
        "EnvironmentVariables": environment,
        "StandardOutPath": str(logs / "roboserver.log"),
        "StandardErrorPath": str(logs / "roboserver.log"),
    }
    try:
        from plistlib import dump

        with plist_path.open("wb") as handle:
            dump(plist, handle, sort_keys=True)
        plist_path.chmod(0o600)
    except OSError as exc:
        raise RuntimeError(f"Could not write RoboServer LaunchAgent: {exc}") from exc
    uid = os.getuid()
    subprocess.run(["/bin/launchctl", "bootout", f"gui/{uid}/{LAUNCH_AGENT_LABEL}"], capture_output=True, timeout=10)
    subprocess.run(["/bin/launchctl", "bootstrap", f"gui/{uid}", str(plist_path)], check=True, timeout=20)
    subprocess.run(["/bin/launchctl", "kickstart", "-k", f"gui/{uid}/{LAUNCH_AGENT_LABEL}"], check=True, timeout=20)
    return {"ok": True, "label": LAUNCH_AGENT_LABEL, "plist": str(plist_path)}


def stop_launch_agent() -> dict[str, Any]:
    if sys.platform != "darwin":
        raise RuntimeError("RoboServer service control currently requires macOS.")
    try:
        import requests

        requests.post(
            f"http://127.0.0.1:{MANAGER_PORT}/api/manager/stop",
            headers={"X-Token-Workshed-Manager-Token": _secret("manager-token")},
            timeout=(0.8, 5),
        )
    except Exception:
        # The service may already be down; launchctl below remains authoritative.
        pass
    uid = os.getuid()
    result = subprocess.run(
        ["/bin/launchctl", "bootout", f"gui/{uid}/{LAUNCH_AGENT_LABEL}"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    return {"ok": result.returncode == 0, "label": LAUNCH_AGENT_LABEL}


def service_status() -> dict[str, Any]:
    try:
        import socket

        with socket.create_connection(("127.0.0.1", MANAGER_PORT), timeout=0.5):
            api_ready = True
    except OSError:
        api_ready = False
    result = subprocess.run(
        ["/bin/launchctl", "print", f"gui/{os.getuid()}/{LAUNCH_AGENT_LABEL}"],
        capture_output=True,
        text=True,
        timeout=5,
    )
    return {
        "ok": True,
        "service_loaded": result.returncode == 0,
        "api_ready": api_ready,
        "manager_port": MANAGER_PORT,
        "inference_port": INFERENCE_PORT,
    }


def _serve() -> None:
    ensure_runtime()
    source = _source_root()
    os.environ["PYTHONPATH"] = str(source) + os.pathsep + os.environ.get("PYTHONPATH", "")
    os.environ["TOKEN_WORKSHED_ROBOSERVER_MODE"] = "1"
    os.environ["TOKEN_WORKSHED_ROBOSERVER_DATA_DIR"] = str(data_dir())
    os.environ["TOKEN_WORKSHED_MANAGER_TOKEN"] = _secret("manager-token")
    os.environ["TOKEN_WORKSHED_AGENT_UI_TOKEN"] = _secret("agent-token")
    os.environ["TOKEN_WORKSHED_AGENT_DATA_DIR"] = str(data_dir() / "agent-data")
    os.environ["TOKEN_WORKSHED_DESKTOP_STATE_PATH"] = str(data_dir() / "desktop-state.json")
    os.environ["HF_HOME"] = str(data_dir() / "models" / "huggingface")
    os.environ["TOKEN_WORKSHED_ROBOSERVER_BOOTSTRAPPED"] = "1"
    sys.path.insert(0, str(source))
    sys.argv = [
        "token-workshed-roboserver-service",
        "--roboserver",
        "--serve-only",
        "--no-model-autostart",
        "--server-host",
        "127.0.0.1",
        "--server-port",
        str(INFERENCE_PORT),
        "--ui-host",
        "127.0.0.1",
        "--ui-port",
        str(MANAGER_PORT),
    ]
    from .desktop_ui import main

    main()


def main() -> None:
    parser = argparse.ArgumentParser(description="Token Workshed RoboServer service")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--serve", action="store_true")
    group.add_argument("--ensure-service", action="store_true")
    group.add_argument("--stop-service", action="store_true")
    group.add_argument("--status", action="store_true")
    group.add_argument("--connection-credentials", action="store_true")
    group.add_argument("--bootstrap", action="store_true")
    args = parser.parse_args()
    try:
        if args.connection_credentials:
            print(json.dumps(connection_credentials(), separators=(",", ":")))
        elif args.status:
            print(json.dumps(service_status(), separators=(",", ":")))
        elif args.stop_service:
            print(json.dumps(stop_launch_agent(), separators=(",", ":")))
        elif args.ensure_service:
            ensure_runtime()
            print(json.dumps(install_launch_agent(), separators=(",", ":")))
        elif args.bootstrap:
            python = ensure_runtime()
            print(json.dumps({"ok": True, "python": str(python), "version": "0.5.22"}, separators=(",", ":")))
        elif args.serve:
            if not _valid_runtime(Path(sys.executable)):
                python = ensure_runtime()
                os.environ["TOKEN_WORKSHED_ROBOSERVER_BOOTSTRAPPED"] = "1"
                os.execv(str(python), [str(python), "-m", "token_workshed_nextcore.roboserver", "--serve"])
            _serve()
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, separators=(",", ":")), file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
