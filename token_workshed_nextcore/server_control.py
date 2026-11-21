# SPDX-License-Identifier: Apache-2.0
"""Local server-mode state and pinned SSH tunnels to RoboServer hosts."""

from __future__ import annotations

import json
import os
import re
import shlex
import socket
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import requests

from .desktop_state import desktop_state_file, mutate_desktop_state, read_desktop_state

_SAFE_HOST = re.compile(r"^[A-Za-z0-9._:-]{1,253}$")
_SAFE_USER = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_FINGERPRINT = re.compile(r"^SHA256:[A-Za-z0-9+/]{20,}={0,2}$")
_REMOTE_API_PATHS = ("/api/manager/", "/api/agent/")


@dataclass(frozen=True)
class RemoteTarget:
    host: str
    username: str
    port: int = 22
    identity_file: str = ""
    app_path: str = "~/Applications/token-workshed-roboserver.app"

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "RemoteTarget":
        host = str(payload.get("host", "")).strip()
        username = str(payload.get("username", "")).strip()
        try:
            port = int(payload.get("port", 22))
        except (TypeError, ValueError) as exc:
            raise ValueError("SSH port must be an integer between 1 and 65535.") from exc
        identity_file = str(payload.get("identity_file", "")).strip()
        app_path = str(payload.get("app_path", "~/Applications/token-workshed-roboserver.app")).strip()
        if not _SAFE_HOST.fullmatch(host) or host.startswith("-"):
            raise ValueError("Enter a valid SSH host name or address.")
        if not _SAFE_USER.fullmatch(username):
            raise ValueError("Enter a valid SSH username.")
        if not 1 <= port <= 65535:
            raise ValueError("SSH port must be between 1 and 65535.")
        if identity_file and ("\n" in identity_file or "\r" in identity_file):
            raise ValueError("SSH identity file path is invalid.")
        if not app_path.startswith("~/") and not app_path.startswith("/"):
            raise ValueError("RoboServer App path must be absolute or start with ~/.")
        if any(char in app_path for char in "\n\r\x00"):
            raise ValueError("RoboServer App path is invalid.")
        return cls(host, username, port, identity_file, app_path)

    @property
    def destination(self) -> str:
        host = f"[{self.host}]" if ":" in self.host and not self.host.startswith("[") else self.host
        return f"{self.username}@{host}"

    @property
    def known_hosts_name(self) -> str:
        return self.host if self.port == 22 else f"[{self.host}]:{self.port}"

    def public_payload(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "username": self.username,
            "port": self.port,
            "identity_file": self.identity_file,
            "app_path": self.app_path,
        }


class ServerConnectionController:
    """Owns local server mode and the SSH tunnel for one selected RoboServer."""

    def __init__(self, app_root: Path | None = None, *, local_inference_port: int = 8010) -> None:
        self.app_root = Path(app_root or Path(__file__).resolve().parent.parent)
        if not 1 <= int(local_inference_port) <= 65535:
            raise ValueError("Local inference port must be between 1 and 65535.")
        self.local_inference_port = int(local_inference_port)
        state_path = desktop_state_file()
        self.private_root = state_path.parent / "ssh"
        self.private_root.mkdir(parents=True, exist_ok=True)
        try:
            self.private_root.chmod(0o700)
        except OSError:
            pass
        self.known_hosts_path = self.private_root / "roboserver_known_hosts"
        self._lock = threading.RLock()
        self._tunnel: subprocess.Popen[bytes] | None = None
        self._target: RemoteTarget | None = None
        self._manager_token = ""
        self._agent_token = ""
        self._manager_port = 0
        self._inference_port = 0
        self._last_error = ""

    @staticmethod
    def _server_config() -> dict[str, Any]:
        payload = read_desktop_state().get("server_connection", {})
        return payload if isinstance(payload, dict) else {}

    def mode(self) -> str:
        return str(self._server_config().get("mode", "local") or "local")

    def local_server_enabled(self) -> bool:
        return bool(self._server_config().get("local_server_enabled", False))

    def set_local_server_enabled(self, enabled: bool) -> bool:
        def update(state: dict[str, Any]) -> bool:
            connection = state.get("server_connection")
            if not isinstance(connection, dict):
                connection = {}
            connection["local_server_enabled"] = bool(enabled)
            state["server_connection"] = connection
            return True

        mutate_desktop_state(update)
        return bool(enabled)

    def inspect_host(self, target: RemoteTarget) -> dict[str, Any]:
        result = subprocess.run(
            ["/usr/bin/ssh-keyscan", "-T", "6", "-p", str(target.port), target.host],
            capture_output=True,
            text=True,
            timeout=10,
        )
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip() and not line.startswith("#")]
        candidates: list[dict[str, str]] = []
        for line in lines:
            fields = line.split()
            if len(fields) < 3:
                continue
            normalized_line = f"{target.known_hosts_name} {fields[1]} {fields[2]}"
            fingerprint = self._fingerprint(normalized_line)
            if fingerprint:
                candidates.append({"algorithm": fields[1], "fingerprint": fingerprint, "known_hosts_line": normalized_line})
        if not candidates:
            detail = result.stderr.strip() or "ssh-keyscan did not return a host key."
            raise RuntimeError(f"Could not inspect SSH host key for {target.host}: {detail[:500]}")
        unique = {item["fingerprint"]: item for item in candidates}
        known = self._known_fingerprints(target)
        return {
            "ok": True,
            "target": target.public_payload(),
            "fingerprints": list(unique.values()),
            "already_trusted": bool(known and any(value in known for value in unique)),
            "changed_key": bool(known and not any(value in known for value in unique)),
        }

    @staticmethod
    def _fingerprint(known_hosts_line: str) -> str:
        result = subprocess.run(
            ["/usr/bin/ssh-keygen", "-lf", "/dev/stdin", "-E", "sha256"],
            input=known_hosts_line + "\n",
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode:
            return ""
        for field in result.stdout.split():
            if field.startswith("SHA256:"):
                return field
        return ""

    def _known_fingerprints(self, target: RemoteTarget) -> set[str]:
        if not self.known_hosts_path.exists():
            return set()
        found: set[str] = set()
        for line in self.known_hosts_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            fields = line.split()
            if len(fields) >= 3 and fields[0] == target.known_hosts_name:
                fingerprint = self._fingerprint(line)
                if fingerprint:
                    found.add(fingerprint)
        return found

    def _accept_host_key(self, target: RemoteTarget, fingerprint: str) -> str:
        if not _FINGERPRINT.fullmatch(fingerprint):
            raise ValueError("Confirm the complete SHA-256 host-key fingerprint shown by NextCore.")
        inspected = self.inspect_host(target)
        match = next((item for item in inspected["fingerprints"] if item["fingerprint"] == fingerprint), None)
        if not match:
            raise RuntimeError("The SSH host key changed after inspection. Inspect it again before connecting.")
        known = self._known_fingerprints(target)
        if known and fingerprint not in known:
            raise RuntimeError("This server's SSH host key differs from the previously trusted key. Remove the old pin deliberately before trusting a replacement.")
        if fingerprint not in known:
            with self.known_hosts_path.open("a", encoding="utf-8") as handle:
                handle.write(match["known_hosts_line"] + "\n")
            try:
                self.known_hosts_path.chmod(0o600)
            except OSError:
                pass
        return fingerprint

    def _ssh_base(self, target: RemoteTarget) -> list[str]:
        args = [
            "/usr/bin/ssh",
            "-p", str(target.port),
            "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=8",
            "-o", "ServerAliveInterval=15",
            "-o", "ServerAliveCountMax=3",
            "-o", "StrictHostKeyChecking=yes",
            "-o", f"UserKnownHostsFile={self.known_hosts_path}",
            "-o", "ExitOnForwardFailure=yes",
        ]
        if target.identity_file:
            args.extend(["-i", str(Path(target.identity_file).expanduser())])
            args.extend(["-o", "IdentitiesOnly=yes"])
        args.append(target.destination)
        return args

    @staticmethod
    def _free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
            handle.bind(("127.0.0.1", 0))
            return int(handle.getsockname()[1])

    def _run_remote(self, target: RemoteTarget, arguments: list[str], timeout: float = 60) -> subprocess.CompletedProcess[str]:
        runner = f"{target.app_path}/Contents/Resources/roboserver-runner"
        command = " ".join([self._quote_remote_path(runner), *(shlex.quote(value) for value in arguments)])
        try:
            return subprocess.run(
                [*self._ssh_base(target), command],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("SSH operation timed out. Check the host, SSH key, and remote RoboServer installation.") from exc

    @staticmethod
    def _quote_remote_path(path: str) -> str:
        if path.startswith("~/"):
            return '"$HOME"/' + shlex.quote(path[2:])
        return shlex.quote(path)

    def _deploy_if_missing(self, target: RemoteTarget) -> None:
        runner = f"{target.app_path}/Contents/Resources/roboserver-runner"
        check = subprocess.run(
            [*self._ssh_base(target), f"test -x {self._quote_remote_path(runner)}"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if check.returncode == 0:
            return
        running_bundle = str(os.environ.get("TOKEN_WORKSHED_NEXTCORE_APP_BUNDLE", "")).strip()
        bundled_robo_app = (
            Path(running_bundle).expanduser()
            / "Contents"
            / "Resources"
            / "token-workshed-roboserver.app"
            if running_bundle
            else None
        )
        bundle = (
            bundled_robo_app
            if bundled_robo_app is not None and bundled_robo_app.is_dir()
            else self.app_root / "dist" / "token-workshed-roboserver.app"
        )
        if not bundle.is_dir():
            raise RuntimeError("RoboServer App is not installed on the remote Mac and the local RoboServer bundle has not been built yet.")
        parent = target.app_path.rsplit("/", 1)[0]
        create = subprocess.run(
            [*self._ssh_base(target), f"mkdir -p {self._quote_remote_path(parent)} && test ! -e {self._quote_remote_path(target.app_path)}"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if create.returncode:
            raise RuntimeError("A RoboServer bundle already exists but is incomplete. Inspect it on the remote Mac before replacing it.")
        copy_args = [
            "/usr/bin/scp",
            "-r",
            "-P",
            str(target.port),
            "-o",
            "BatchMode=yes",
            "-o",
            f"UserKnownHostsFile={self.known_hosts_path}",
            "-o",
            "StrictHostKeyChecking=yes",
        ]
        if target.identity_file:
            copy_args.extend(["-i", str(Path(target.identity_file).expanduser()), "-o", "IdentitiesOnly=yes"])
        copy_args.extend([str(bundle), f"{target.destination}:{parent}/"])
        copy = subprocess.run(
            copy_args,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if copy.returncode:
            raise RuntimeError(f"Could not install RoboServer on the remote Mac: {(copy.stderr or copy.stdout).strip()[:700]}")

    def connect(
        self, target: RemoteTarget, fingerprint: str, *, ensure_service: bool = True
    ) -> dict[str, Any]:
        with self._lock:
            self.disconnect(persist=False)
            self._accept_host_key(target, fingerprint)
            self._deploy_if_missing(target)
            if ensure_service:
                ensure = self._run_remote(target, ["--ensure-service"], timeout=2400)
                if ensure.returncode:
                    detail = (ensure.stderr or ensure.stdout).strip()
                    raise RuntimeError(f"Could not start RoboServer: {detail[-1200:] or 'remote service command failed'}")
            credentials = self._run_remote(target, ["--connection-credentials"], timeout=30)
            if credentials.returncode:
                raise RuntimeError(f"Could not authenticate to RoboServer: {(credentials.stderr or credentials.stdout).strip()[:800]}")
            try:
                secret = json.loads(credentials.stdout)
                manager_token = str(secret["manager_token"])
                agent_token = str(secret["agent_token"])
                remote_manager_port = int(secret.get("manager_port", 17872))
                remote_inference_port = int(secret.get("inference_port", 18010))
            except (ValueError, TypeError, KeyError) as exc:
                raise RuntimeError("RoboServer returned invalid connection credentials.") from exc
            manager_port = self._free_port()
            inference_port = self.local_inference_port
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                    probe.bind(("127.0.0.1", inference_port))
            except OSError as exc:
                raise RuntimeError(
                    f"Local inference port {inference_port} is already in use. Stop the local model server before connecting remotely."
                ) from exc
            command = [
                *self._ssh_base(target),
                "-N",
                "-L", f"127.0.0.1:{manager_port}:127.0.0.1:{remote_manager_port}",
                "-L", f"127.0.0.1:{inference_port}:127.0.0.1:{remote_inference_port}",
            ]
            tunnel = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, start_new_session=True)
            deadline = time.monotonic() + 10
            ready = False
            while time.monotonic() < deadline:
                if tunnel.poll() is not None:
                    detail = ""
                    if tunnel.stderr:
                        detail = tunnel.stderr.read(1000).decode("utf-8", errors="replace")
                    raise RuntimeError(f"SSH tunnel stopped during setup. {detail.strip()}")
                try:
                    response = requests.get(
                        f"http://127.0.0.1:{manager_port}/api/config",
                        headers={"X-Token-Workshed-Manager-Token": manager_token},
                        timeout=1.5,
                    )
                    if response.ok:
                        ready = True
                        break
                except requests.RequestException:
                    pass
                time.sleep(0.2)
            if not ready:
                tunnel.terminate()
                try:
                    tunnel.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    tunnel.kill()
                raise RuntimeError("SSH tunnel opened, but RoboServer's private manager API did not become ready.")
            self._target = target
            self._manager_token = manager_token
            self._agent_token = agent_token
            self._manager_port = manager_port
            self._inference_port = inference_port
            self._tunnel = tunnel
            self._last_error = ""
            self._save_target(target, fingerprint)
            return self.status()

    def restore(self) -> dict[str, Any]:
        config = self._server_config()
        target_data = config.get("target")
        fingerprint = str(config.get("fingerprint", ""))
        if config.get("mode") != "remote" or not isinstance(target_data, dict) or not fingerprint:
            return self.status()
        try:
            # Reopen tunnels after transient SSH drops without overriding an
            # intentional Stop from the RoboServer Dashboard.
            return self.connect(
                RemoteTarget.from_payload(target_data), fingerprint, ensure_service=False
            )
        except Exception as exc:
            with self._lock:
                self._last_error = str(exc)
            return self.status()

    def _save_target(self, target: RemoteTarget, fingerprint: str) -> None:
        def update(state: dict[str, Any]) -> bool:
            connection = state.get("server_connection")
            if not isinstance(connection, dict):
                connection = {}
            connection.update({
                "mode": "remote",
                "target": target.public_payload(),
                "fingerprint": fingerprint,
            })
            # The local service's keep-running preference is independent from
            # which runtime currently receives inference requests.
            connection.setdefault("local_server_enabled", False)
            state["server_connection"] = connection
            return True

        mutate_desktop_state(update)

    def disconnect(self, *, persist: bool = True) -> dict[str, Any]:
        with self._lock:
            tunnel = self._tunnel
            self._tunnel = None
            self._target = None
            self._manager_token = ""
            self._agent_token = ""
            self._manager_port = 0
            self._inference_port = 0
            if tunnel is not None and tunnel.poll() is None:
                tunnel.terminate()
                try:
                    tunnel.wait(timeout=4)
                except subprocess.TimeoutExpired:
                    tunnel.kill()
            if persist:
                def update(state: dict[str, Any]) -> bool:
                    config = state.get("server_connection")
                    if not isinstance(config, dict):
                        config = {}
                    config["mode"] = "local"
                    state["server_connection"] = config
                    return True
                mutate_desktop_state(update)
            return self.status()

    def status(self) -> dict[str, Any]:
        with self._lock:
            connected = bool(self._tunnel and self._tunnel.poll() is None and self._manager_port)
            saved = self._server_config()
            target = self._target.public_payload() if self._target else saved.get("target", {})
            return {
                "ok": True,
                "mode": "remote" if connected else ("remote-disconnected" if saved.get("mode") == "remote" else "local"),
                "local_server_enabled": bool(saved.get("local_server_enabled", False)),
                "connected": connected,
                "target": target if isinstance(target, dict) else {},
                "fingerprint": str(saved.get("fingerprint", "")),
                "server_url": f"http://127.0.0.1:{self._inference_port}" if connected else "",
                "manager_url": f"http://127.0.0.1:{self._manager_port}" if connected else "",
                "last_error": self._last_error,
            }

    def should_proxy(self, path: str, *, agent_session_valid: bool = False) -> bool:
        if not self.status()["connected"]:
            return False
        if path == "/api/status":
            return True
        if path.startswith(_REMOTE_API_PATHS[0]):
            return not path.startswith((
                "/api/manager/remote/",
                "/api/manager/server-connection",
                "/api/manager/server-mode",
                "/api/manager/local-server",
                "/api/manager/agent-data/local-",
            ))
        if path.startswith(_REMOTE_API_PATHS[1]):
            return agent_session_valid
        if path in {"/api/chat/stream", "/api/chat/cancel"}:
            return agent_session_valid
        return False

    def proxy_url(self, path: str, query: str = "") -> str:
        with self._lock:
            if not self._manager_port:
                raise RuntimeError("Remote RoboServer is not connected.")
            return f"http://127.0.0.1:{self._manager_port}{path}{'?' + query if query else ''}"

    def proxy_headers(self, *, agent_session_valid: bool) -> dict[str, str]:
        with self._lock:
            token = self._agent_token if agent_session_valid else self._manager_token
            header = "X-Token-Workshed-App-Session" if agent_session_valid else "X-Token-Workshed-Manager-Token"
            return {header: token} if token else {}

    def remote_inference_url(self) -> str:
        with self._lock:
            return f"http://127.0.0.1:{self._inference_port}" if self._inference_port else ""


def safe_inference_url(value: str) -> bool:
    try:
        parsed = urlsplit(str(value or ""))
        _ = parsed.port
    except ValueError:
        return False
    return parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost", "::1"}
