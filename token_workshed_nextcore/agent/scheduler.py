"""Login-persistent, user-scoped scheduler for NextCore Agent tasks."""

from __future__ import annotations

import argparse
import json
import ipaddress
import logging
import os
import plistlib
import signal
import socket
import socketserver
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

import requests

from ..sglang_runtime import server_command
from .mcp_runtime import MCPRuntime
from .runtime import AgentCancelledError, SGLangAgent
from .schedule import next_run_time
from .store import AgentStateStore

logger = logging.getLogger("token_workshed_nextcore.agent.scheduler")
LABEL = "com.taylorhe.tokenworkshed.nextcore.agent"


def _launch_agent_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"


def _application_runner() -> Path | None:
    helper = Path(os.environ.get("TOKEN_WORKSHED_NATIVE_UI_BIN", "")).expanduser()
    if helper.name == "token-workshed-native-ui" and "NextCore UI.app" in str(helper):
        app = helper.parents[5]
        runner = app / "Contents" / "MacOS" / "nextcore-runner"
        if runner.exists():
            return runner
    return None


def send_scheduler_command(data_dir: Path, command: str, *, job_id: str = "", profile_id: str = "") -> dict:
    """Send a bounded command over the scheduler's user-private Unix socket."""
    socket_path = Path(data_dir).expanduser() / "agent" / "scheduler.sock"
    request = {"command": str(command)[:40]}
    if job_id:
        request["job_id"] = str(job_id)[:80]
    if profile_id:
        request["profile_id"] = str(profile_id)[:100]
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout(3)
    try:
        client.connect(str(socket_path))
        client.sendall((json.dumps(request) + "\n").encode("utf-8"))
        response = bytearray()
        while len(response) < 4096:
            chunk = client.recv(1024)
            if not chunk or b"\n" in chunk:
                response.extend(chunk)
                break
            response.extend(chunk)
        result = json.loads(bytes(response).split(b"\n", 1)[0] or b"{}")
        if not isinstance(result, dict):
            raise RuntimeError("Scheduler returned an invalid response.")
        return result
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("The NextCore background scheduler is not available.") from exc
    finally:
        client.close()


def install_launch_agent(data_dir: Path) -> dict[str, str]:
    """Install/reload the per-user scheduler LaunchAgent."""
    if sys.platform != "darwin":
        raise RuntimeError("NextCore scheduled background tasks require macOS.")
    data_dir = Path(data_dir).expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    path = _launch_agent_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    runner = _application_runner()
    if runner:
        program = ["/bin/bash", str(runner), "--agent-scheduler", "--data-dir", str(data_dir)]
        environment = {"TOKEN_WORKSHED_APP_BUNDLE_ID": os.environ.get("TOKEN_WORKSHED_APP_BUNDLE_ID", "com.taylorhe.tokenworkshed.nextcore"), "TOKEN_WORKSHED_AGENT_DATA_DIR": str(data_dir)}
    else:
        program = [sys.executable, "-m", "token_workshed_nextcore.agent.scheduler", "--data-dir", str(data_dir)]
        source_root = str(Path(__file__).resolve().parents[2])
        environment = {"PYTHONPATH": source_root, "TOKEN_WORKSHED_AGENT_DATA_DIR": str(data_dir)}
    payload = {
        "Label": LABEL,
        "ProgramArguments": program,
        "RunAtLoad": True,
        "KeepAlive": True,
        "ThrottleInterval": 15,
        "ProcessType": "Background",
        "EnvironmentVariables": environment,
        "StandardOutPath": str(data_dir / "agent" / "scheduler.log"),
        "StandardErrorPath": str(data_dir / "agent" / "scheduler.log"),
    }
    try:
        current_payload = plistlib.loads(path.read_bytes())
    except (OSError, plistlib.InvalidFileException, ValueError):
        current_payload = None
    temporary = path.with_suffix(".plist.tmp")
    with temporary.open("wb") as handle:
        plistlib.dump(payload, handle)
    os.chmod(temporary, 0o600)
    temporary.replace(path)
    domain = f"gui/{os.getuid()}"
    if current_payload == payload:
        status = subprocess.run(
            ["/bin/launchctl", "print", f"{domain}/{LABEL}"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if status.returncode == 0:
            return {"label": LABEL, "state": "loaded", "plist": str(path)}
    subprocess.run(["/bin/launchctl", "bootout", f"{domain}/{LABEL}"], capture_output=True, text=True, timeout=10, check=False)
    loaded = subprocess.run(["/bin/launchctl", "bootstrap", domain, str(path)], capture_output=True, text=True, timeout=10, check=False)
    if loaded.returncode:
        raise RuntimeError((loaded.stderr or loaded.stdout or "launchctl bootstrap failed").strip()[:500])
    return {"label": LABEL, "state": "loaded", "plist": str(path)}


class _PrivateSocketHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        try:
            payload = json.loads(self.rfile.readline(4096) or b"{}")
        except (json.JSONDecodeError, UnicodeDecodeError):
            payload = {}
        server = self.server
        command = payload.get("command") if isinstance(payload, dict) else ""
        response = {"ok": True, "state": "running"}
        if command == "stop":
            response["state"] = "stopping"
            server.stop_event.set()
            server.cancel_running_jobs()
        elif command == "cancel_job":
            job_id = str(payload.get("job_id") or "")[:80]
            cancelled = server.cancel_job(job_id)
            response = {"ok": cancelled, "state": "cancelling" if cancelled else "not_running"}
        elif command == "revoke_browser_profile":
            profile_id = str(payload.get("profile_id") or "")[:100]
            cancelled = server.revoke_browser_profile(profile_id)
            response = {"ok": True, "cancelled_jobs": cancelled}
        elif command != "status":
            response = {"ok": False, "error": "unsupported command"}
        self.wfile.write((json.dumps(response) + "\n").encode("utf-8"))


class _PrivateUnixServer(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, handler, scheduler):
        self.scheduler = scheduler
        self.stop_event = scheduler.stop_event
        self.cancel_job = scheduler.cancel_job
        self.cancel_running_jobs = scheduler.cancel_running_jobs
        self.revoke_browser_profile = scheduler.revoke_browser_profile
        super().__init__(address, handler)


class AgentScheduler:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.store = AgentStateStore(self.data_dir)
        self.stop_event = threading.Event()
        self._server_process: subprocess.Popen | None = None
        self._server_url = ""
        self._server_model = ""
        self._running_jobs: dict[str, threading.Event] = {}
        self._running_job_profiles: dict[str, set[str]] = {}
        self._running_jobs_lock = threading.Lock()
        self.mcp = MCPRuntime(self.store)
        self.socket_path = self.data_dir / "agent" / "scheduler.sock"

    def serve(self) -> None:
        self.store.recover_interrupted_runs()
        if self.socket_path.exists():
            try:
                self.socket_path.unlink()
            except OSError:
                pass
        ipc = _PrivateUnixServer(str(self.socket_path), _PrivateSocketHandler, self)
        os.chmod(self.socket_path, 0o600)
        ipc_thread = threading.Thread(target=ipc.serve_forever, name="nextcore-scheduler-ipc", daemon=True)
        ipc_thread.start()
        logger.info("NextCore Agent scheduler started")
        try:
            while not self.stop_event.wait(15):
                self.run_due_jobs()
        finally:
            ipc.shutdown()
            ipc.server_close()
            ipc_thread.join(timeout=2)
            self.mcp.close()
            self._stop_server()
            try:
                self.socket_path.unlink()
            except FileNotFoundError:
                pass

    @staticmethod
    def _same_model(expected: str, served: str) -> bool:
        expected = str(expected or "").strip().rstrip("/").casefold()
        served = str(served or "").strip().rstrip("/").casefold()
        if not expected or not served:
            return False
        if expected == served:
            return True
        expected_tail = expected.replace("\\", "/").rsplit("/", 1)[-1]
        served_tail = served.replace("\\", "/").rsplit("/", 1)[-1]
        return expected_tail == served_tail

    def _server_ready(self, url: str, model_id: str) -> bool:
        try:
            response = requests.get(f"{url.rstrip('/')}/v1/models", timeout=3)
            response.raise_for_status()
            payload = response.json()
            models = payload.get("data") if isinstance(payload, dict) else None
            return isinstance(models, list) and any(
                isinstance(item, dict)
                and self._same_model(model_id, str(item.get("id") or ""))
                for item in models
            )
        except (requests.RequestException, ValueError):
            return False

    def _start_model_if_needed(self, job: dict) -> None:
        url = str(job.get("server_url") or "http://127.0.0.1:30000").rstrip("/")
        parsed = urlsplit(url)
        if parsed.scheme != "http" or not parsed.hostname:
            raise RuntimeError("Scheduled tasks require an HTTP SGLang URL on this Mac.")
        try:
            loopback = ipaddress.ip_address(parsed.hostname).is_loopback
        except ValueError:
            loopback = parsed.hostname.casefold() == "localhost"
        if not loopback:
            raise RuntimeError("Scheduled Agent tasks can connect only to SGLang on this Mac.")
        if parsed.port is None or parsed.port < 1024:
            raise RuntimeError("Scheduled SGLang must use an explicit unprivileged local port.")
        if self._server_ready(url, str(job["model_id"])):
            return
        if (
            self._server_process
            and self._server_process.poll() is None
            and self._server_url == url
            and self._same_model(self._server_model, str(job["model_id"]))
        ):
            pass
        else:
            self._stop_server()
            port = parsed.port
            try:
                with socket.create_connection((parsed.hostname, port), timeout=0.4):
                    occupied = True
            except OSError:
                occupied = False
            if occupied:
                try:
                    response = requests.get(f"{url}/v1/models", timeout=3)
                    response.raise_for_status()
                    payload = response.json()
                except (requests.RequestException, ValueError) as exc:
                    raise RuntimeError(
                        "The scheduled SGLang port is already occupied by a service that NextCore cannot identify; no second server was started."
                    ) from exc
                models = payload.get("data") if isinstance(payload, dict) else None
                served = [
                    str(item.get("id") or "")
                    for item in models or []
                    if isinstance(item, dict) and item.get("id")
                ]
                if served:
                    raise RuntimeError(
                        "The local SGLang endpoint is serving a different model; NextCore left that service running. "
                        f"Expected {job['model_id']!s}; found {', '.join(served[:3])}."
                    )
                raise RuntimeError(
                    "The scheduled SGLang port is occupied but has not reported a ready model; NextCore left it untouched."
                )
            log_path = self.data_dir / "agent" / "scheduled-sglang.log"
            log_handle = log_path.open("ab")
            command = server_command(
                sys.executable,
                str(job["model_id"]),
                parsed.hostname,
                port,
                backend="mlx",
                served_model_name=str(job["model_id"]),
                tool_parser=str(job.get("tool_parser") or "") or None,
            )
            env = dict(os.environ)
            env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2]) + os.pathsep + env.get("PYTHONPATH", "")
            self._server_process = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[2], env=env, stdout=log_handle, stderr=subprocess.STDOUT, start_new_session=True)
            self._server_url = url
            self._server_model = str(job["model_id"])
        deadline = time.monotonic() + 420
        while time.monotonic() < deadline:
            if self.stop_event.is_set():
                raise InterruptedError("Scheduler is stopping.")
            if self._server_ready(url, str(job["model_id"])):
                return
            if self._server_process and self._server_process.poll() is not None:
                raise RuntimeError("SGLang stopped before the scheduled model became ready. See scheduler log.")
            time.sleep(2)
        raise TimeoutError("SGLang did not become ready within seven minutes.")

    def _stop_server(self) -> None:
        process, self._server_process = self._server_process, None
        if process is None or process.poll() is not None:
            self._server_url = ""
            self._server_model = ""
            return
        process.terminate()
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            process.kill()
        self._server_url = ""
        self._server_model = ""

    def run_due_jobs(self) -> None:
        for due in self.store.due_jobs():
            if self.stop_event.is_set():
                return
            now = time.time()
            try:
                next_at = next_run_time(due["schedule"], due["timezone"], after=now)
            except Exception:
                next_at = None
            try:
                # Move the next occurrence forward atomically with the claim.
                # If the process crashes, the current occurrence is recorded as
                # interrupted at next launch and never replayed.
                job = self.store.start_job_run(due["id"], next_run_at=next_at)
            except RuntimeError:
                continue
            run_id = job["run_id"]
            cancellation = threading.Event()
            with self._running_jobs_lock:
                self._running_jobs[job["id"]] = cancellation
                self._running_job_profiles[job["id"]] = set(job.get("authorization_scope", {}).get("browser_profile_ids", []))
            state = "failed"
            summary = ""
            error = ""
            try:
                self._start_model_if_needed(job)
                messages = [{"role": "user", "content": str(job["prompt"])}]
                context = "\n\n".join(filter(None, (self.store.get_memory_context(), self.store.get_skill_context())))
                if context:
                    messages.insert(0, {"role": "system", "content": context})
                agent = SGLangAgent(
                    server_url=job["server_url"], model=job["model_id"], messages=messages,
                    max_tokens=2048, temperature=0.4, app_data_dir=self.data_dir,
                    run_id=f"scheduled-{run_id}", tool_parser=job.get("tool_parser") or None,
                    cancel_event=cancellation, allowed_tools=set(job.get("tool_scope", [])),
                    authorization_scope=job.get("authorization_scope", {}), mcp_runtime=self.mcp,
                )
                result = agent.run()
                summary = result["reply"][:4_000]
                state = "succeeded"
            except InterruptedError as exc:
                state = "interrupted"
                error = str(exc)[:1000]
            except AgentCancelledError as exc:
                state = "cancelled" if cancellation.is_set() else "interrupted"
                error = str(exc)[:1000]
            except Exception as exc:
                error = f"{type(exc).__name__}: {str(exc)[:900]}"
                logger.warning("Scheduled Agent task failed id=%s type=%s", job["id"], type(exc).__name__)
            finally:
                with self._running_jobs_lock:
                    self._running_jobs.pop(job["id"], None)
                    self._running_job_profiles.pop(job["id"], None)
            self.store.finish_job_run(run_id, job["id"], state=state, summary=summary, error=error, next_run_at=next_at)

    def cancel_job(self, job_id: str) -> bool:
        with self._running_jobs_lock:
            event = self._running_jobs.get(job_id)
            if event is None:
                return False
            event.set()
            return True

    def cancel_running_jobs(self) -> None:
        with self._running_jobs_lock:
            events = list(self._running_jobs.values())
        for event in events:
            event.set()

    def revoke_browser_profile(self, profile_id: str) -> list[str]:
        with self._running_jobs_lock:
            affected = [
                (job_id, event)
                for job_id, event in self._running_jobs.items()
                if profile_id in self._running_job_profiles.get(job_id, set())
            ]
        for _, event in affected:
            event.set()
        if affected:
            from .tools import close_computer_backend
            close_computer_backend()
        return [job_id for job_id, _ in affected]


def main() -> None:
    parser = argparse.ArgumentParser(description="NextCore private Agent scheduler")
    default_dir = os.environ.get("TOKEN_WORKSHED_AGENT_DATA_DIR") or str(Path.home() / "Library/Application Support/token-workshed-nextcore")
    parser.add_argument("--data-dir", type=Path, default=Path(default_dir))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    scheduler = AgentScheduler(args.data_dir)
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: (scheduler.stop_event.set(), scheduler.cancel_running_jobs()))
    scheduler.serve()


if __name__ == "__main__":
    main()
