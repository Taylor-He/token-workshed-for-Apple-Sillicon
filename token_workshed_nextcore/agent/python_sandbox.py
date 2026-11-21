"""Fail-closed Python execution using macOS Seatbelt sandbox-exec."""

from __future__ import annotations

import resource
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

MAX_CODE_BYTES = 24_000
MAX_OUTPUT_BYTES = 20_000


def execute_python(code: str, *, timeout_seconds: int = 10, cancel_event=None) -> str:
    if sys.platform != "darwin":
        raise RuntimeError("Isolated Python is unavailable on this platform; execution was refused.")
    sandbox_exec = shutil.which("sandbox-exec")
    if not sandbox_exec:
        raise RuntimeError("macOS sandbox-exec is unavailable; isolated Python was refused.")
    source = str(code or "")
    if not source.strip():
        raise ValueError("Python code is required.")
    if len(source.encode("utf-8")) > MAX_CODE_BYTES:
        raise ValueError(f"Python code is limited to {MAX_CODE_BYTES} bytes.")
    timeout = max(1, min(int(timeout_seconds), 30))
    with tempfile.TemporaryDirectory(prefix="nextcore-python-") as temp:
        workdir = Path(temp).resolve()
        script = workdir / "main.py"
        script.write_text(source, encoding="utf-8")
        prefix = Path(sys.prefix).resolve()
        executable = Path(sys.executable).resolve()
        # No network or host filesystem rule is granted. The interpreter may
        # read its own runtime and standard OS libraries, and can write only
        # inside this single temporary work directory.
        allowed_read = {str(prefix), str(Path(sys.base_prefix).resolve()), "/System/Library", "/usr/lib", "/dev"}
        rules = ["(version 1)", "(deny default)", "(allow process-fork)", f'(allow process-exec (literal "{executable}"))']
        rules.extend(f'(allow file-read* (subpath "{path}"))' for path in sorted(allowed_read))
        rules.extend([
            f'(allow file-read* (subpath "{workdir}"))',
            f'(allow file-write* (subpath "{workdir}"))',
            '(allow sysctl-read)',
            '(allow mach-lookup (global-name "com.apple.system.logger"))',
        ])
        profile = workdir / "sandbox.sb"
        profile.write_text("\n".join(rules), encoding="utf-8")
        env = {
            "PATH": str(Path(sys.executable).parent),
            "HOME": str(workdir),
            "TMPDIR": str(workdir),
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        def limit_resources() -> None:
            resource.setrlimit(resource.RLIMIT_CPU, (timeout, timeout + 1))
            resource.setrlimit(resource.RLIMIT_FSIZE, (1_000_000, 1_000_000))
            resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))

        with tempfile.TemporaryFile(dir=workdir) as output_file:
            process = subprocess.Popen(
                [sandbox_exec, "-f", str(profile), str(executable), "-I", "-S", str(script)],
                cwd=workdir,
                env=env,
                stdout=output_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                preexec_fn=limit_resources,
            )
            deadline = time.monotonic() + timeout
            while process.poll() is None:
                if cancel_event is not None and cancel_event.is_set():
                    process.kill()
                    process.wait()
                    raise InterruptedError("Python sandbox execution was cancelled.")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    process.kill()
                    process.wait()
                    raise TimeoutError(f"Python sandbox exceeded its {timeout}-second limit.")
                time.sleep(min(0.1, remaining))
            output_file.seek(0)
            output = output_file.read(MAX_OUTPUT_BYTES + 1).decode("utf-8", errors="replace")
        output = (output or "")[:MAX_OUTPUT_BYTES]
        if process.returncode:
            if process.returncode == 71 and (
                "sandbox_apply" in output or "Operation not permitted" in output
            ):
                raise RuntimeError(
                    "macOS denied the Python sandbox policy; isolated execution was refused and the code was not run."
                )
            return f"Python sandbox exited with code {process.returncode}.\n{output}"[:MAX_OUTPUT_BYTES]
        return output or "Python sandbox completed without output."
