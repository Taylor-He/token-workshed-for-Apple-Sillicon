"""Launch the separately installed, experimental SGLang-Omni ASR runtime."""
from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> None:
    root = Path(os.environ.get("TOKEN_WORKSHED_NEXTCORE_OMNI_ROOT", Path.home() / "Library/Application Support/token-workshed-nextcore/omni"))
    executable = root / ".venv/bin/sgl-omni"
    ffmpeg = Path("/opt/homebrew/opt/ffmpeg@7/lib")
    if not executable.is_file() or not ffmpeg.is_dir():
        raise SystemExit("SGLang-Omni ASR is unavailable. Install the pinned Omni runtime and its ffmpeg@7 prerequisite first.")
    env = os.environ.copy()
    env["DYLD_LIBRARY_PATH"] = str(ffmpeg) + (":" + env["DYLD_LIBRARY_PATH"] if env.get("DYLD_LIBRARY_PATH") else "")
    args = [str(executable), "serve", "--model-path", "Qwen/Qwen3-ASR-0.6B", "--host", "127.0.0.1", "--port", "8012", *sys.argv[1:]]
    os.execve(str(executable), args, env)


if __name__ == "__main__":
    main()
