"""Pinned SGLang server launcher shared by the desktop and Workshed."""

from __future__ import annotations

import argparse
import os
import platform
import sys
from importlib.metadata import PackageNotFoundError, version

SGLANG_VERSION = "0.5.22"
SGLANG_COMMIT = "baa9befa381511cd4d56f83980951e5a45fffc6d"
OMNI_VERSION = "0.2.0"
OMNI_COMMIT = "e8c635ecb9c68bcf2fbc32ceb3c40f85f3590657"

# Names supported by the pinned SGLang release. Old vLLM parser names are
# translated explicitly so unsupported names never reach the SGLang CLI.
_TOOL_PARSERS = {
    "auto",
    "apertus2509",
    "cohere_command4",
    "deepseekv3",
    "deepseekv31",
    "deepseekv32",
    "deepseekv4",
    "deepseekv41",
    "dots",
    "glm",
    "glm45",
    "glm47",
    "gpt-oss",
    "k2_horizon",
    "kimi_k2",
    "kimi_k3",
    "lfm2",
    "ling3",
    "llama3",
    "mimo",
    "minicpm5",
    "mistral",
    "muse",
    "poolside_v1",
    "pythonic",
    "qwen",
    "qwen25",
    "qwen3_coder",
    "spark25",
    "step3",
    "step3p5",
    "minimax-m2",
    "minimax-m3",
    "nanbeige",
    "trinity",
    "interns1",
    "hermes",
    "hunyuan",
    "gigachat3",
    "gigachat35",
    "gemma4",
    "inkling",
    "iquest_q1",
}
_REASONING_PARSERS = {
    "auto",
    "deepseek-r1",
    "deepseek-v3",
    "deepseek-v4",
    "deepseek-v41",
    "glm45",
    "gpt-oss",
    "kimi",
    "kimi_k2",
    "kimi_k3",
    "mistral",
    "qwen3",
    "qwen3-thinking",
    "minimax",
    "granite_thinking_parser",
    "gemma4",
}


def server_command(
    python: str,
    model: str,
    host: str,
    port: int,
    *,
    backend: str = "mlx",
    served_model_name: str | None = None,
    tool_parser: str | None = None,
    reasoning_parser: str | None = None,
    chunked_prefill_size: int = 0,
) -> list[str]:
    if backend not in {"mlx", "torch-mps"}:
        raise ValueError(f"Unsupported SGLang backend: {backend}")
    command = [
        python,
        "-m",
        "token_workshed_nextcore.sglang_runtime",
        "--backend",
        backend,
        "--model-path",
        model,
        "--host",
        host,
        "--port",
        str(port),
        "--cuda-graph-backend-decode",
        "disabled",
        "--cuda-graph-backend-prefill",
        "disabled",
    ]
    # Deliberately omit --context-length: SGLang should use the loaded model's
    # configured context instead of inheriting an Agent-side fixed ceiling.
    if backend == "torch-mps":
        command.extend(["--device", "mps"])
    if served_model_name:
        command.extend(["--served-model-name", served_model_name])
    if chunked_prefill_size > 0:
        command.extend(["--chunked-prefill-size", str(chunked_prefill_size)])
    parser = str(tool_parser or "").strip().lower()
    parser = {
        "qwen3": "qwen",
        "qwen25": "qwen",
        "deepseek_v3": "deepseekv3",
    }.get(parser, parser)
    if parser:
        command.extend(
            ["--tool-call-parser", parser if parser in _TOOL_PARSERS else "auto"]
        )
    reasoning = str(reasoning_parser or "").strip().lower()
    reasoning = {
        "deepseek_r1": "deepseek-r1",
        "deepseek_v3": "deepseek-v3",
        "gpt_oss": "gpt-oss",
        "qwen3": "qwen3",
    }.get(reasoning, reasoning)
    if reasoning:
        command.extend(
            [
                "--reasoning-parser",
                reasoning if reasoning in _REASONING_PARSERS else "auto",
            ]
        )
    return command


def _require_metal_device(backend: str) -> None:
    if sys.platform != "darwin" or platform.machine().lower() not in {"arm64", "aarch64"}:
        raise SystemExit("NextCore SGLang local runtimes require Apple Silicon macOS.")
    try:
        import torch

        mps_available = bool(torch.backends.mps.is_available())
    except Exception as exc:
        raise SystemExit(
            "NextCore could not initialize PyTorch MPS for the SGLang runtime."
        ) from exc
    if not mps_available:
        raise SystemExit(
            "SGLang cannot start because this process has no accessible Metal/MPS device. "
            "Run NextCore in an interactive macOS session with Apple Silicon GPU access."
        )
    if backend == "mlx":
        try:
            import mlx.core as mx

            mlx_available = bool(mx.metal.is_available())
        except Exception as exc:
            raise SystemExit("NextCore could not initialize the SGLang MLX backend.") from exc
        if not mlx_available:
            raise SystemExit(
                "SGLang MLX cannot start because this process has no accessible Metal device."
            )


def main() -> None:
    try:
        installed_version = version("sglang")
    except PackageNotFoundError as exc:
        raise SystemExit(
            f"SGLang Runtime {SGLANG_VERSION} is not installed. "
            "Run scripts/install_sglang_runtime.sh first."
        ) from exc
    if installed_version != SGLANG_VERSION:
        raise SystemExit(
            f"NextCore requires SGLang Runtime {SGLANG_VERSION}; "
            f"found {installed_version}. Reinstall the pinned runtime."
        )
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--backend", choices=("mlx", "torch-mps"), default="mlx")
    options, remaining = parser.parse_known_args()
    _require_metal_device(options.backend)
    if options.backend == "mlx":
        os.environ["SGLANG_USE_MLX"] = "1"
    else:
        os.environ.pop("SGLANG_USE_MLX", None)
    # Use the pinned runtime's supported `sglang serve` entrypoint. Passing a
    # Hub ID lets SGLang's own backend resolve and fetch the checkpoint.
    sys.argv = ["sglang", "serve", *remaining]
    from sglang.cli.main import main as sglang_cli_main

    sglang_cli_main()


if __name__ == "__main__":
    main()
