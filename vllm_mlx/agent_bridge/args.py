"""CLI args for OpenClaw/Hermes to vllm-mlx bridge."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="openclaw-hermes-vllm-bridge",
        description="Analyze model compatibility and connect OpenClaw/Hermes to local vllm-mlx.",
    )
    parser.add_argument(
        "model",
        help="Model id (e.g. NousResearch/Hermes-3-Llama-3.1-8B) or local model path.",
    )
    parser.add_argument(
        "--agent",
        choices=("openclaw", "hermes", "all"),
        default="all",
        help="Target runtime config to generate.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="vLLM host.")
    parser.add_argument("--port", type=int, default=8000, help="vLLM port.")
    parser.add_argument(
        "--context-length",
        type=int,
        default=None,
        help="Preferred context length hint.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Default max_tokens for generated runtime configs.",
    )
    parser.add_argument(
        "--gpu-arg",
        action="append",
        default=[],
        help="Extra GPU/server argument (repeatable, parsed by shell-style splitting).",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Start vllm-mlx server and run health checks + smoke test.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze and render command/config/report only, no server start.",
    )
    parser.add_argument(
        "--tool-call-parser",
        default=None,
        help="Manual override for tool call parser.",
    )
    parser.add_argument(
        "--reasoning-parser",
        default=None,
        help="Manual override for reasoning parser.",
    )
    parser.add_argument(
        "--chat-template",
        default=None,
        help="Manual override for chat template path/name.",
    )
    parser.add_argument(
        "--served-model-name",
        default=None,
        help="Manual override for served model name (upstream reference only).",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Manual override for max model length (upstream reference only).",
    )
    parser.add_argument(
        "--report-file",
        default=None,
        help="Optional path to save machine-readable JSON report.",
    )
    parser.add_argument(
        "--save-script",
        default=None,
        help="Optional path to save generated shell script.",
    )
    parser.add_argument(
        "--registry",
        default=None,
        help="Optional YAML/JSON registry file path.",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow downloading Hugging Face model metadata if not cached locally.",
    )
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=90.0,
        help="Startup readiness timeout in seconds.",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=20.0,
        help="HTTP request timeout for probe/smoke checks in seconds.",
    )
    parser.add_argument(
        "--stop-after-check",
        action="store_true",
        help="Terminate launched vLLM process after checks complete.",
    )
    parser.add_argument(
        "--no-reuse-running",
        action="store_true",
        help="Fail if target port is already serving; do not probe existing server.",
    )
    parser.add_argument(
        "--api-key",
        default="token-workshed-local",
        help="API key placeholder used in generated OpenClaw/Hermes config.",
    )
    parser.add_argument(
        "--openclaw-config-path",
        default=None,
        help="Override OpenClaw config output path.",
    )
    parser.add_argument(
        "--hermes-home",
        default=None,
        help="Override Hermes home directory for .env/config.yaml output.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable for primary vllm-mlx command.",
    )
    parser.add_argument(
        "--primary-module",
        default="vllm_mlx.cli",
        help="Primary serve module to call with -m.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print additional debugging details.",
    )
    return parser


def parse_bridge_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.run and args.dry_run:
        parser.error("Choose either --run or --dry-run, not both.")
    if not args.run and not args.dry_run:
        args.dry_run = True
    return args
