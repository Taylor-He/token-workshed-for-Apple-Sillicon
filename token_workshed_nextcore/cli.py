#!/usr/bin/env python3
"""NextCore command line: serve and benchmark through SGLang."""
from __future__ import annotations

import argparse
import os
import sys

from .sglang_runtime import server_command


def main() -> None:
    parser = argparse.ArgumentParser(prog="token-workshed-nextcore")
    sub = parser.add_subparsers(dest="action", required=True)
    serve = sub.add_parser("serve", help="Start a local SGLang server")
    serve.add_argument("model")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8010)
    serve.add_argument("--backend", choices=("mlx", "torch-mps"), default="mlx")
    serve.add_argument("--tool-call-parser")
    serve.add_argument("--reasoning-parser")
    serve.add_argument("--served-model-name")
    serve.add_argument("--enable-auto-tool-choice", action="store_true")
    sub.add_parser("bench", help="Run the SGLang serving benchmark")
    args, extra = parser.parse_known_args()
    if args.action == "bench":
        command = [sys.executable, "-m", "sglang.bench_serving", *extra]
    else:
        command = server_command(
            sys.executable, args.model, args.host, args.port, backend=args.backend,
            served_model_name=args.served_model_name,
            tool_parser=args.tool_call_parser or ("auto" if args.enable_auto_tool_choice else None),
            reasoning_parser=args.reasoning_parser,
        )
        command.extend(extra)
    os.execv(command[0], command)


if __name__ == "__main__":
    main()
