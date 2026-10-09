"""Build primary and reference startup commands."""

from __future__ import annotations

import shlex
from typing import Any


def _extend_raw_args(command: list[str], raw_args: list[str]) -> None:
    for raw in raw_args:
        text = str(raw or "").strip()
        if not text:
            continue
        command.extend(shlex.split(text))


def _shell_join(command: list[str]) -> str:
    return " ".join(shlex.quote(item) for item in command)


def build_commands(
    *,
    model: str,
    host: str,
    port: int,
    settings: dict[str, Any],
    python_executable: str,
    primary_module: str,
    gpu_args: list[str],
    context_length: int | None,
    served_model_name: str | None,
    max_model_len: int | None,
) -> dict[str, Any]:
    parser = str(settings.get("tool_call_parser") or "").strip()
    reasoning = str(settings.get("reasoning_parser") or "").strip()
    chat_template = str(settings.get("chat_template") or "").strip()
    enable_auto_tool_choice = bool(settings.get("enable_auto_tool_choice", False))
    unsupported: list[str] = []

    primary = [
        python_executable,
        "-m",
        primary_module,
        "serve",
        model,
        "--host",
        host,
        "--port",
        str(port),
    ]
    if parser and enable_auto_tool_choice:
        primary.extend(["--enable-auto-tool-choice", "--tool-call-parser", parser])
    elif enable_auto_tool_choice:
        primary.append("--enable-auto-tool-choice")
    if reasoning:
        primary.extend(["--reasoning-parser", reasoning])
    _extend_raw_args(primary, gpu_args)

    if chat_template:
        unsupported.append(
            "mlx primary command does not expose --chat-template; kept in report/reference command only."
        )
    if served_model_name:
        unsupported.append(
            "mlx primary command does not expose --served-model-name; kept in report/reference command only."
        )
    if max_model_len is not None:
        unsupported.append(
            "mlx primary command does not expose --max-model-len; kept in report/reference command only."
        )
    if context_length is not None and max_model_len is None:
        unsupported.append(
            "--context-length provided but mlx primary command has no direct max-model-len flag."
        )

    reference = [
        "vllm",
        "serve",
        model,
        "--host",
        host,
        "--port",
        str(port),
    ]
    if parser and enable_auto_tool_choice:
        reference.extend(["--enable-auto-tool-choice", "--tool-call-parser", parser])
    elif enable_auto_tool_choice:
        reference.append("--enable-auto-tool-choice")
    if reasoning:
        reference.extend(["--reasoning-parser", reasoning])
    if chat_template:
        reference.extend(["--chat-template", chat_template])
    if served_model_name:
        reference.extend(["--served-model-name", served_model_name])
    effective_max_model_len = max_model_len
    if effective_max_model_len is None and context_length is not None:
        effective_max_model_len = context_length
    if effective_max_model_len is not None:
        reference.extend(["--max-model-len", str(int(effective_max_model_len))])
    _extend_raw_args(reference, gpu_args)

    return {
        "mlx_primary_command": primary,
        "upstream_reference_command": reference,
        "mlx_primary_command_text": _shell_join(primary),
        "upstream_reference_command_text": _shell_join(reference),
        "unsupported_overrides": unsupported,
        "parser_used": parser or None,
        "reasoning_parser_used": reasoning or None,
        "chat_template_used": chat_template or None,
        "enable_auto_tool_choice": enable_auto_tool_choice,
    }
