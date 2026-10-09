"""Main orchestration for OpenClaw/Hermes bridge workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from .args import parse_bridge_args
from .command_builder import build_commands
from .config_writer import (
    detect_api_mode,
    write_hermes_config,
    write_openclaw_config,
    write_result_to_dict,
)
from .defaults import BridgePaths, default_paths
from .family_detect import detect_model_family
from .launcher_probe import launch_and_probe
from .model_meta import read_model_metadata
from .registry import load_registry, match_rule, resolve_settings
from .report import build_human_summary, build_report
from .smoke import run_tool_call_smoke


def _build_paths(args: Any) -> BridgePaths:
    paths = default_paths()
    if not args.openclaw_config_path and not args.hermes_home:
        return paths
    openclaw_config_path = (
        Path(args.openclaw_config_path).expanduser()
        if args.openclaw_config_path
        else paths.openclaw_config_path
    )
    if args.hermes_home:
        hermes_home = Path(args.hermes_home).expanduser()
        hermes_env_path = hermes_home / ".env"
        hermes_config_path = hermes_home / "config.yaml"
    else:
        hermes_home = paths.hermes_home_dir
        hermes_env_path = paths.hermes_env_path
        hermes_config_path = paths.hermes_config_path
    return BridgePaths(
        root_dir=paths.root_dir,
        openclaw_config_path=openclaw_config_path,
        openclaw_workspace_dir=paths.openclaw_workspace_dir,
        hermes_home_dir=hermes_home,
        hermes_env_path=hermes_env_path,
        hermes_config_path=hermes_config_path,
    )


def _write_script(path: str, primary_command: str, reference_command: str) -> str:
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    body = (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n\n"
        "# Primary (vllm-mlx)\n"
        f"{primary_command}\n\n"
        "# Reference (upstream vllm)\n"
        f"# {reference_command}\n"
    )
    output_path.write_text(body, encoding="utf-8")
    output_path.chmod(0o755)
    return str(output_path)


def _run_bridge_with_args(
    args: Any,
    *,
    emit_output: bool,
) -> tuple[int, dict[str, Any]]:
    paths = _build_paths(args)
    run_mode = "run" if args.run else "dry-run"
    base_url = f"http://{args.host}:{args.port}"
    base_url_v1 = f"{base_url}/v1"

    metadata = read_model_metadata(
        model=args.model, allow_download=bool(args.allow_download)
    )
    family = detect_model_family(args.model, metadata)

    rules, registry_warnings = load_registry(args.registry)
    matched_rule, needs_manual_review = match_rule(
        model_id=args.model,
        family=family,
        metadata=metadata,
        rules=rules,
    )
    settings = resolve_settings(
        family=family,
        matched_rule=matched_rule,
        override_tool_call_parser=args.tool_call_parser,
        override_reasoning_parser=args.reasoning_parser,
        override_chat_template=args.chat_template,
    )

    commands = build_commands(
        model=args.model,
        host=args.host,
        port=args.port,
        settings=settings,
        python_executable=args.python,
        primary_module=args.primary_module,
        gpu_args=list(args.gpu_arg or []),
        context_length=args.context_length,
        served_model_name=args.served_model_name,
        max_model_len=args.max_model_len,
    )
    saved_script_path = None
    if args.save_script:
        saved_script_path = _write_script(
            args.save_script,
            commands["mlx_primary_command_text"],
            commands["upstream_reference_command_text"],
        )

    health = {
        "ok": False,
        "skipped": True,
        "launched": False,
        "reused_existing": False,
        "pid": None,
        "error": "dry-run",
        "models": [],
    }
    smoke = {
        "ok": False,
        "skipped": True,
        "reason": "dry-run",
        "tool_calls": 0,
    }
    failures: list[str] = []
    fixes: list[str] = []
    warnings: list[str] = []

    if args.run:
        health = launch_and_probe(
            command=commands["mlx_primary_command"],
            host=args.host,
            port=args.port,
            base_url=base_url,
            expected_model=args.model,
            startup_timeout_seconds=float(args.startup_timeout),
            request_timeout_seconds=float(args.request_timeout),
            reuse_running=not bool(args.no_reuse_running),
            stop_after_check=bool(args.stop_after_check),
        )
        health["skipped"] = False
        if not health.get("ok"):
            failures.append(str(health.get("error") or "health check failed"))
            fixes.append(
                "Check port occupancy, model id, and startup logs; then retry with --run."
            )
        else:
            smoke = run_tool_call_smoke(
                base_url=base_url,
                # vLLM-compatible servers may expose a basename or absolute
                # snapshot path instead of the UI's repo id.  Health probing
                # resolves the exact served alias so the smoke request is not
                # rejected before it reaches the model.
                model=str(health.get("served_model") or args.model),
                timeout_seconds=float(args.request_timeout),
                allow_text_fallback=bool(getattr(args, "allow_text_fallback", False)),
            )
            smoke["skipped"] = False
            smoke_warnings = smoke.get("warnings")
            if isinstance(smoke_warnings, list):
                for warning in smoke_warnings:
                    warning_text = str(warning or "").strip()
                    if warning_text and warning_text not in warnings:
                        warnings.append(warning_text)
            if not smoke.get("ok"):
                failures.append(str(smoke.get("reason") or "tool-call smoke failed"))
                fixes.append(
                    "Select a stronger tool-capable model or override parser with --tool-call-parser."
                )
    else:
        fixes.append(
            "Run with --run to execute health check and strict tool-call smoke test."
        )

    openclaw_result = None
    hermes_results = None
    inferred_context = metadata.get("max_context_length")
    context_window = int(
        args.context_length or inferred_context or args.max_model_len or 16384
    )
    if args.agent in {"openclaw", "all"}:
        openclaw_result = write_openclaw_config(
            paths=paths,
            model_id=args.model,
            base_url_v1=base_url_v1,
            context_window=context_window,
            max_tokens=int(args.max_tokens),
            api_key=str(args.api_key),
            dry_run=bool(args.dry_run),
        )
    if args.agent in {"hermes", "all"}:
        hermes_results = write_hermes_config(
            paths=paths,
            model_id=args.model,
            base_url_v1=base_url_v1,
            api_key=str(args.api_key),
            dry_run=bool(args.dry_run),
        )
        mode = detect_api_mode(base_url_v1)
        if mode == "anthropic_messages":
            needs_manual_review = True
            failures.append(
                "Hermes detected non-OpenAI-compatible transport for this base URL."
            )
            fixes.append(
                "Use an OpenAI-compatible /v1 endpoint for Hermes or configure provider-specific flow manually."
            )

    metadata_warnings = metadata.get("warnings", [])
    if isinstance(metadata_warnings, str):
        metadata_warnings = [metadata_warnings]
    elif not isinstance(metadata_warnings, list):
        metadata_warnings = []
    for warning in registry_warnings + metadata_warnings:
        warning_text = str(warning or "").strip()
        if warning_text and warning_text not in warnings:
            warnings.append(warning_text)

    report = build_report(
        model=args.model,
        family=family,
        matched_rule_name=matched_rule["name"] if matched_rule else None,
        needs_manual_review=needs_manual_review,
        run_mode=run_mode,
        commands=commands,
        metadata=metadata,
        health=health,
        smoke=smoke,
        openclaw_written=bool(openclaw_result and openclaw_result.written),
        hermes_written=bool(
            hermes_results
            and hermes_results.get("env")
            and hermes_results.get("config")
            and hermes_results["env"].written
            and hermes_results["config"].written
        ),
        failures=failures,
        fixes=fixes,
        warnings=warnings,
        extras={
            "saved_script_path": saved_script_path,
            "openclaw_write": (
                write_result_to_dict(openclaw_result) if openclaw_result else None
            ),
            "hermes_write": (
                {
                    "env": write_result_to_dict(hermes_results["env"]),
                    "config": write_result_to_dict(hermes_results["config"]),
                }
                if hermes_results
                else None
            ),
        },
    )

    if args.report_file:
        report_path = Path(args.report_file).expanduser()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    if emit_output:
        print(build_human_summary(report))
        print("\nJSON report:")
        print(json.dumps(report, ensure_ascii=False, indent=2))

    exit_code = 1
    if args.run and report.get("agent_ready"):
        exit_code = 0
    elif not args.run:
        # dry-run should not fail the process unless configuration analysis exploded.
        exit_code = 0
    return exit_code, report


def run_bridge_programmatic(
    *,
    model: str,
    agent: str = "all",
    host: str = "127.0.0.1",
    port: int = 8000,
    context_length: int | None = None,
    max_tokens: int = 2048,
    gpu_arg: list[str] | None = None,
    run: bool = True,
    dry_run: bool = False,
    tool_call_parser: str | None = None,
    reasoning_parser: str | None = None,
    chat_template: str | None = None,
    served_model_name: str | None = None,
    max_model_len: int | None = None,
    report_file: str | None = None,
    save_script: str | None = None,
    registry: str | None = None,
    allow_download: bool = False,
    startup_timeout: float = 90.0,
    request_timeout: float = 20.0,
    stop_after_check: bool = False,
    no_reuse_running: bool = False,
    api_key: str = "token-workshed-local",
    openclaw_config_path: str | None = None,
    hermes_home: str | None = None,
    python: str | None = None,
    primary_module: str = "vllm_mlx.cli",
    verbose: bool = False,
    allow_text_fallback: bool = True,
) -> dict[str, Any]:
    model_name = str(model or "").strip()
    if not model_name:
        raise ValueError("model is required")
    if run and dry_run:
        raise ValueError("run and dry_run cannot both be true")
    if not run and not dry_run:
        dry_run = True

    mode_flag = "--run" if run and not dry_run else "--dry-run"
    args = parse_bridge_args([model_name, mode_flag])
    args.agent = agent
    args.host = host
    args.port = int(port)
    args.context_length = context_length
    args.max_tokens = int(max_tokens)
    args.gpu_arg = list(gpu_arg or [])
    args.run = bool(run and not dry_run)
    args.dry_run = bool(dry_run or not args.run)
    args.tool_call_parser = tool_call_parser
    args.reasoning_parser = reasoning_parser
    args.chat_template = chat_template
    args.served_model_name = served_model_name
    args.max_model_len = max_model_len
    args.report_file = report_file
    args.save_script = save_script
    args.registry = registry
    args.allow_download = bool(allow_download)
    args.startup_timeout = float(startup_timeout)
    args.request_timeout = float(request_timeout)
    args.stop_after_check = bool(stop_after_check)
    args.no_reuse_running = bool(no_reuse_running)
    args.api_key = api_key
    args.openclaw_config_path = openclaw_config_path
    args.hermes_home = hermes_home
    if python:
        args.python = python
    args.primary_module = primary_module
    args.verbose = bool(verbose)
    # Configure uses a capability matrix: structured tool calls are preferred,
    # but a healthy text-only model is still useful for direct agent replies.
    # The CLI remains strict unless this programmatic flag is explicitly set.
    args.allow_text_fallback = bool(allow_text_fallback)

    exit_code, report = _run_bridge_with_args(args, emit_output=False)
    report["exit_code"] = exit_code
    return report


def run_bridge(argv: Sequence[str] | None = None) -> int:
    args = parse_bridge_args(argv)
    exit_code, _ = _run_bridge_with_args(args, emit_output=True)
    return exit_code


def main(argv: Sequence[str] | None = None) -> int:
    return run_bridge(argv)
