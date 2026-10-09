"""Report helpers for bridge execution."""

from __future__ import annotations

from typing import Any


def build_report(
    *,
    model: str,
    family: str,
    matched_rule_name: str | None,
    needs_manual_review: bool,
    run_mode: str,
    commands: dict[str, Any],
    metadata: dict[str, Any],
    health: dict[str, Any],
    smoke: dict[str, Any],
    openclaw_written: bool,
    hermes_written: bool,
    failures: list[str],
    fixes: list[str],
    warnings: list[str] | None = None,
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    smoke_ok = bool(smoke.get("ok"))
    tool_calling = str(smoke.get("tool_calling") or "").strip().lower()
    if tool_calling not in {"passed", "degraded"}:
        tool_calling = "passed" if smoke_ok else "failed"
    health_ok = bool(health.get("ok"))
    report: dict[str, Any] = {
        "model": model,
        "family": family,
        "matched_rule": matched_rule_name,
        "run_mode": run_mode,
        "agent_ready": bool(health_ok and smoke_ok and not failures),
        # ``degraded`` means the model/server answered successfully but did
        # not produce structured tool_calls.  It is still agent-ready for
        # direct replies; runtime code can use the state to avoid advertising
        # tools the model cannot reliably invoke.
        "tool_calling": tool_calling,
        "parser_used": commands.get("parser_used"),
        "reasoning_parser_used": commands.get("reasoning_parser_used"),
        "chat_template_used": commands.get("chat_template_used"),
        "openclaw_config_written": bool(openclaw_written),
        "hermes_config_written": bool(hermes_written),
        "needs_manual_review": bool(needs_manual_review),
        "metadata": {
            "model_dir": metadata.get("model_dir"),
            "model_type": metadata.get("model_type"),
            "architecture": metadata.get("architecture"),
            "max_context_length": metadata.get("max_context_length"),
            "chat_template": metadata.get("chat_template"),
            "special_tokens": metadata.get("special_tokens"),
            "warnings": metadata.get("warnings", []),
        },
        "commands": {
            "mlx_primary_command": commands.get("mlx_primary_command"),
            "mlx_primary_command_text": commands.get("mlx_primary_command_text"),
            "upstream_reference_command": commands.get("upstream_reference_command"),
            "upstream_reference_command_text": commands.get(
                "upstream_reference_command_text"
            ),
            "unsupported_overrides": commands.get("unsupported_overrides", []),
        },
        "health_check": health,
        "smoke_test": smoke,
        "failures": failures,
        "fix_suggestions": fixes,
        # Metadata/registry warnings are useful diagnostics, but they do not
        # make a runtime probe fail.  Keeping them separate prevents a missing
        # local cache from blocking Configure when the live smoke test passes.
        "warnings": list(warnings or []),
    }
    if extras:
        report.update(extras)
    return report


def build_human_summary(report: dict[str, Any]) -> str:
    lines = [
        f"Model: {report.get('model')}",
        f"Family: {report.get('family')} | rule: {report.get('matched_rule') or 'none'}",
        f"Run mode: {report.get('run_mode')}",
        f"Agent ready: {report.get('agent_ready')}",
        f"Tool calling: {report.get('tool_calling')}",
        f"Parser used: {report.get('parser_used') or 'none'}",
        f"Reasoning parser: {report.get('reasoning_parser_used') or 'none'}",
        f"Chat template: {report.get('chat_template_used') or 'none'}",
        f"OpenClaw config written: {report.get('openclaw_config_written')}",
        f"Hermes config written: {report.get('hermes_config_written')}",
        f"Needs manual review: {report.get('needs_manual_review')}",
        "Primary command:",
        f"  {report.get('commands', {}).get('mlx_primary_command_text', '')}",
        "Reference command:",
        f"  {report.get('commands', {}).get('upstream_reference_command_text', '')}",
    ]
    failures = report.get("failures") or []
    if failures:
        lines.append("Failures:")
        lines.extend(f"  - {item}" for item in failures)
    warnings = report.get("warnings") or []
    if warnings:
        lines.append("Warnings:")
        lines.extend(f"  - {item}" for item in warnings)
    return "\n".join(lines)
