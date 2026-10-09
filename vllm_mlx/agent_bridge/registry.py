"""Registry loading and matching for model-specific adapter rules."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from .defaults import DEFAULT_REGISTRY_RULES, FAMILY_FALLBACK_SETTINGS


def _normalize_rule(raw: dict[str, Any]) -> dict[str, Any]:
    match = raw.get("match", [])
    if isinstance(match, str):
        match = [match]
    if not isinstance(match, list):
        match = []
    families = raw.get("families", [])
    if isinstance(families, str):
        families = [families]
    if not isinstance(families, list):
        families = []
    return {
        "name": str(raw.get("name", "")).strip() or "unnamed",
        "match": [str(item).strip() for item in match if str(item).strip()],
        "families": [str(item).strip() for item in families if str(item).strip()],
        "tool_call_parser": raw.get("tool_call_parser"),
        "reasoning_parser": raw.get("reasoning_parser"),
        "chat_template": raw.get("chat_template"),
        "enable_auto_tool_choice": bool(raw.get("enable_auto_tool_choice", False)),
    }


def load_registry(path: str | None = None) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    rules: list[dict[str, Any]] = [_normalize_rule(item) for item in DEFAULT_REGISTRY_RULES]
    if not path:
        return rules, warnings

    registry_path = Path(path).expanduser()
    if not registry_path.exists():
        warnings.append(f"Registry path not found: {registry_path}")
        return rules, warnings

    try:
        raw_text = registry_path.read_text(encoding="utf-8")
        if registry_path.suffix.lower() in {".yaml", ".yml"}:
            payload = yaml.safe_load(raw_text)
        else:
            payload = json.loads(raw_text)
    except Exception as exc:
        warnings.append(f"Failed to load registry '{registry_path}': {exc}")
        return rules, warnings

    external_rules: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        for value in payload.values():
            if isinstance(value, dict):
                external_rules.append(_normalize_rule(value))
        if "rules" in payload and isinstance(payload["rules"], list):
            external_rules.extend(
                _normalize_rule(item) for item in payload["rules"] if isinstance(item, dict)
            )
    elif isinstance(payload, list):
        external_rules.extend(
            _normalize_rule(item) for item in payload if isinstance(item, dict)
        )
    else:
        warnings.append(
            f"Unsupported registry payload format in '{registry_path}', expected dict or list."
        )
        return rules, warnings

    if external_rules:
        # Custom rules have higher priority.
        rules = external_rules + rules
    return rules, warnings


def match_rule(
    *,
    model_id: str,
    family: str,
    metadata: dict[str, Any],
    rules: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, bool]:
    text_bits: list[str] = [str(model_id or "").lower(), str(family or "").lower()]
    model_type = metadata.get("model_type")
    if isinstance(model_type, str):
        text_bits.append(model_type.lower())
    architecture = metadata.get("architecture")
    if isinstance(architecture, list):
        text_bits.extend(str(item).lower() for item in architecture if isinstance(item, str))
    merged = " ".join(text_bits)

    best_rule: dict[str, Any] | None = None
    best_score = -1
    for rule in rules:
        score = 0
        families = rule.get("families", [])
        if isinstance(families, list) and family in families:
            score += 3
        match_tokens = rule.get("match", [])
        if isinstance(match_tokens, list):
            token_hits = 0
            for token in match_tokens:
                token_text = str(token).strip().lower()
                if token_text and token_text in merged:
                    token_hits += 1
            score += token_hits
        if score > best_score and score > 0:
            best_rule = rule
            best_score = score

    return best_rule, best_rule is None


def resolve_settings(
    *,
    family: str,
    matched_rule: dict[str, Any] | None,
    override_tool_call_parser: str | None,
    override_reasoning_parser: str | None,
    override_chat_template: str | None,
) -> dict[str, Any]:
    base: dict[str, Any] = dict(FAMILY_FALLBACK_SETTINGS.get(family, FAMILY_FALLBACK_SETTINGS["unknown"]))
    if matched_rule:
        for key in (
            "tool_call_parser",
            "reasoning_parser",
            "chat_template",
            "enable_auto_tool_choice",
        ):
            value = matched_rule.get(key)
            if value is not None:
                base[key] = value
    if override_tool_call_parser:
        base["tool_call_parser"] = override_tool_call_parser
        base["enable_auto_tool_choice"] = True
    if override_reasoning_parser:
        base["reasoning_parser"] = override_reasoning_parser
    if override_chat_template:
        base["chat_template"] = override_chat_template
    return base
