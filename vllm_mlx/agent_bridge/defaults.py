"""Defaults and shared paths for the agent bridge."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
INTEGRATIONS_DIR = ROOT_DIR / "integrations"
OPENCLAW_DIR = INTEGRATIONS_DIR / "openclaw"
HERMES_DIR = INTEGRATIONS_DIR / "hermes"

OPENCLAW_STATE_DIR = OPENCLAW_DIR / ".token-workshed-state"
OPENCLAW_CONFIG_PATH = OPENCLAW_STATE_DIR / "openclaw.json"
OPENCLAW_WORKSPACE_DIR = OPENCLAW_STATE_DIR / "workspace"

HERMES_STATE_DIR = HERMES_DIR / ".token-workshed-state"
HERMES_HOME_DIR = HERMES_STATE_DIR / "home"
HERMES_ENV_PATH = HERMES_HOME_DIR / ".env"
HERMES_CONFIG_PATH = HERMES_HOME_DIR / "config.yaml"


@dataclass(frozen=True)
class BridgePaths:
    root_dir: Path
    openclaw_config_path: Path
    openclaw_workspace_dir: Path
    hermes_home_dir: Path
    hermes_env_path: Path
    hermes_config_path: Path


def default_paths() -> BridgePaths:
    return BridgePaths(
        root_dir=ROOT_DIR,
        openclaw_config_path=OPENCLAW_CONFIG_PATH,
        openclaw_workspace_dir=OPENCLAW_WORKSPACE_DIR,
        hermes_home_dir=HERMES_HOME_DIR,
        hermes_env_path=HERMES_ENV_PATH,
        hermes_config_path=HERMES_CONFIG_PATH,
    )


DEFAULT_REGISTRY_RULES: list[dict[str, object]] = [
    {
        "name": "hermes",
        "match": ["hermes", "nousresearch/hermes"],
        "families": ["hermes"],
        "tool_call_parser": "hermes",
        "enable_auto_tool_choice": True,
    },
    {
        "name": "qwen3_coder",
        "match": ["qwen3-coder", "qwen3 coder"],
        "families": ["qwen3_coder"],
        "tool_call_parser": "qwen3_coder",
        "reasoning_parser": "qwen3",
        "enable_auto_tool_choice": True,
    },
    {
        "name": "qwen3",
        "match": ["qwen3", "qwq"],
        "families": ["qwen3"],
        "tool_call_parser": "qwen",
        "reasoning_parser": "qwen3",
        "enable_auto_tool_choice": True,
    },
    {
        "name": "qwen",
        "match": ["qwen"],
        "families": ["qwen"],
        "tool_call_parser": "qwen",
        "enable_auto_tool_choice": True,
    },
    {
        "name": "llama31",
        "match": ["llama-3.1", "llama 3.1"],
        "families": ["llama31"],
        "tool_call_parser": "llama",
        "enable_auto_tool_choice": True,
    },
    {
        "name": "llama32",
        "match": ["llama-3.2", "llama 3.2"],
        "families": ["llama32"],
        "tool_call_parser": "llama",
        "enable_auto_tool_choice": True,
    },
    {
        "name": "llama4",
        "match": ["llama-4", "llama 4"],
        "families": ["llama4"],
        "tool_call_parser": "llama",
        "enable_auto_tool_choice": True,
    },
    {
        "name": "mistral",
        "match": ["mistral"],
        "families": ["mistral"],
        "tool_call_parser": "mistral",
        "enable_auto_tool_choice": True,
    },
    {
        "name": "granite",
        "match": ["granite", "ibm-granite"],
        "families": ["granite"],
        "tool_call_parser": "granite",
        "enable_auto_tool_choice": True,
    },
    {
        "name": "deepseek",
        "match": ["deepseek"],
        "families": ["deepseek"],
        "tool_call_parser": "deepseek",
        "reasoning_parser": "deepseek_r1",
        "enable_auto_tool_choice": True,
    },
    {
        "name": "glm",
        "match": ["glm-4.7", "glm47", "glm"],
        "families": ["glm"],
        "tool_call_parser": "glm47",
        "enable_auto_tool_choice": True,
    },
    {
        "name": "gpt_oss",
        "match": ["gpt-oss", "harmony"],
        "families": ["gpt_oss"],
        "tool_call_parser": "auto",
        "reasoning_parser": "gpt_oss",
        "enable_auto_tool_choice": True,
    },
]


FAMILY_FALLBACK_SETTINGS: dict[str, dict[str, object]] = {
    "hermes": {"tool_call_parser": "hermes", "enable_auto_tool_choice": True},
    "qwen3_coder": {
        "tool_call_parser": "qwen3_coder",
        "reasoning_parser": "qwen3",
        "enable_auto_tool_choice": True,
    },
    "qwen3": {
        "tool_call_parser": "qwen",
        "reasoning_parser": "qwen3",
        "enable_auto_tool_choice": True,
    },
    "qwen": {"tool_call_parser": "qwen", "enable_auto_tool_choice": True},
    "llama31": {"tool_call_parser": "llama", "enable_auto_tool_choice": True},
    "llama32": {"tool_call_parser": "llama", "enable_auto_tool_choice": True},
    "llama4": {"tool_call_parser": "llama", "enable_auto_tool_choice": True},
    "mistral": {"tool_call_parser": "mistral", "enable_auto_tool_choice": True},
    "granite": {"tool_call_parser": "granite", "enable_auto_tool_choice": True},
    "deepseek": {
        "tool_call_parser": "deepseek",
        "reasoning_parser": "deepseek_r1",
        "enable_auto_tool_choice": True,
    },
    "glm": {"tool_call_parser": "glm47", "enable_auto_tool_choice": True},
    "gpt_oss": {
        "tool_call_parser": "auto",
        "reasoning_parser": "gpt_oss",
        "enable_auto_tool_choice": True,
    },
    "unknown": {"tool_call_parser": "auto", "enable_auto_tool_choice": True},
}
