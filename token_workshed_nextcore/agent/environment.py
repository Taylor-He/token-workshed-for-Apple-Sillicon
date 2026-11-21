"""Credential-safe environment helpers for Agent-owned child processes."""

from __future__ import annotations

import os
import re
from urllib.parse import parse_qs, urlparse

_SENSITIVE_ENV_MARKERS = (
    "API_KEY",
    "APIKEY",
    "ACCESS_KEY",
    "SECRET",
    "TOKEN",
    "PASSWORD",
    "PASSWD",
    "CREDENTIAL",
    "AUTHORIZATION",
    "_AUTH",
    "COOKIE",
    "SESSION",
    "BEARER",
    "PRIVATE_KEY",
    "PRIVATEKEY",
    "SIGNING_KEY",
    "CLIENT_SECRET",
    "CERTIFICATE",
    "_CERT",
    "JWT",
    "_PAT",
    "SSH_AUTH_SOCK",
    "SSH_AGENT_PID",
    "GIT_ASKPASS",
    "DYLD_",
)
_SENSITIVE_URL_PARAMETER = re.compile(
    r"(?i)(?:API[_-]?KEY|ACCESS[_-]?KEY|SECRET|TOKEN|PASSWORD|PASSWD|CREDENTIAL|AUTH|COOKIE|SESSION|BEARER|PRIVATE[_-]?KEY|JWT|PAT)"
)
_SHELL_STARTUP_ENV = {"BASH_ENV", "ENV", "ZDOTDIR"}


def sanitized_environment(base_env: dict[str, str] | None = None) -> dict[str, str]:
    """Remove likely credentials and shell injection hooks from inherited env."""
    values = dict(base_env if base_env is not None else os.environ)
    safe: dict[str, str] = {}
    for key, value in values.items():
        normalized = key.upper()
        if normalized in _SHELL_STARTUP_ENV or any(
            marker in normalized for marker in _SENSITIVE_ENV_MARKERS
        ):
            continue
        try:
            parsed_value = urlparse(value.strip()) if "://" in value else None
        except ValueError:
            # A malformed URL may still contain credentials; fail closed.
            continue
        if parsed_value and parsed_value.netloc and "@" in parsed_value.netloc:
            continue
        if parsed_value and any(
            _SENSITIVE_URL_PARAMETER.search(name)
            for name in parse_qs(parsed_value.query)
        ):
            continue
        safe[key] = value
    return safe
