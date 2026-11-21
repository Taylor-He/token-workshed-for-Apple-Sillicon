"""Reference schema for NextCore's native desktop-control tool."""

from __future__ import annotations

from typing import Any

COMPUTER_USE_SCHEMA: dict[str, Any] = {
    "name": "computer_use",
    "description": (
        "Operate desktop applications through the CUA Driver managed by "
        "Token Workshed NextCore. The pinned driver is downloaded on first "
        "use. macOS Accessibility and Screen Recording permissions may be "
        "required. Capture a fresh accessibility tree before using element indices."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "capture", "list_apps", "launch_app", "focus_app", "click",
                    "double_click", "right_click", "middle_click", "drag",
                    "scroll", "type", "key", "set_value", "wait",
                ],
            },
            "app": {"type": "string"},
            "mode": {"type": "string", "enum": ["ax", "som", "vision"]},
            "element": {"type": "integer"},
            "x": {"type": "integer"},
            "y": {"type": "integer"},
            "from_element": {"type": "integer"},
            "to_element": {"type": "integer"},
            "from_x": {"type": "integer"},
            "from_y": {"type": "integer"},
            "to_x": {"type": "integer"},
            "to_y": {"type": "integer"},
            "value": {"type": "string"},
            "text": {"type": "string"},
            "keys": {"type": "string"},
            "direction": {"type": "string", "enum": ["up", "down", "left", "right"]},
            "amount": {"type": "integer"},
            "seconds": {"type": "number"},
        },
        "required": ["action"],
    },
}


def get_computer_use_schema() -> dict[str, Any]:
    """Return the OpenAI function-calling schema."""
    return COMPUTER_USE_SCHEMA
