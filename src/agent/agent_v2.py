from __future__ import annotations

import ast
import json
import re
from typing import Any, Dict, Optional

from src.agent.agent import Action, ReActAgent, _strip_code_fence


def parse_action_v2(text: str) -> Optional[Action]:
    """
    V2 parser recovery for malformed args found in the failed trace.

    It keeps JSON as the preferred contract, then accepts a Python literal dict
    such as {'item_name': 'iPhone'} when the structure is otherwise safe.
    """

    match = re.search(r"Action:\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*?)\)", text, re.DOTALL)
    if not match:
        return None

    tool_name = match.group(1)
    raw_args = _extract_object_text(_strip_code_fence(match.group(2).strip()))
    if raw_args is None:
        return None

    args = _loads_dict(raw_args)
    if args is None:
        return None
    return tool_name, args


class ReActAgentV2(ReActAgent):
    agent_version = "v2"

    def parse_action(self, text: str) -> Optional[Action]:
        return parse_action_v2(text)


def _loads_dict(raw_args: str) -> Optional[Dict[str, Any]]:
    try:
        parsed = json.loads(raw_args)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(raw_args)
        except (SyntaxError, ValueError):
            return None

    if not isinstance(parsed, dict):
        return None
    return parsed


def _extract_object_text(text: str) -> Optional[str]:
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string: Optional[str] = None
    escaped = False
    for index, char in enumerate(text[start:], start=start):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == in_string:
                in_string = None
            continue

        if char in ("'", '"'):
            in_string = char
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    return None
