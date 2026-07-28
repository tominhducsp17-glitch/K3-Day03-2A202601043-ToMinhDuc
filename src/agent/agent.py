from __future__ import annotations

import json
import re
from datetime import date
from time import perf_counter
from typing import Any, Callable, Dict, List, Optional, Tuple
from uuid import uuid4

from src.core.llm_provider import LLMProvider
from src.telemetry.logger import logger


Action = Tuple[str, Dict[str, Any]]


def parse_action(text: str) -> Optional[Action]:
    """
    Parse: Action: tool_name({"arg": "value"})

    Returns None when no action is present or when the arguments are not valid
    JSON. Parser recovery is intentionally kept small for V1; Phase 5 can
    improve malformed-output handling based on failed traces.
    """

    match = re.search(r"Action:\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*?)\)", text, re.DOTALL)
    if not match:
        return None

    tool_name = match.group(1)
    raw_args = _strip_code_fence(match.group(2).strip())
    try:
        args = json.loads(raw_args)
    except json.JSONDecodeError:
        return None

    if not isinstance(args, dict):
        return None
    return tool_name, args


def parse_final_answer(text: str) -> Optional[str]:
    match = re.search(r"Final Answer:\s*(.+)", text, re.DOTALL)
    if not match:
        return None
    answer = match.group(1).strip()
    return answer or None


class ReActAgent:
    """
    ReAct-style agent following Thought -> Action -> Observation.

    V1 focuses on the core orchestration invariants:
    - bounded by max_steps
    - one tool action produces one application-written observation
    - observations are included before the next LLM call
    - final answer ends the loop
    """

    agent_version = "v1"

    def __init__(self, llm: LLMProvider, tools: List[Dict[str, Any]], max_steps: int = 5):
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps
        self.history: List[Dict[str, Any]] = []
        self.steps_taken = 0
        self.tool_calls = 0
        self.tool_path: List[str] = []
        self.last_trace: List[Dict[str, Any]] = []
        self._registry = self._build_registry(tools)

    def get_system_prompt(self) -> str:
        tool_descriptions = "\n".join(
            [
                f"- {tool['name']}: {tool.get('description', '')} "
                f"Example: {tool.get('example', 'no example provided')}"
                for tool in self.tools
            ]
        )
        available_names = ", ".join(self._registry)
        today = date.today().isoformat()
        research_rule = (
            "- Use web_research for current events, obscure facts, or source-sensitive questions.\n"
            "- When using research results, cite the source URLs from the Observation."
            if "web_research" in self._registry
            else "- If a question needs live web research and no research tool is available, say what you can and cannot verify."
        )
        return f"""
Your name is Đệ của thầy Tô Đức. You are a general-purpose ReAct assistant.

Current date: {today}

Available tools:
{tool_descriptions}

Allowed tool names: {available_names}

Rules:
- Answer directly with Final Answer when the question does not need a tool.
- Use e-commerce tools for stock, prices, coupon validity, shipping, and order totals.
{research_rule}
- Use at most one Action per step.
- Never invent Observation. The application will execute the Action and add Observation.
- If a tool returns ok=false, explain the limitation or choose a valid next action.
- Do not claim stock, discount, shipping, or order total before tool evidence exists.
- When enough evidence exists, stop with Final Answer.

Output format:
Thought: short reasoning.
Action: tool_name({{"arg": "value"}})

Or:
Final Answer: final customer-facing response.
""".strip()

    def run(self, user_input: str) -> str:
        model_name = getattr(self.llm, "model_name", self.llm.__class__.__name__)
        run_id = str(uuid4())
        agent_version = self.agent_version
        logger.log_event(
            "AGENT_START",
            {"run_id": run_id, "agent_version": agent_version, "input": user_input, "model": model_name},
        )

        self.steps_taken = 0
        self.tool_calls = 0
        self.tool_path = []
        self.last_trace = []
        transcript = f"Question: {user_input}"

        for step in range(1, self.max_steps + 1):
            self.steps_taken = step
            llm_start = perf_counter()
            result = self._generate(transcript)
            measured_latency_ms = int((perf_counter() - llm_start) * 1000)
            llm_output = self._extract_content(result)
            final_answer = self.parse_final_answer(llm_output)
            action = self.parse_action(llm_output)
            usage = result.get("usage", {}) if isinstance(result, dict) else {}
            provider = result.get("provider", "unknown") if isinstance(result, dict) else "scripted"
            latency_ms = result.get("latency_ms", measured_latency_ms) if isinstance(result, dict) else measured_latency_ms

            trace_step: Dict[str, Any] = {
                "run_id": run_id,
                "agent_version": agent_version,
                "step": step,
                "prompt": transcript,
                "llm_output": llm_output,
            }

            logger.log_event(
                "AGENT_STEP",
                {
                    "step": step,
                    "llm_output": llm_output,
                    "usage": usage,
                },
            )
            logger.log_event(
                "LLM_METRIC",
                {
                    "run_id": run_id,
                    "agent_version": agent_version,
                    "step": step,
                    "provider": provider,
                    "model": model_name,
                    "latency_ms": latency_ms,
                    "usage": usage,
                },
            )

            if action:
                tool_name, args = action
                logger.log_event(
                    "TOOL_CALL",
                    {
                        "run_id": run_id,
                        "agent_version": agent_version,
                        "step": step,
                        "tool_name": tool_name,
                        "args": self._sanitize(args),
                    },
                )
                observation = self._execute_tool(tool_name, args)
                self.tool_calls += 1
                self.tool_path.append(tool_name)

                observation_text = f"Observation: {json.dumps(observation, ensure_ascii=False)}"
                transcript = f"{transcript}\n\n{llm_output}\n{observation_text}"

                trace_step.update(
                    {
                        "action": {"tool_name": tool_name, "args": args},
                        "observation": observation,
                    }
                )
                self.last_trace.append(trace_step)
                event_name = "TOOL_RESULT" if observation.get("ok") is not False else "TOOL_ERROR"
                logger.log_event(
                    event_name,
                    {
                        "run_id": run_id,
                        "agent_version": agent_version,
                        "step": step,
                        "tool_name": tool_name,
                        "args": self._sanitize(args),
                        "observation": self._sanitize(observation),
                    },
                )
                continue

            if final_answer:
                trace_step["final_answer"] = final_answer
                self.last_trace.append(trace_step)
                self.history.append({"role": "user", "content": user_input})
                self.history.append({"role": "assistant", "content": final_answer})
                logger.log_event(
                    "AGENT_END",
                    {
                        "run_id": run_id,
                        "agent_version": agent_version,
                        "steps": step,
                        "tool_path": self.tool_path,
                        "status": "final",
                    },
                )
                return final_answer

            observation = {
                "ok": False,
                "error": "parse_error",
                "message": "No valid Action or Final Answer found.",
            }
            transcript = f"{transcript}\n\n{llm_output}\nObservation: {json.dumps(observation)}"
            trace_step["observation"] = observation
            self.last_trace.append(trace_step)
            logger.log_event(
                "AGENT_PARSE_ERROR",
                {
                    "run_id": run_id,
                    "agent_version": agent_version,
                    "step": step,
                    "llm_output": llm_output,
                },
            )

        fallback = "I could not complete the request within the step limit. Please try again with more specific details."
        logger.log_event(
            "AGENT_END",
            {
                "run_id": run_id,
                "agent_version": agent_version,
                "steps": self.max_steps,
                "tool_path": self.tool_path,
                "status": "max_steps",
            },
        )
        return fallback

    def _execute_tool(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        tool = self._registry.get(tool_name)
        if tool is None:
            return {
                "ok": False,
                "error": "unknown_tool",
                "message": f"Tool '{tool_name}' is not available.",
                "available_tools": list(self._registry),
            }

        try:
            result = tool(**args)
        except TypeError as exc:
            return {"ok": False, "error": "invalid_input", "message": str(exc)}
        except Exception as exc:
            return {"ok": False, "error": "tool_error", "message": str(exc)}

        if isinstance(result, dict):
            return result
        return {"ok": True, "result": result}

    def parse_action(self, text: str) -> Optional[Action]:
        return parse_action(text)

    def parse_final_answer(self, text: str) -> Optional[str]:
        return parse_final_answer(text)

    def _generate(self, prompt: str) -> Any:
        try:
            return self.llm.generate(prompt, system_prompt=self.get_system_prompt())
        except TypeError as exc:
            if "system_prompt" not in str(exc):
                raise
            prompt_with_system = f"System: {self.get_system_prompt()}\n\n{prompt}"
            return self.llm.generate(prompt_with_system)

    @staticmethod
    def _sanitize(value: Any) -> Any:
        secret_words = ("api_key", "apikey", "token", "secret", "password", "authorization")
        if isinstance(value, dict):
            return {
                key: "[REDACTED]" if any(secret in key.lower() for secret in secret_words) else ReActAgent._sanitize(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [ReActAgent._sanitize(item) for item in value]
        return value

    @staticmethod
    def _extract_content(result: Any) -> str:
        if isinstance(result, dict):
            return str(result.get("content", "")).strip()
        return str(result).strip()

    @staticmethod
    def _build_registry(tools: List[Dict[str, Any]]) -> Dict[str, Callable[..., Any]]:
        registry: Dict[str, Callable[..., Any]] = {}
        for tool in tools:
            function = tool.get("func") or tool.get("function")
            if callable(function):
                registry[tool["name"]] = function
        return registry


def _strip_code_fence(text: str) -> str:
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[-1].strip() == "```":
            return "\n".join(lines[1:-1]).strip()
    return text
