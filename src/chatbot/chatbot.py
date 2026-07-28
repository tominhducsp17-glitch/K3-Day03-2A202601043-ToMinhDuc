from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.core.llm_provider import LLMProvider


ECOMMERCE_BASELINE_SYSTEM_PROMPT = """
You are a helpful e-commerce support chatbot.

You answer static store-policy questions clearly and concisely.

You do not have access to live inventory, prices, coupon validation,
shipping calculators, databases, or tools. For questions that require
live business evidence, say what information would need to be checked
and do not invent prices, stock status, discounts, shipping costs, or
order totals. Do not claim that a coupon, purchase, shipment, or any
other action has been applied, placed, booked, or completed.

Return only the final customer-facing answer. Do not write Thought,
Action, Observation, tool calls, or JSON.
""".strip()


class ChatbotBaseline:
    """
    One-call chatbot baseline for Lab 03 Phase 2.

    This class deliberately does not accept tools or a tool registry. It
    provides a fair baseline against the later ReAct agent: system prompt
    plus user message plus optional text history goes to exactly one LLM call.
    """

    def __init__(
        self,
        llm: LLMProvider,
        system_prompt: str = ECOMMERCE_BASELINE_SYSTEM_PROMPT,
        keep_history: bool = True,
    ):
        self.llm = llm
        self.system_prompt = system_prompt
        self.keep_history = keep_history
        self.history: List[Dict[str, str]] = []
        self.llm_calls = 0
        self.tool_calls = 0

    def run(self, user_message: str) -> str:
        prompt = self._build_prompt(user_message)
        result = self.llm.generate(prompt, system_prompt=self.system_prompt)
        self.llm_calls += 1

        answer = self._extract_content(result)
        if self.keep_history:
            self.history.append({"role": "user", "content": user_message})
            self.history.append({"role": "assistant", "content": answer})

        return answer

    def reset(self) -> None:
        self.history.clear()
        self.llm_calls = 0
        self.tool_calls = 0

    def _build_prompt(self, user_message: str) -> str:
        if not self.keep_history or not self.history:
            return f"User: {user_message}"

        transcript = "\n".join(
            f"{turn['role'].title()}: {turn['content']}" for turn in self.history
        )
        return f"{transcript}\nUser: {user_message}"

    @staticmethod
    def _extract_content(result: Any) -> str:
        if isinstance(result, dict):
            return str(result.get("content", "")).strip()
        return str(result).strip()


def classify_baseline_output(answer: str) -> str:
    """
    Coarse Phase 2 label for a chatbot answer.

    The labels are intentionally simple because Phase 2 is about inspecting
    baseline behavior, not claiming model-independent truth from tests.
    """

    normalized = answer.lower()
    fallback_markers = [
        "không có quyền truy cập",
        "khong co quyen truy cap",
        "không thể xác minh",
        "khong the xac minh",
        "không có dữ liệu",
        "khong co du lieu",
        "cannot verify",
        "do not have access",
        "don't have access",
        "need to check",
    ]
    hallucination_markers = [
        "tổng là",
        "tong la",
        "total is",
        "shipping cost is",
        "discount is valid",
        "coupon applied",
        "order placed",
        "shipment booked",
        "has been applied",
        "has been placed",
        "has been booked",
        "còn hàng",
        "con hang",
    ]

    if any(marker in normalized for marker in fallback_markers):
        return "safe_fallback"
    if any(marker in normalized for marker in hallucination_markers):
        return "hallucinated"
    return "correct"
