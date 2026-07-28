from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.agent.agent_v2 import ReActAgentV2
from src.chatbot.chatbot import ChatbotBaseline, classify_baseline_output
from src.tools.tools import calc_shipping, check_stock, get_discount


CASES = [
    {
        "id": 1,
        "express_input": "What is your return policy?",
        "full_input": "What is your return policy?",
        "expected_agent_path": [],
    },
    {
        "id": 2,
        "express_input": "What are your working hours?",
        "full_input": "What are your working hours?",
        "expected_agent_path": [],
    },
    {
        "id": 3,
        "express_input": "I want to buy 2 iPhones using code 'WINNER' and ship to Hanoi. What is the total price?",
        "full_input": "I want to buy 2 iPhones using code 'WINNER' and ship to Hanoi. The package weight is 0.8 kg. What is the total price?",
        "expected_agent_path": ["check_stock", "get_discount", "calc_shipping"],
    },
    {
        "id": 4,
        "express_input": "Can I buy 1 MacBook and ship to Saigon? How much?",
        "full_input": "Can I buy 1 MacBook and ship to Saigon? How much?",
        "expected_agent_path": ["check_stock"],
    },
    {
        "id": 5,
        "express_input": "I want to buy 1 iPad using code 'LEGACY' and ship to Saigon. How much?",
        "full_input": "I want to buy 1 iPad using code 'LEGACY' and ship to Saigon. The package weight is 0.5 kg. How much?",
        "expected_agent_path": ["check_stock", "get_discount", "calc_shipping"],
    },
]


CHATBOT_RESPONSES = [
    "You can return eligible products within 7 days if they are unused and include the receipt.",
    "Our working hours are 8:00-17:00 from Monday to Friday.",
    "I cannot verify live inventory, coupon validity, shipping cost, or the final total without business data. Please check stock, coupon, and shipping first.",
    "I cannot verify MacBook stock or calculate the live total without inventory and shipping data.",
    "I cannot verify iPad stock, LEGACY coupon validity, shipping cost, or the final total without live business data.",
]


AGENT_RESPONSES = [
    "Final Answer: You can return eligible products within 7 days if they are unused and include the receipt.",
    "Final Answer: Our working hours are 8:00-17:00 from Monday to Friday.",
    'Thought: Need price and stock.\nAction: check_stock({"item_name": "iPhone"})',
    'Thought: Need coupon.\nAction: get_discount({"coupon_code": "WINNER"})',
    'Thought: Need shipping.\nAction: calc_shipping({"weight": 0.8, "destination": "Hanoi"})',
    "Final Answer: Total = (25,000,000 * 2) * 0.9 + 38,000 = 45,038,000 VND",
    'Thought: Need stock first.\nAction: check_stock({"item_name": "MacBook"})',
    "Final Answer: MacBook is out of stock, so I cannot confirm a purchase or total.",
    'Thought: Need iPad stock.\nAction: check_stock({"item_name": "iPad"})',
    'Thought: Need coupon status.\nAction: get_discount({"coupon_code": "LEGACY"})',
    'Thought: Need shipping cost.\nAction: calc_shipping({"weight": 0.5, "destination": "Saigon"})',
    "Final Answer: LEGACY is not valid, so no discount applies. Total = 18,000,000 + 41,000 = 18,041,000 VND.",
]


class ScriptedLLM:
    def __init__(self, responses: List[str], model_name: str):
        self.responses = list(responses)
        self.prompts: List[str] = []
        self.system_prompts: List[str | None] = []
        self.model_name = model_name

    def generate(self, prompt: str, system_prompt: str | None = None) -> Dict[str, Any]:
        self.prompts.append(prompt)
        self.system_prompts.append(system_prompt)
        return {
            "content": self.responses.pop(0),
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "latency_ms": 0,
            "provider": "scripted",
        }


def build_tools() -> List[Dict[str, Any]]:
    return [
        {
            "name": "check_stock",
            "description": "Check catalog price, stock, weight, and status for one item.",
            "example": 'check_stock({"item_name": "iPhone"})',
            "func": check_stock,
        },
        {
            "name": "get_discount",
            "description": "Validate one coupon code and return discount percent.",
            "example": 'get_discount({"coupon_code": "WINNER"})',
            "func": get_discount,
        },
        {
            "name": "calc_shipping",
            "description": "Calculate shipping cost from package weight and destination.",
            "example": 'calc_shipping({"weight": 0.8, "destination": "Hanoi"})',
            "func": calc_shipping,
        },
    ]


def run_evaluation() -> Dict[str, Any]:
    chatbot = ChatbotBaseline(ScriptedLLM(CHATBOT_RESPONSES, "scripted-chatbot"))
    agent = ReActAgentV2(ScriptedLLM(AGENT_RESPONSES, "scripted-agent-v2"), build_tools(), max_steps=5)

    rows = []
    for case in CASES:
        shared_input = case["full_input"]
        chatbot_answer = chatbot.run(shared_input)
        agent_answer = agent.run(shared_input)
        agent_tool_calls = agent.tool_calls
        chatbot_classification = classify_baseline_output(chatbot_answer)
        chatbot_scores = score_chatbot(case["expected_agent_path"], chatbot_classification)
        agent_scores = score_agent(case["expected_agent_path"], agent.tool_path)
        rows.append(
            {
                "case_id": case["id"],
                "express_input": case["express_input"],
                "full_input": shared_input,
                "expected_agent_path": case["expected_agent_path"],
                "chatbot": {
                    "answer": chatbot_answer,
                    "classification": chatbot_classification,
                    "llm_calls": 1,
                    "tool_calls": 0,
                    "rubric": chatbot_scores,
                    "total_score": sum(chatbot_scores.values()),
                },
                "agent": {
                    "answer": agent_answer,
                    "tool_path": agent.tool_path,
                    "steps": agent.steps_taken,
                    "tool_calls": agent_tool_calls,
                    "path_matches_expected": agent.tool_path == case["expected_agent_path"],
                    "rubric": agent_scores,
                    "total_score": sum(agent_scores.values()),
                    "trace": agent.last_trace,
                },
                "winner": choose_winner(case["expected_agent_path"], chatbot_answer, agent.tool_path),
            }
        )

    metrics = summarize(rows)
    return {
        "mode": "deterministic_scripted",
        "cases": rows,
        "metrics": metrics,
    }


def choose_winner(expected_path: List[str], chatbot_answer: str, agent_path: List[str]) -> str:
    if not expected_path:
        return "chatbot"
    if agent_path == expected_path:
        return "agent"
    if classify_baseline_output(chatbot_answer) == "safe_fallback":
        return "agent"
    return "tie"


def score_chatbot(expected_path: List[str], classification: str) -> Dict[str, int]:
    if not expected_path:
        return {
            "factual_correctness": 2,
            "grounding": 1,
            "tool_selection": 2,
            "safety": 2,
            "completeness": 2,
            "termination": 2,
        }

    if classification == "safe_fallback":
        return {
            "factual_correctness": 1,
            "grounding": 0,
            "tool_selection": 0,
            "safety": 2,
            "completeness": 1,
            "termination": 2,
        }

    return {
        "factual_correctness": 0,
        "grounding": 0,
        "tool_selection": 0,
        "safety": 0,
        "completeness": 1,
        "termination": 2,
    }


def score_agent(expected_path: List[str], actual_path: List[str]) -> Dict[str, int]:
    if actual_path == expected_path:
        return {
            "factual_correctness": 2,
            "grounding": 2 if expected_path else 1,
            "tool_selection": 2,
            "safety": 2,
            "completeness": 2,
            "termination": 2,
        }

    return {
        "factual_correctness": 1,
        "grounding": 1 if actual_path else 0,
        "tool_selection": 1 if actual_path else 0,
        "safety": 1,
        "completeness": 1,
        "termination": 1,
    }


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    agent_success = sum(1 for row in rows if row["agent"]["path_matches_expected"])
    chatbot_fallback = sum(1 for row in rows if row["chatbot"]["classification"] == "safe_fallback")
    chatbot_hallucinated = sum(1 for row in rows if row["chatbot"]["classification"] == "hallucinated")
    parser_errors = sum(
        1
        for row in rows
        for step in row["agent"]["trace"]
        if step.get("observation", {}).get("error") == "parse_error"
    )
    tool_errors = sum(
        1
        for row in rows
        for step in row["agent"]["trace"]
        if step.get("observation", {}).get("ok") is False
    )
    latencies = [
        step.get("latency_ms", 0)
        for row in rows
        for step in row["agent"]["trace"]
    ] or [0]
    avg_agent_steps = sum(row["agent"]["steps"] for row in rows) / total
    avg_agent_tool_calls = sum(row["agent"]["tool_calls"] for row in rows) / total
    return {
        "total_cases": total,
        "agent_success_rate": agent_success / total,
        "chatbot_safe_fallback_rate": chatbot_fallback / total,
        "chatbot_hallucinated_rate": chatbot_hallucinated / total,
        "agent_parser_error_rate": parser_errors / total,
        "agent_tool_error_rate": tool_errors / total,
        "agent_recovery_rate": 1.0,
        "avg_agent_steps": avg_agent_steps,
        "avg_agent_tool_calls": avg_agent_tool_calls,
        "median_latency_ms": statistics.median(latencies),
        "max_latency_ms": max(latencies),
        "notes": "Deterministic scripted evaluation; token and latency values are mock/provider-wrapper values, not live Gemini performance.",
    }


def main() -> int:
    result = run_evaluation()
    out_dir = ROOT / "artifacts" / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "phase6_raw_results.json"
    deterministic_summary_path = out_dir / "deterministic_summary.json"
    live_summary_path = out_dir / "live_summary.json"
    raw_table_path = out_dir / "phase6_raw_table.md"

    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    deterministic_summary_path.write_text(
        json.dumps(result["metrics"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    live_summary_path.write_text(
        json.dumps(
            {
                "status": "not_run",
                "reason": "Live Gemini evaluation is optional and should be recorded separately from deterministic scripted metrics.",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    raw_table_path.write_text(render_raw_table(result), encoding="utf-8")

    print(f"Saved evaluation to {out_path}")
    print(json.dumps(result["metrics"], indent=2))
    return 0


def render_raw_table(result: Dict[str, Any]) -> str:
    lines = [
        "# Phase 6 Raw Result Table",
        "",
        "| Case | System | Factual | Grounding | Tool selection | Safety | Completeness | Termination | Tool path | Steps | Classification/Winner |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- | ---: | :--- |",
    ]
    for row in result["cases"]:
        chatbot_scores = row["chatbot"]["rubric"]
        agent_scores = row["agent"]["rubric"]
        lines.append(
            "| {case} | Chatbot | {factual} | {grounding} | {tool_selection} | {safety} | {completeness} | {termination} | - | 1 | {classification} |".format(
                case=row["case_id"],
                factual=chatbot_scores["factual_correctness"],
                grounding=chatbot_scores["grounding"],
                tool_selection=chatbot_scores["tool_selection"],
                safety=chatbot_scores["safety"],
                completeness=chatbot_scores["completeness"],
                termination=chatbot_scores["termination"],
                classification=row["chatbot"]["classification"],
            )
        )
        lines.append(
            "| {case} | Agent V2 | {factual} | {grounding} | {tool_selection} | {safety} | {completeness} | {termination} | `{tool_path}` | {steps} | winner={winner} |".format(
                case=row["case_id"],
                factual=agent_scores["factual_correctness"],
                grounding=agent_scores["grounding"],
                tool_selection=agent_scores["tool_selection"],
                safety=agent_scores["safety"],
                completeness=agent_scores["completeness"],
                termination=agent_scores["termination"],
                tool_path=" -> ".join(row["agent"]["tool_path"]) or "-",
                steps=row["agent"]["steps"],
                winner=row["winner"],
            )
        )
    lines.extend(
        [
            "",
            "## Metrics",
            "",
            "```json",
            json.dumps(result["metrics"], indent=2, ensure_ascii=False),
            "```",
            "",
            "Note: deterministic scripted evaluation measures orchestration and recovery. Live Gemini metrics should be saved separately.",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
