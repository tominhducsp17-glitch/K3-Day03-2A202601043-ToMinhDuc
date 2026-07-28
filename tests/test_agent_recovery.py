from src.agent.agent import ReActAgent
from src.agent.agent_v2 import ReActAgentV2, parse_action_v2
from src.tools.tools import calc_shipping, check_stock, get_discount


class ScriptedLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []
        self.system_prompts = []
        self.model_name = "scripted-recovery"

    def generate(self, prompt, system_prompt=None):
        self.prompts.append(prompt)
        self.system_prompts.append(system_prompt)
        return {
            "content": self.responses.pop(0),
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "latency_ms": 0,
            "provider": "fake",
        }


def build_tools():
    return [
        {
            "name": "check_stock",
            "description": "Check catalog price, stock, and status for one item.",
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


def test_parse_action_v2_recovers_single_quoted_dict_args():
    parsed = parse_action_v2("Thought: Need stock.\nAction: check_stock({'item_name': 'iPhone'})")

    assert parsed == ("check_stock", {"item_name": "iPhone"})


def test_parse_action_v2_ignores_trailing_text_after_args():
    parsed = parse_action_v2(
        'Thought: Need coupon.\nAction: get_discount({"coupon_code": "WINNER"} trailing text)'
    )

    assert parsed == ("get_discount", {"coupon_code": "WINNER"})


def test_malformed_args_failed_trace_v1_vs_v2_before_after():
    malformed_output = "Thought: Need stock.\nAction: check_stock({'item_name': 'iPhone'})"

    v1_llm = ScriptedLLM([malformed_output, "Final Answer: I cannot verify stock."])
    v1 = ReActAgent(llm=v1_llm, tools=build_tools(), max_steps=2)
    v1_answer = v1.run("Is iPhone available?")

    assert v1_answer == "I cannot verify stock."
    assert v1.tool_calls == 0
    assert v1.tool_path == []
    assert v1.last_trace[0]["observation"]["error"] == "parse_error"

    v2_llm = ScriptedLLM([malformed_output, "Final Answer: iPhone is in stock."])
    v2 = ReActAgentV2(llm=v2_llm, tools=build_tools(), max_steps=2)
    v2_answer = v2.run("Is iPhone available?")

    assert v2_answer == "iPhone is in stock."
    assert v2.tool_calls == 1
    assert v2.tool_path == ["check_stock"]
    assert v2.last_trace[0]["observation"]["status"] == "in_stock"
    assert "Observation:" in v2_llm.prompts[1]
