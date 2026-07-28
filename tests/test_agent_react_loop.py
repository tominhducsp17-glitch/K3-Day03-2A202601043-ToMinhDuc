from src.agent.agent import ReActAgent, parse_action, parse_final_answer
from src.tools.tools import calc_shipping, check_stock, get_discount


class ScriptedLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []
        self.system_prompts = []
        self.model_name = "scripted-react"

    def generate(self, prompt, system_prompt=None):
        self.prompts.append(prompt)
        self.system_prompts.append(system_prompt)
        return {
            "content": self.responses.pop(0),
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "latency_ms": 0,
            "provider": "fake",
        }


class MinimalScriptedLLM:
    def __init__(self, responses):
        self._iter = iter(responses)
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return next(self._iter)


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


def test_parse_action_extracts_tool_name_and_json_args():
    parsed = parse_action('Thought: Need price.\nAction: check_stock({"item_name": "iPhone"})')

    assert parsed == ("check_stock", {"item_name": "iPhone"})


def test_parse_action_accepts_whitespace_and_multiline_json():
    text = """
Thought: Need shipping.
Action:
  calc_shipping(
    {
      "weight": 0.8,
      "destination": "Hanoi"
    }
  )
"""

    parsed = parse_action(text)

    assert parsed == ("calc_shipping", {"weight": 0.8, "destination": "Hanoi"})


def test_parse_action_accepts_json_code_fence():
    text = """Thought: Need coupon.
Action: get_discount(```json
{"coupon_code": "WINNER"}
```)
"""

    parsed = parse_action(text)

    assert parsed == ("get_discount", {"coupon_code": "WINNER"})


def test_parse_action_returns_none_for_malformed_json():
    parsed = parse_action("Thought: Need stock.\nAction: check_stock({'item_name': 'iPhone'})")

    assert parsed is None


def test_parse_action_returns_none_when_action_missing():
    parsed = parse_action("Thought: I should answer directly.")

    assert parsed is None


def test_parse_final_answer_extracts_answer_text():
    parsed = parse_final_answer("Thought: done.\nFinal Answer: Total is 45,038,000 VND")

    assert parsed == "Total is 45,038,000 VND"


def test_react_loop_executes_three_tools_and_returns_final_answer():
    llm = ScriptedLLM(
        [
            'Thought: Need stock and price.\nAction: check_stock({"item_name": "iPhone"})',
            'Thought: Need coupon.\nAction: get_discount({"coupon_code": "WINNER"})',
            'Thought: Need shipping.\nAction: calc_shipping({"weight": 0.8, "destination": "Hanoi"})',
            "Final Answer: Total = (25,000,000 * 2) * 0.9 + 38,000 = 45,038,000 VND",
        ]
    )
    agent = ReActAgent(llm=llm, tools=build_tools(), max_steps=5)

    answer = agent.run("2 iPhone + WINNER + Hanoi; weight 0.8 kg. Total?")

    assert answer == "Total = (25,000,000 * 2) * 0.9 + 38,000 = 45,038,000 VND"
    assert agent.steps_taken == 4
    assert agent.tool_calls == 3
    assert len(agent.last_trace) == 4
    assert agent.last_trace[0]["observation"]["price"] == 25_000_000
    assert agent.last_trace[1]["observation"]["discount_percent"] == 10
    assert agent.last_trace[2]["observation"]["shipping_cost"] == 38_000


def test_observation_is_included_before_next_llm_step():
    llm = ScriptedLLM(
        [
            'Thought: Need stock.\nAction: check_stock({"item_name": "iPad"})',
            "Final Answer: iPad is in stock.",
        ]
    )
    agent = ReActAgent(llm=llm, tools=build_tools(), max_steps=3)

    agent.run("Can I buy an iPad?")

    assert len(llm.prompts) == 2
    assert "Observation:" in llm.prompts[1]
    assert '"item_name": "iPad"' in llm.prompts[1]
    assert '"status": "in_stock"' in llm.prompts[1]


def test_react_agent_supports_minimal_scripted_llm_from_lab_guide():
    llm = MinimalScriptedLLM(
        [
            'Thought: Need stock.\nAction: check_stock({"item_name": "iPhone"})',
            "Final Answer: iPhone is in stock.",
        ]
    )
    agent = ReActAgent(llm=llm, tools=build_tools(), max_steps=3)

    answer = agent.run("Is iPhone available?")

    assert answer == "iPhone is in stock."
    assert len(llm.prompts) == 2
    assert llm.prompts[0].startswith("System:")
    assert "Observation:" in llm.prompts[1]


def test_unknown_tool_becomes_structured_observation_then_can_recover():
    llm = ScriptedLLM(
        [
            'Thought: Search product.\nAction: search_product({"q": "iPhone"})',
            'Thought: Use available tool.\nAction: check_stock({"item_name": "iPhone"})',
            "Final Answer: iPhone is in stock.",
        ]
    )
    agent = ReActAgent(llm=llm, tools=build_tools(), max_steps=4)

    answer = agent.run("Is iPhone available?")

    assert answer == "iPhone is in stock."
    assert agent.last_trace[0]["observation"]["error"] == "unknown_tool"
    assert "check_stock" in agent.last_trace[0]["observation"]["available_tools"]


def test_action_is_executed_before_final_answer_when_both_are_present():
    llm = ScriptedLLM(
        [
            'Thought: Need evidence.\nAction: check_stock({"item_name": "iPhone"})\nFinal Answer: It is available.',
            "Final Answer: iPhone is in stock.",
        ]
    )
    agent = ReActAgent(llm=llm, tools=build_tools(), max_steps=3)

    answer = agent.run("Is iPhone available?")

    assert answer == "iPhone is in stock."
    assert agent.tool_calls == 1
    assert "Observation:" in llm.prompts[1]


def test_telemetry_uses_phase_four_event_names(monkeypatch):
    events = []

    def capture(event_type, data):
        events.append((event_type, data))

    monkeypatch.setattr("src.agent.agent.logger.log_event", capture)
    llm = ScriptedLLM(
        [
            'Thought: Need stock.\nAction: check_stock({"item_name": "iPhone"})',
            "Final Answer: iPhone is in stock.",
        ]
    )
    agent = ReActAgent(llm=llm, tools=build_tools(), max_steps=3)

    agent.run("Is iPhone available?")

    event_names = [event[0] for event in events]
    assert "AGENT_START" in event_names
    assert "AGENT_STEP" in event_names
    assert "LLM_METRIC" in event_names
    assert "TOOL_CALL" in event_names
    assert "TOOL_RESULT" in event_names
    assert "AGENT_END" in event_names


def test_tool_error_event_is_logged_for_unknown_tool(monkeypatch):
    events = []

    def capture(event_type, data):
        events.append((event_type, data))

    monkeypatch.setattr("src.agent.agent.logger.log_event", capture)
    llm = ScriptedLLM(
        [
            'Thought: Search product.\nAction: search_product({"q": "iPhone"})',
            "Final Answer: I cannot use that tool.",
        ]
    )
    agent = ReActAgent(llm=llm, tools=build_tools(), max_steps=3)

    agent.run("Is iPhone available?")

    tool_error_events = [data for event_type, data in events if event_type == "TOOL_ERROR"]
    assert tool_error_events
    assert tool_error_events[0]["observation"]["error"] == "unknown_tool"


def test_agent_stops_with_safe_fallback_at_max_steps():
    llm = ScriptedLLM(
        [
            'Thought: Need stock.\nAction: check_stock({"item_name": "iPhone"})',
            'Thought: Need stock again.\nAction: check_stock({"item_name": "iPhone"})',
        ]
    )
    agent = ReActAgent(llm=llm, tools=build_tools(), max_steps=2)

    answer = agent.run("Loop test")

    assert "step limit" in answer
    assert agent.steps_taken == 2
    assert agent.tool_calls == 2
