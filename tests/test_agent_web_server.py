from ui.agent_web_server import build_web_tools, run_agent_chat


def test_agent_web_server_demo_chat_returns_trace():
    result = run_agent_chat(
        "I want to buy 2 iPhones using code WINNER and ship to Hanoi. The package weight is 0.8 kg. What is the total price?",
        engine="demo",
    )

    assert result["answer"] == "Total = (25,000,000 * 2) * 0.9 + 38,000 = 45,038,000 VND"
    assert result["tool_path"] == ["check_stock", "get_discount", "calc_shipping"]
    assert result["tool_calls"] == 3
    assert result["steps"] == 4
    assert result["trace"][0]["observation"]["status"] == "in_stock"


def test_agent_web_tools_include_research_without_changing_lab_registry():
    tool_names = {tool["name"] for tool in build_web_tools()}

    assert "web_research" in tool_names
    assert {"check_stock", "get_discount", "calc_shipping"}.issubset(tool_names)


def test_agent_web_demo_answers_simple_date_question():
    result = run_agent_chat("Hôm nay ngày bao nhiêu?", engine="demo")

    assert "Hôm nay là" in result["answer"]
    assert result["tool_calls"] == 0
