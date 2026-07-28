from scripts.run_lab_evaluation import run_evaluation


def test_phase6_evaluation_runs_five_cases_and_summarizes_metrics():
    result = run_evaluation()

    assert len(result["cases"]) == 5
    assert result["metrics"]["total_cases"] == 5
    assert result["metrics"]["agent_success_rate"] == 1.0
    assert result["metrics"]["chatbot_safe_fallback_rate"] == 0.6
    assert result["cases"][2]["agent"]["tool_path"] == [
        "check_stock",
        "get_discount",
        "calc_shipping",
    ]
