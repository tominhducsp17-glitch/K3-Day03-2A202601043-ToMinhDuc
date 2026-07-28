# Phase 5 RCA - Malformed Args

| Item | Content |
| --- | --- |
| User input | `Is iPhone available?` |
| Expected path | `check_stock` |
| Actual path in V1 | No tool call; parser emitted `parse_error`. |
| First divergence | Step 1: LLM produced `Action: check_stock({'item_name': 'iPhone'})`; V1 expected strict JSON only. |
| Error class | Parser |
| Root cause | `parse_action` used `json.loads` only, so single-quoted dict args could not be parsed even though the tool name and argument structure were recoverable. |
| Smallest fix | Keep JSON as the primary contract, then in V2 fall back to `ast.literal_eval` for safe Python literal dicts. |
| Regression test | `tests/test_agent_recovery.py::test_malformed_args_failed_trace_v1_vs_v2_before_after` |
| Before metric | V1: `tool_calls = 0`, `actual_path = []`, status `failed`. |
| After metric | V2: `tool_calls = 1`, `actual_path = ["check_stock"]`, status `recovered`. |
