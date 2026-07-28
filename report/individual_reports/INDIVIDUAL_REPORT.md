# Individual Report: Lab 3 - Chatbot vs ReAct Agent

- **Student Name**: To Minh Duc
- **Student ID**: 2A202601043
- **Date**: 2026-07-28

---

## I. Technical Contribution

Implemented the main deterministic lab pipeline:

- `src/chatbot/chatbot.py`: one-call chatbot baseline with no tools.
- `src/tools/tools.py`: deterministic e-commerce tools.
- `src/agent/agent.py`: ReAct Agent V1 with parser, executor, loop, max steps, and telemetry.
- `src/agent/agent_v2.py`: parser recovery improvement for malformed args.
- `scripts/run_lab_evaluation.py`: five-case deterministic evaluation.

The code separates baseline conversation, tool contracts, provider interface, agent orchestration, telemetry, and evaluation artifacts.

---

## II. Debugging Case Study

**Problem**: V1 failed to parse malformed arguments:

```text
Action: check_stock({'item_name': 'iPhone'})
```

**Log/trace source**:

- `artifacts/traces/react_v1_failed_trace_malformed_args.json`
- `artifacts/traces/phase5_rca_malformed_args.md`

**Diagnosis**: The model output was structurally recoverable, but V1 only accepted strict JSON via `json.loads`, so the agent generated `parse_error` and did not call `check_stock`.

**Solution**: V2 keeps strict JSON as the preferred contract, then falls back to `ast.literal_eval` for safe Python literal dictionaries.

**Regression test**:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_agent_recovery.py -q
```

Before/after metric: V1 `tool_calls = 0`; V2 `tool_calls = 1`.

---

## III. Personal Insights: Chatbot vs ReAct

1. **Reasoning**: The `Thought` block forces the agent to decompose multi-step requests into inventory, coupon, and shipping checks instead of answering from memory.
2. **Reliability**: The chatbot is better for static Q&A because it is cheaper and stops after one call. The agent adds overhead when no tools are needed.
3. **Observation**: Tool observations ground the next step. The model does not invent stock or shipping; the application inserts real tool output.

---

## IV. Future Improvements

- Add schema validation with Pydantic for tool arguments.
- Add repeated-action detection in Agent V2/V3.
- Add live Gemini evaluation as a separate artifact from deterministic results.
- Build a small UI after evaluation is stable, showing user input, tool path, observations, final answer, and status.
