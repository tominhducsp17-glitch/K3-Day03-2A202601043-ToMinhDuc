# Group Report: Lab 3 - Chatbot vs ReAct Agent

- **Team Name**: Lab 03 Team
- **Team Members**: [Tô Minh Đức - 2A202601043 , Phạm Văn Sâm - 2A202601837]
- **Deployment Date**: 2026-07-28

---

## 1. Executive Summary

This project compares a one-call e-commerce chatbot baseline with a ReAct agent that can call deterministic tools for inventory, coupon, and shipping evidence.

- **Evaluation mode**: deterministic scripted LLM, not live Gemini.
- **Command**: `.\.venv\Scripts\python.exe scripts\run_lab_evaluation.py`
- **Raw results**: `artifacts/evaluation/phase6_raw_results.json`
- **Agent success rate**: 5/5 = 100%
- **Chatbot safe-fallback rate**: 3/5 = 60%
- **Average agent steps**: 2.4
- **Average agent tool calls**: 1.4

Key outcome: the chatbot is simpler and sufficient for static Q&A, while the ReAct agent is stronger for multi-step questions that require grounded inventory, discount, and shipping evidence.

---

## 2. System Architecture & Tooling

### 2.1 ReAct Loop Implementation

The V1/V2 agent follows:

```text
User input -> call LLM -> parse Action or Final Answer
Action -> validate tool -> execute tool -> append Observation -> repeat
Final Answer or max_steps fallback -> stop
```

Flowchart artifact: `artifacts/traces/react_v1_flowchart.md`

### 2.2 Tool Definitions

| Tool Name | Input Format | Output | Use Case |
| :--- | :--- | :--- | :--- |
| `check_stock` | `{"item_name": "iPhone"}` | `price`, `stock`, `weight_kg`, `status` | Check product price and availability. |
| `get_discount` | `{"coupon_code": "WINNER"}` | `discount_percent`, `valid` | Validate coupon status. |
| `calc_shipping` | `{"weight": 0.8, "destination": "Hanoi"}` | `shipping_cost`, `estimated_days` | Calculate shipping cost. |

Tool registry source: `src/tools/tools.py`

### 2.3 LLM Providers Used

- **Live provider target**: Gemini via `src/core/gemini_provider.py`
- **Default env provider**: `DEFAULT_PROVIDER=google`
- **Deterministic tests**: scripted/fake LLMs to avoid API variance
- **Local LLM**: optional only; not required for this setup

---

## 3. Evaluation Results

### 3.1 Five Cases

| Case | Full input used for grading | Expected Agent Path | Chatbot Classification | Agent Path | Winner |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | What is your return policy? | `[]` | `correct` | `[]` | Chatbot |
| 2 | What are your working hours? | `[]` | `correct` | `[]` | Chatbot |
| 3 | 2 iPhones + WINNER + Hanoi + 0.8 kg | `check_stock -> get_discount -> calc_shipping` | `safe_fallback` | `check_stock -> get_discount -> calc_shipping` | Agent |
| 4 | 1 MacBook + Saigon | `check_stock` | `safe_fallback` | `check_stock` | Agent |
| 5 | 1 iPad + LEGACY + Saigon + 0.5 kg | `check_stock -> get_discount -> calc_shipping` | `safe_fallback` | `check_stock -> get_discount -> calc_shipping` | Agent |

### 3.2 Metrics

Formulae:

- Agent success rate = cases where `agent.tool_path == expected_agent_path` / total cases.
- Chatbot safe-fallback rate = chatbot outputs classified `safe_fallback` / total cases.
- Average steps = sum agent steps / total cases.
- Average tool calls = sum agent tool calls / total cases.

Deterministic results:

| Metric | Value |
| :--- | :--- |
| Total cases | 5 |
| Agent success rate | 1.0 |
| Chatbot safe-fallback rate | 0.6 |
| Chatbot hallucinated rate | 0.0 |
| Agent parser-error rate | 0.0 |
| Agent tool-error rate | 0.0 |
| Agent recovery rate | 1.0 |
| Average agent steps | 2.4 |
| Average agent tool calls | 1.4 |
| Median latency | 0 ms scripted |
| Max latency | 0 ms scripted |

The latency and token values are scripted/mock values. A live Gemini run should be recorded separately and not mixed into this deterministic success rate.

---

## 4. Root Cause Analysis

Failed trace artifact: `artifacts/traces/react_v1_failed_trace_malformed_args.json`

RCA worksheet: `artifacts/traces/phase5_rca_malformed_args.md`

Summary:

- **Failure**: malformed Action args with single quotes.
- **First divergence**: V1 parser rejected `Action: check_stock({'item_name': 'iPhone'})`.
- **Root cause**: V1 used strict `json.loads` only.
- **Smallest fix**: V2 keeps JSON as primary contract, then safely falls back to `ast.literal_eval` for dict literals.
- **Regression test**: `tests/test_agent_recovery.py::test_malformed_args_failed_trace_v1_vs_v2_before_after`
- **Before/after**: V1 tool calls 0 -> V2 tool calls 1.

---

## 5. Production Readiness Review

- **Security**: `.env` is ignored; no API key should be committed.
- **Guardrails**: `max_steps` prevents infinite loops.
- **Tool safety**: tools return structured `ok=false` errors for invalid input.
- **Telemetry**: agent logs `AGENT_START`, `AGENT_STEP`, `TOOL_CALL`, `TOOL_RESULT`, `TOOL_ERROR`, `AGENT_PARSE_ERROR`, `LLM_METRIC`, and `AGENT_END`.
- **Known limitation**: deterministic scripted tests validate orchestration, not live model behavior. Live Gemini traces should be stored separately if used.

---

## 6. Reproduction Commands

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\run_lab_evaluation.py
.\.venv\Scripts\python.exe -m pytest tests\test_agent_recovery.py -q
```
