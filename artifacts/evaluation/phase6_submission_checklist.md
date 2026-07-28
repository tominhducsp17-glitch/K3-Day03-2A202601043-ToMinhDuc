# Phase 6 Submission Checklist

## Consistency

- [x] Tool inventory in report matches `src/tools/tools.py`.
- [x] Success rate includes formula and raw outcomes.
- [x] Failed trace includes first divergence, root cause, fix, and regression test.
- [x] Traces contain no API key or PII.
- [x] Important claims include reproduction commands.

## Required Artifacts

- [x] Chatbot baseline: `src/chatbot/chatbot.py`
- [x] ReAct Agent V1: `src/agent/agent.py`
- [x] ReAct Agent V2: `src/agent/agent_v2.py`
- [x] Regression test: `tests/test_agent_recovery.py`
- [x] Five-case evaluation: `artifacts/evaluation/phase6_raw_results.json`
- [x] Flowchart: `artifacts/traces/react_v1_flowchart.md`
- [x] Success trace: `artifacts/traces/react_v1_success_trace.json`
- [x] Failed trace: `artifacts/traces/react_v1_failed_trace_malformed_args.json`
- [x] Recovery trace: `artifacts/traces/react_v2_recovery_trace_malformed_args.json`
- [x] Group report: `report/group_report/GROUP_REPORT.md`
- [x] Individual report: `report/individual_reports/INDIVIDUAL_REPORT.md`

## Security

- [x] `.env` is in `.gitignore`.
- [x] `.venv/`, `logs/`, `models/`, and `__pycache__/` are ignored.
- [x] No local LLM dependency is required by default.

## Reproduction Commands

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\run_lab_evaluation.py
.\.venv\Scripts\python.exe scripts\chat_gemini_baseline.py
```
