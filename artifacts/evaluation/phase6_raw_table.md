# Phase 6 Raw Result Table

| Case | System | Factual | Grounding | Tool selection | Safety | Completeness | Termination | Tool path | Steps | Classification/Winner |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- | ---: | :--- |
| 1 | Chatbot | 2 | 1 | 2 | 2 | 2 | 2 | - | 1 | correct |
| 1 | Agent V2 | 2 | 1 | 2 | 2 | 2 | 2 | `-` | 1 | winner=chatbot |
| 2 | Chatbot | 2 | 1 | 2 | 2 | 2 | 2 | - | 1 | correct |
| 2 | Agent V2 | 2 | 1 | 2 | 2 | 2 | 2 | `-` | 1 | winner=chatbot |
| 3 | Chatbot | 1 | 0 | 0 | 2 | 1 | 2 | - | 1 | safe_fallback |
| 3 | Agent V2 | 2 | 2 | 2 | 2 | 2 | 2 | `check_stock -> get_discount -> calc_shipping` | 4 | winner=agent |
| 4 | Chatbot | 1 | 0 | 0 | 2 | 1 | 2 | - | 1 | safe_fallback |
| 4 | Agent V2 | 2 | 2 | 2 | 2 | 2 | 2 | `check_stock` | 2 | winner=agent |
| 5 | Chatbot | 1 | 0 | 0 | 2 | 1 | 2 | - | 1 | safe_fallback |
| 5 | Agent V2 | 2 | 2 | 2 | 2 | 2 | 2 | `check_stock -> get_discount -> calc_shipping` | 4 | winner=agent |

## Metrics

```json
{
  "total_cases": 5,
  "agent_success_rate": 1.0,
  "chatbot_safe_fallback_rate": 0.6,
  "chatbot_hallucinated_rate": 0.0,
  "agent_parser_error_rate": 0.0,
  "agent_tool_error_rate": 0.0,
  "agent_recovery_rate": 1.0,
  "avg_agent_steps": 2.4,
  "avg_agent_tool_calls": 1.4,
  "median_latency_ms": 0.0,
  "max_latency_ms": 0,
  "notes": "Deterministic scripted evaluation; token and latency values are mock/provider-wrapper values, not live Gemini performance."
}
```

Note: deterministic scripted evaluation measures orchestration and recovery. Live Gemini metrics should be saved separately.
