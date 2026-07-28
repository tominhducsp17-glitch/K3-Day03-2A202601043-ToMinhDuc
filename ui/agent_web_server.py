from __future__ import annotations

import json
import os
import sys
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.agent.agent_v2 import ReActAgentV2
from src.core.gemini_provider import GeminiProvider
from src.tools.tools import calc_shipping, check_stock, get_discount, web_research


HOST = "127.0.0.1"
PORT = int(os.getenv("AGENT_WEB_PORT", "8765"))


class DemoAgentLLM:
    model_name = "demo-agent-v2"

    def generate(self, prompt: str, system_prompt: str | None = None) -> Dict[str, Any]:
        lower = prompt.lower()
        observation_count = lower.count("observation:")

        if "hôm nay" in lower or "hom nay" in lower or "today" in lower:
            return _llm_result(f"Final Answer: Hôm nay là {date.today().strftime('%d/%m/%Y')}.")
        if "return policy" in lower:
            return _llm_result("Final Answer: You can return eligible products within 7 days if they are unused and include the receipt.")
        if "working hours" in lower:
            return _llm_result("Final Answer: Our working hours are 8:00-17:00 from Monday to Friday.")
        if "macbook" in lower:
            if observation_count == 0:
                return _llm_result('Thought: Need stock first.\nAction: check_stock({"item_name": "MacBook"})')
            return _llm_result("Final Answer: MacBook is out of stock, so I cannot confirm a purchase or total.")
        if "ipad" in lower:
            if observation_count == 0:
                return _llm_result('Thought: Need iPad stock.\nAction: check_stock({"item_name": "iPad"})')
            if observation_count == 1:
                return _llm_result('Thought: Need coupon status.\nAction: get_discount({"coupon_code": "LEGACY"})')
            if observation_count == 2:
                return _llm_result('Thought: Need shipping cost.\nAction: calc_shipping({"weight": 0.5, "destination": "Saigon"})')
            return _llm_result("Final Answer: LEGACY is not valid, so no discount applies. Total = 18,000,000 + 41,000 = 18,041,000 VND.")
        if "iphone" in lower:
            if observation_count == 0:
                return _llm_result('Thought: Need price and stock.\nAction: check_stock({"item_name": "iPhone"})')
            if observation_count == 1:
                return _llm_result('Thought: Need coupon.\nAction: get_discount({"coupon_code": "WINNER"})')
            if observation_count == 2:
                return _llm_result('Thought: Need shipping.\nAction: calc_shipping({"weight": 0.8, "destination": "Hanoi"})')
            return _llm_result("Final Answer: Total = (25,000,000 * 2) * 0.9 + 38,000 = 45,038,000 VND")
        if "research" in lower or "search" in lower or "tìm kiếm" in lower or "tin tức" in lower:
            if observation_count == 0:
                return _llm_result('Thought: Need external evidence.\nAction: web_research({"query": "latest AI news", "max_results": 3})')
            return _llm_result("Final Answer: I found web results in the trace. Use Live Gemini planner for a real synthesized research answer.")

        return _llm_result("Final Answer: I can answer simple questions in demo mode. For broad research or arbitrary questions, switch to Live Gemini planner.")


def _llm_result(content: str) -> Dict[str, Any]:
    return {
        "content": content,
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "latency_ms": 0,
        "provider": "demo",
    }


def run_agent_chat(message: str, engine: str = "demo") -> Dict[str, Any]:
    load_dotenv(ROOT / ".env")
    llm = _make_llm(engine)
    agent = ReActAgentV2(llm=llm, tools=build_web_tools(), max_steps=6)
    answer = agent.run(message)
    return {
        "answer": answer,
        "engine": engine,
        "steps": agent.steps_taken,
        "tool_calls": agent.tool_calls,
        "tool_path": agent.tool_path,
        "trace": agent.last_trace,
    }


def _make_llm(engine: str):
    if engine == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key or api_key == "your_gemini_api_key_here":
            raise ValueError("Missing GEMINI_API_KEY in .env.")
        model_name = os.getenv("DEFAULT_MODEL", "gemini-1.5-flash")
        return GeminiProvider(model_name=model_name, api_key=api_key)
    return DemoAgentLLM()


def build_web_tools() -> List[Dict[str, Any]]:
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
        {
            "name": "web_research",
            "description": "Search the web for current or external information and return titles, URLs, and snippets.",
            "example": 'web_research({"query": "current Gemini API documentation", "max_results": 5})',
            "func": web_research,
        },
    ]


class AgentWebHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in {"/", "/index.html"}:
            self._send_file(ROOT / "ui" / "index.html", "text/html; charset=utf-8")
            return
        self._send_json({"error": "not_found"}, status=404)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/api/chat":
            self._send_json({"error": "not_found"}, status=404)
            return

        try:
            payload = self._read_json()
            message = str(payload.get("message", "")).strip()
            engine = str(payload.get("engine", "demo")).strip().lower()
            if not message:
                self._send_json({"error": "message_required"}, status=400)
                return

            result = run_agent_chat(message, engine=engine)
            self._send_json(result)
        except Exception as exc:
            self._send_json({"error": "server_error", "message": str(exc)}, status=500)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8")
        return json.loads(raw) if raw else {}

    def _send_file(self, path: Path, content_type: str) -> None:
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, payload: Dict[str, Any], status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> int:
    server = ThreadingHTTPServer((HOST, PORT), AgentWebHandler)
    print(f"Agent chat web server running at http://{HOST}:{PORT}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
