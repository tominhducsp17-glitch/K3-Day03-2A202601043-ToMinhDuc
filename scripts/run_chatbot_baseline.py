from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.chatbot.chatbot import ChatbotBaseline, classify_baseline_output
from src.core.gemini_provider import GeminiProvider


def main() -> int:
    parser = argparse.ArgumentParser(description="Ask the Phase 2 chatbot baseline using Gemini.")
    parser.add_argument("question", nargs="*", help="Question to ask the chatbot baseline.")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    api_key = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("DEFAULT_MODEL", "gemini-1.5-flash")
    if not api_key or api_key == "your_gemini_api_key_here":
        print("Missing GEMINI_API_KEY in .env. Add your Gemini key first.")
        return 1

    question = " ".join(args.question).strip()
    if not question:
        question = input("Question: ").strip()
    if not question:
        print("No question provided.")
        return 1

    chatbot = ChatbotBaseline(GeminiProvider(model_name=model_name, api_key=api_key))
    answer = chatbot.run(question)

    print("\nAnswer:")
    print(answer)
    print(f"\nClassification: {classify_baseline_output(answer)}")
    print(f"LLM calls: {chatbot.llm_calls}")
    print(f"Tool calls: {chatbot.tool_calls}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
