from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.chatbot.chatbot import ChatbotBaseline, classify_baseline_output
from src.core.gemini_provider import GeminiProvider


def main() -> int:
    load_dotenv(ROOT / ".env")

    api_key = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("DEFAULT_MODEL", "gemini-1.5-flash")
    if not api_key or api_key == "your_gemini_api_key_here":
        print("Missing GEMINI_API_KEY in .env. Add your Gemini key first.")
        return 1

    chatbot = ChatbotBaseline(
        GeminiProvider(model_name=model_name, api_key=api_key),
        keep_history=True,
    )

    print(f"Gemini chatbot baseline is ready. Model: {model_name}")
    print("Type /reset to clear history, /exit to quit.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return 0

        if not question:
            continue
        if question.lower() in {"/exit", "exit", "quit"}:
            print("Bye.")
            return 0
        if question.lower() == "/reset":
            chatbot.reset()
            print("History cleared.\n")
            continue

        answer = chatbot.run(question)
        print(f"\nGemini: {answer}")
        print(
            f"[classification={classify_baseline_output(answer)}, "
            f"llm_calls={chatbot.llm_calls}, tool_calls={chatbot.tool_calls}]\n"
        )


if __name__ == "__main__":
    raise SystemExit(main())
