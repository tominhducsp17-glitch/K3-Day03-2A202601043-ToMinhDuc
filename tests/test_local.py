import os
import sys

import pytest
from dotenv import load_dotenv


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytest.importorskip("llama_cpp", reason="Local GGUF support is optional for this lab setup.")

from src.core.local_provider import LocalProvider


def test_local_phi3():
    load_dotenv()
    model_path = os.getenv("LOCAL_MODEL_PATH", "./models/Phi-3-mini-4k-instruct-q4.gguf")

    if not os.path.exists(model_path):
        pytest.skip(f"Local model file not found at {model_path}; Gemini is the default provider.")

    provider = LocalProvider(model_path=model_path)
    prompt = "Explain what an AI Agent is in one sentence."
    chunks = list(provider.stream(prompt))

    assert "".join(chunks).strip()


if __name__ == "__main__":
    test_local_phi3()
