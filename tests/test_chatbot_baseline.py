from src.chatbot.chatbot import ChatbotBaseline, classify_baseline_output


class ScriptedLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []
        self.system_prompts = []
        self.model_name = "scripted-baseline"

    def generate(self, prompt, system_prompt=None):
        self.prompts.append(prompt)
        self.system_prompts.append(system_prompt)
        return {
            "content": self.responses.pop(0),
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "latency_ms": 0,
            "provider": "fake",
        }


def test_static_policy_question_uses_one_llm_call_and_no_tools():
    llm = ScriptedLLM(
        [
            "Bạn có thể đổi trả trong 7 ngày nếu sản phẩm còn nguyên tem và hóa đơn."
        ]
    )
    chatbot = ChatbotBaseline(llm)

    answer = chatbot.run("Chính sách đổi trả là gì?")

    assert "đổi trả" in answer.lower()
    assert chatbot.llm_calls == 1
    assert chatbot.tool_calls == 0
    assert len(llm.prompts) == 1
    assert "Action:" not in llm.system_prompts[0]
    assert classify_baseline_output(answer) == "correct"


def test_multistep_purchase_question_falls_back_without_ground_truth():
    llm = ScriptedLLM(
        [
            "Mình không thể xác minh tồn kho, mã WINNER, phí giao hàng hoặc tổng tiền "
            "vì không có dữ liệu live. Cần kiểm tra giá iPhone, coupon và phí ship Hà Nội trước."
        ]
    )
    chatbot = ChatbotBaseline(llm)

    answer = chatbot.run("Tôi muốn mua 2 iPhone, dùng mã WINNER và giao tới Hà Nội. Tổng tiền là bao nhiêu?")

    assert chatbot.llm_calls == 1
    assert chatbot.tool_calls == 0
    assert classify_baseline_output(answer) == "safe_fallback"
    assert "Observation:" not in llm.prompts[0]


def test_classifier_labels_ungrounded_order_claim_as_hallucinated():
    answer = "Coupon applied. Total is 45,038,000 VND and the shipment booked."

    assert classify_baseline_output(answer) == "hallucinated"


def test_history_is_included_but_still_single_call_per_turn():
    llm = ScriptedLLM(["Chào bạn, mình có thể hỗ trợ.", "Bạn muốn hỏi thêm gì?"])
    chatbot = ChatbotBaseline(llm)

    chatbot.run("Xin chào")
    chatbot.run("Bạn giúp gì được?")

    assert chatbot.llm_calls == 2
    assert chatbot.tool_calls == 0
    assert "Assistant: Chào bạn" in llm.prompts[1]
