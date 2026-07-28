from src.tools.tools import calc_shipping, check_stock, get_discount, get_tool_registry, web_research


def test_check_stock_returns_catalog_data_for_known_item():
    result = check_stock("iPhone")

    assert result == {
        "ok": True,
        "item_name": "iPhone",
        "price": 25_000_000,
        "stock": 15,
        "weight_kg": 0.4,
        "status": "in_stock",
    }


def test_check_stock_returns_structured_error_for_unknown_item():
    result = check_stock("AirPods")

    assert result["ok"] is False
    assert result["error"] == "item_not_found"
    assert "AirPods" in result["message"]


def test_check_stock_returns_structured_error_for_missing_argument():
    result = check_stock()

    assert result["ok"] is False
    assert result["error"] == "invalid_input"


def test_get_discount_returns_valid_coupon_data():
    result = get_discount("winner")

    assert result == {
        "ok": True,
        "coupon_code": "WINNER",
        "discount_percent": 10,
        "valid": True,
    }


def test_get_discount_returns_invalid_coupon_without_crashing():
    result = get_discount("LEGACY")

    assert result == {
        "ok": True,
        "coupon_code": "LEGACY",
        "discount_percent": 0,
        "valid": False,
    }


def test_get_discount_returns_structured_error_for_unknown_coupon():
    result = get_discount("NOPE")

    assert result["ok"] is False
    assert result["error"] == "coupon_not_found"


def test_get_discount_returns_structured_error_for_missing_argument():
    result = get_discount()

    assert result["ok"] is False
    assert result["error"] == "invalid_input"


def test_calc_shipping_returns_cost_for_supported_destination():
    result = calc_shipping(weight=0.8, destination="Hanoi")

    assert result == {
        "ok": True,
        "weight": 0.8,
        "destination": "Hanoi",
        "shipping_cost": 38_000,
        "estimated_days": 1,
    }


def test_calc_shipping_returns_structured_error_for_bad_weight():
    result = calc_shipping(weight=0, destination="Hanoi")

    assert result["ok"] is False
    assert result["error"] == "invalid_input"


def test_calc_shipping_returns_structured_error_for_missing_destination():
    result = calc_shipping(weight=1)

    assert result["ok"] is False
    assert result["error"] == "invalid_input"


def test_calc_shipping_returns_structured_error_for_unsupported_destination():
    result = calc_shipping(weight=1, destination="Hue")

    assert result["ok"] is False
    assert result["error"] == "destination_not_supported"


def test_tools_are_deterministic_for_same_input():
    assert check_stock("iPad") == check_stock("iPad")
    assert get_discount("STUDENT") == get_discount("STUDENT")
    assert calc_shipping(0.5, "Saigon") == calc_shipping(0.5, "Saigon")


def test_tool_registry_exposes_all_phase_three_tools():
    registry = get_tool_registry()

    assert set(registry) == {"check_stock", "get_discount", "calc_shipping"}
    assert registry["check_stock"](item_name="MacBook")["status"] == "out_of_stock"


def test_web_research_parses_search_results(monkeypatch):
    class FakeResponse:
        text = """
        <html>
          <body>
            <a class="result__a" href="/l/?uddg=https%3A%2F%2Fexample.com%2Fagent">AI Agents</a>
            <a class="result__snippet">A practical article about AI agents.</a>
          </body>
        </html>
        """

        def raise_for_status(self):
            return None

    def fake_get(url, params, headers, timeout):
        assert "duckduckgo.com" in url
        assert params["q"] == "AI agents"
        return FakeResponse()

    monkeypatch.setattr("src.tools.tools.requests.get", fake_get)

    result = web_research("AI agents", max_results=3)

    assert result["ok"] is True
    assert result["results"][0] == {
        "title": "AI Agents",
        "url": "https://example.com/agent",
        "snippet": "A practical article about AI agents.",
    }
