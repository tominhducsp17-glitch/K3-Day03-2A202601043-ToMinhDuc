from __future__ import annotations

from html.parser import HTMLParser
from typing import Any, Callable, Dict, List
from urllib.parse import parse_qs, unquote, urlparse

import requests


CATALOG: Dict[str, Dict[str, Any]] = {
    "iphone": {
        "display_name": "iPhone",
        "price": 25_000_000,
        "stock": 15,
        "weight_kg": 0.4,
    },
    "ipad": {
        "display_name": "iPad",
        "price": 18_000_000,
        "stock": 8,
        "weight_kg": 0.5,
    },
    "macbook": {
        "display_name": "MacBook",
        "price": 35_000_000,
        "stock": 0,
        "weight_kg": 2.0,
    },
}

COUPONS: Dict[str, Dict[str, Any]] = {
    "WINNER": {"discount_percent": 10, "valid": True},
    "STUDENT": {"discount_percent": 5, "valid": True},
    "LEGACY": {"discount_percent": 0, "valid": False},
}

SHIPPING_ZONES: Dict[str, Dict[str, int]] = {
    "hanoi": {"base_cost": 30_000, "per_kg_cost": 10_000, "estimated_days": 1},
    "ha noi": {"base_cost": 30_000, "per_kg_cost": 10_000, "estimated_days": 1},
    "saigon": {"base_cost": 35_000, "per_kg_cost": 12_000, "estimated_days": 2},
    "ho chi minh": {"base_cost": 35_000, "per_kg_cost": 12_000, "estimated_days": 2},
    "hcm": {"base_cost": 35_000, "per_kg_cost": 12_000, "estimated_days": 2},
    "danang": {"base_cost": 32_000, "per_kg_cost": 11_000, "estimated_days": 2},
    "da nang": {"base_cost": 32_000, "per_kg_cost": 11_000, "estimated_days": 2},
}


def check_stock(item_name: str | None = None) -> Dict[str, Any]:
    """
    Read-only tool.

    Input: item_name, required string.
    Success: price, stock, weight_kg, and inventory status.
    Error: structured invalid_input or item_not_found response.
    """

    if not isinstance(item_name, str) or not item_name.strip():
        return _error("invalid_input", "item_name is required and must be a non-empty string.")

    item = CATALOG.get(_normalize_key(item_name))
    if item is None:
        return _error("item_not_found", f"Item '{item_name}' was not found in the catalog.")

    status = "in_stock" if item["stock"] > 0 else "out_of_stock"
    return {
        "ok": True,
        "item_name": item["display_name"],
        "price": item["price"],
        "stock": item["stock"],
        "weight_kg": item["weight_kg"],
        "status": status,
    }


def get_discount(coupon_code: str | None = None) -> Dict[str, Any]:
    """
    Read-only tool.

    Input: coupon_code, required string.
    Success: validity and discount_percent.
    Error: structured invalid_input or coupon_not_found response.
    """

    if not isinstance(coupon_code, str) or not coupon_code.strip():
        return _error("invalid_input", "coupon_code is required and must be a non-empty string.")

    normalized_code = coupon_code.strip().upper()
    coupon = COUPONS.get(normalized_code)
    if coupon is None:
        return _error("coupon_not_found", f"Coupon '{coupon_code}' was not found.")

    return {
        "ok": True,
        "coupon_code": normalized_code,
        "discount_percent": coupon["discount_percent"],
        "valid": coupon["valid"],
    }


def calc_shipping(weight: float | int | None = None, destination: str | None = None) -> Dict[str, Any]:
    """
    Read-only tool.

    Input: weight in kg and destination, both required.
    Success: shipping_cost and estimated_days.
    Error: structured invalid_input or destination_not_supported response.
    """

    if weight is None:
        return _error("invalid_input", "weight is required.")
    if not isinstance(weight, (int, float)) or isinstance(weight, bool) or weight <= 0:
        return _error("invalid_input", "weight must be a positive number.")
    if not isinstance(destination, str) or not destination.strip():
        return _error("invalid_input", "destination is required and must be a non-empty string.")

    zone = SHIPPING_ZONES.get(_normalize_key(destination))
    if zone is None:
        return _error(
            "destination_not_supported",
            f"Destination '{destination}' is not supported.",
        )

    shipping_cost = int(zone["base_cost"] + float(weight) * zone["per_kg_cost"])
    return {
        "ok": True,
        "weight": float(weight),
        "destination": destination.strip(),
        "shipping_cost": shipping_cost,
        "estimated_days": zone["estimated_days"],
    }


def web_research(query: str | None = None, max_results: int = 5) -> Dict[str, Any]:
    """
    Read-only web research tool.

    Input: query, required string. max_results is capped at 8.
    Success: search result titles, URLs, and snippets.
    Error: structured invalid_input or network/search error response.
    """

    if not isinstance(query, str) or not query.strip():
        return _error("invalid_input", "query is required and must be a non-empty string.")
    if not isinstance(max_results, int) or isinstance(max_results, bool):
        return _error("invalid_input", "max_results must be an integer.")

    capped_results = max(1, min(max_results, 8))
    try:
        response = requests.get(
            "https://duckduckgo.com/html/",
            params={"q": query.strip()},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        return _error("research_unavailable", f"Web research request failed: {exc}")

    parser = _DuckDuckGoHTMLParser()
    parser.feed(response.text)
    parser.close()
    results = parser.results[:capped_results]
    if not results:
        return _error("no_results", f"No research results found for '{query}'.")

    return {
        "ok": True,
        "query": query.strip(),
        "source": "duckduckgo_html",
        "results": results,
    }


def get_tool_registry() -> Dict[str, Callable[..., Dict[str, Any]]]:
    return {
        "check_stock": check_stock,
        "get_discount": get_discount,
        "calc_shipping": calc_shipping,
    }


def _normalize_key(value: str) -> str:
    return " ".join(value.strip().lower().split())


def _error(code: str, message: str) -> Dict[str, Any]:
    return {"ok": False, "error": code, "message": message}


class _DuckDuckGoHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.results: List[Dict[str, str]] = []
        self._current: Dict[str, str] | None = None
        self._capture_title = False
        self._capture_snippet = False

    def handle_starttag(self, tag: str, attrs: List[tuple[str, str | None]]) -> None:
        attrs_dict = {name: value or "" for name, value in attrs}
        classes = attrs_dict.get("class", "")
        if tag == "a" and "result__a" in classes:
            self._append_current_if_ready()
            self._current = {"title": "", "url": _clean_duckduckgo_url(attrs_dict.get("href", "")), "snippet": ""}
            self._capture_title = True
        elif self._current is not None and tag in {"a", "div"} and "result__snippet" in classes:
            self._capture_snippet = True

    def handle_data(self, data: str) -> None:
        if self._current is None:
            return
        text = " ".join(data.split())
        if not text:
            return
        if self._capture_title:
            self._current["title"] = (self._current["title"] + " " + text).strip()
        elif self._capture_snippet:
            self._current["snippet"] = (self._current["snippet"] + " " + text).strip()

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._capture_title:
            self._capture_title = False
        elif tag in {"a", "div"} and self._capture_snippet:
            self._capture_snippet = False
            self._append_current_if_ready()

    def close(self) -> None:
        self._append_current_if_ready()
        super().close()

    def _append_current_if_ready(self) -> None:
        if self._current and self._current.get("title") and self._current.get("url"):
            self.results.append(self._current)
        self._current = None


def _clean_duckduckgo_url(raw_url: str) -> str:
    parsed = urlparse(raw_url)
    if parsed.path == "/l/":
        redirected = parse_qs(parsed.query).get("uddg", [""])[0]
        return unquote(redirected) or raw_url
    return raw_url
