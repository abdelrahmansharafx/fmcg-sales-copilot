"""The chat page at "/": it is served, it is locked down, and it keeps the API key safe."""

import logging
import re

import pytest
from fastapi.testclient import TestClient

from scripts.demo_remote import STEPS
from src.api.main import create_app
from src.api.ui import PAGE, _hash, build_csp

KEY = "ui-test-key-0123456789abcdef"


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("API_KEY", KEY)
    return TestClient(create_app(), raise_server_exceptions=False)


def inline(tag: str, html: str) -> str:
    return re.search(rf"<{tag}>(.*?)</{tag}>", html, flags=re.DOTALL).group(1)


# --- it is served -------------------------------------------------------------------------------
def test_root_serves_the_chat_page_without_authentication(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "<title>FMCG Sales Copilot</title>" in response.text
    assert 'id="question"' in response.text and 'id="key-input"' in response.text
    assert "All data is synthetic and for demonstration only" in response.text


def test_page_works_even_when_no_api_key_is_configured(monkeypatch, client):
    monkeypatch.delenv("API_KEY")
    assert client.get("/").status_code == 200  # the page holds no data; only /ask needs the key


def test_the_page_is_not_part_of_the_api_schema_and_the_docs_still_work(client):
    assert "/" not in client.get("/openapi.json").json()["paths"]
    assert client.get("/docs").status_code == 200
    assert client.get("/health").status_code in (200, 503)


def test_page_requests_are_logged_like_any_other(client, caplog):
    with caplog.at_level(logging.INFO):
        client.get("/")
    entries = [
        r.fields for r in caplog.records if getattr(r, "fields", {}).get("event") == "request"
    ]
    assert entries[-1]["path"] == "/" and entries[-1]["status"] == 200


# --- the browser is locked down -------------------------------------------------------------------
def test_security_headers(client):
    headers = client.get("/").headers
    assert headers["x-content-type-options"] == "nosniff"
    assert headers["referrer-policy"] == "no-referrer"
    assert headers["x-frame-options"] == "DENY"


def test_csp_pins_the_exact_inline_script_and_style_and_blocks_other_origins(client):
    response = client.get("/")
    policy = response.headers["content-security-policy"]
    html = response.text
    assert _hash(inline("script", html)) in policy
    assert _hash(inline("style", html)) in policy
    assert "default-src 'none'" in policy
    assert "connect-src 'self'" in policy  # the key can only ever be sent to this server
    assert "frame-ancestors 'none'" in policy
    assert "unsafe-inline" not in policy and "unsafe-eval" not in policy
    assert "script-src 'self'" not in policy and "*" not in policy


def test_csp_changes_when_the_page_changes():
    """The policy is derived from the file, so the script cannot change without the policy."""
    html = PAGE.read_text(encoding="utf-8")
    assert build_csp(html) != build_csp(html.replace("use strict", "use  strict"))
    with pytest.raises(ValueError):
        build_csp(html + "<script>alert(1)</script>")  # a second inline script is refused


# --- the page cannot leak the key or run injected markup -------------------------------
@pytest.fixture(scope="module")
def page() -> str:
    return PAGE.read_text(encoding="utf-8")


def test_page_loads_nothing_from_other_origins(page):
    external = re.findall(r"""(?:src|href|action)\s*=\s*["']\s*(?:https?:)?//""", page)
    assert external == []
    assert "@import" not in page and "url(http" not in page


def test_csp_compatible_markup_only(page):
    assert not re.search(r"\son[a-z]+\s*=", page, flags=re.IGNORECASE)  # no onclick= handlers
    assert not re.search(
        r"<[^>]+\sstyle\s*=", page, flags=re.IGNORECASE
    )  # no style="..." attributes


def test_model_output_is_never_parsed_as_html(page):
    script = inline("script", page)
    for dangerous in (
        "innerHTML",
        "outerHTML",
        "insertAdjacentHTML",
        "document.write",
        "eval(",
        "new Function",
    ):
        assert dangerous not in script, dangerous
    assert script.count("textContent") >= 3


def test_the_key_goes_only_into_the_x_api_key_header(page):
    script = inline("script", page)
    code = re.sub(r"//.*", "", script)  # ignore comments

    # Remove every legitimate use of the `apiKey` variable; nothing else may remain.
    allowed = [
        r"let apiKey = \"\"",  # declaration
        r"apiKey = (?:value|loadKey\(\)|\"\")",  # assignments
        r"!apiKey",
        r"apiKey (?:!==|===) \"\"",  # checks
        r"\"X-API-Key\": apiKey",  # the one place it leaves the page: a request header
    ]
    remaining = code
    for pattern in allowed:
        remaining = re.sub(pattern, "", remaining)
    assert "apiKey" not in remaining, "unexpected use of the API key"
    assert code.count('"X-API-Key": apiKey') == 1

    assert "localStorage" in script
    assert "console." not in script  # nothing is written to the console
    assert not re.search(r"""[?&](api_?key|key|token)=""", script, flags=re.IGNORECASE)
    # Every network call targets this server, with a fixed relative path.
    calls = re.findall(r"""fetchWithTimeout\(\s*["'`]([^"'`]+)["'`]""", script)
    assert sorted(calls) == ["/ask", "/health"]
    assert "http://" not in script and "https://" not in script
    forbidden = ("sendBeacon", "XMLHttpRequest", "WebSocket", "document.cookie", "sessionStorage")
    for pattern in forbidden:
        assert pattern not in script, pattern


def test_the_input_for_the_key_is_a_password_field(page):
    assert re.search(r'<input[^>]+id="key-input"[^>]+type="password"', page)


# --- content ----------------------------------------------------------------------------
def test_example_questions_are_the_ones_the_demo_script_verifies(page):
    """The five clickable examples must be questions scripts/demo_remote.py already checks."""
    script = inline("script", page)
    examples = re.findall(r'question:\s*"([^"]+)"', script)
    assert len(examples) == 5
    verified = {step.body["question"] for step in STEPS if step.body and "question" in step.body}
    assert set(examples) <= verified
    labels = re.findall(r'label:\s*"([^"]+)"', script)
    assert labels == [
        "Forecast",
        "Order recommendation",
        "Promo impact",
        "Policy",
        "Not in the data",
    ]


def test_the_waking_up_message_is_there(page):
    assert "Waking up the server" in page and "45 seconds" in page


def test_the_waking_banner_can_never_start_a_second_timer(page):
    """Regression: two timers were started, only one was cleared, so the banner never went away."""
    script = inline("script", page)
    assert re.search(r"function startWakeBanner\(startedAt\) \{\s*if \(wakeTimer\) return;", script)
    assert script.count("setInterval(") == 1  # the only interval is the one guarded above
