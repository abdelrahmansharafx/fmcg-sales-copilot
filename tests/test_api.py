"""API tests: auth, status codes, request logging (latency + cost) and the /ask path.

The agent is replaced by a fake wherever a real LLM would be needed, so these run offline.
Tests that use the real data and model are skipped if they have not been built.
"""

import json
import logging
import os

import httpx
import openai
import pytest
import sqlalchemy as sa
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from src.agent import AgentError
from src.agent.graph import AgentResult, ToolCallRecord
from src.api.auth import key_id, key_is_valid
from src.api.logging_setup import JsonFormatter, clean_request_id
from src.api.main import create_app, get_agent_runner
from src.config import DB_URL, MODELS_DIR
from src.tools import forecast_demand, recommend_order

KEY = "test-key-0123456789abcdef"
DB_PATH = sa.engine.make_url(DB_URL).database
needs_data = pytest.mark.skipif(
    not (DB_PATH and os.path.exists(DB_PATH) and (MODELS_DIR / "model_meta.json").exists()),
    reason="needs data/sales.db and a trained model",
)
AUTH = {"X-API-Key": KEY}


@pytest.fixture()
def app(monkeypatch):
    monkeypatch.setenv("API_KEY", KEY)
    return create_app()  # a fresh app per test; the lifespan (warm-up) only runs in a real server


@pytest.fixture()
def client(app):
    return TestClient(app, raise_server_exceptions=False)


def fake_result(**overrides) -> AgentResult:
    """What run_agent would return for one question that used query_sales."""
    defaults = dict(
        question="q",
        answer="Cafe 115 dropped the most.",
        tool_calls=[
            ToolCallRecord("query_sales", {"question": "q"}, True, '{"sql": "SELECT 1"}', 0.4)
        ],
        llm_calls=3,
        prompt_tokens=5000,
        completion_tokens=500,
        seconds=4.2,
        hit_step_limit=False,
        model_name="test-model",
    )
    return AgentResult(**(defaults | overrides))


def use_runner(app, runner):
    app.dependency_overrides[get_agent_runner] = lambda: runner


def request_logs(caplog) -> list[dict]:
    """The structured fields of every 'request' log line."""
    return [r.fields for r in caplog.records if getattr(r, "fields", {}).get("event") == "request"]


# --- authentication ----------------------------------------------------------------------------
PROTECTED = [
    ("post", "/ask", {"json": {"question": "hello"}}),
    ("post", "/forecast", {"json": {"outlet_id": "O001", "sku_id": "S01"}}),
    ("get", "/recommend/O001", {}),
]


@pytest.mark.parametrize(("method", "path", "kwargs"), PROTECTED)
def test_protected_endpoints_reject_missing_and_wrong_keys(client, method, path, kwargs):
    missing = getattr(client, method)(path, **kwargs)
    assert missing.status_code == 401
    assert missing.headers["www-authenticate"] == "ApiKey"
    wrong = getattr(client, method)(path, headers={"X-API-Key": "wrong"}, **kwargs)
    assert wrong.status_code == 401
    assert wrong.json()["detail"] == missing.json()["detail"]  # does not reveal which part failed


def test_no_key_configured_fails_closed(monkeypatch, client):
    monkeypatch.delenv("API_KEY")
    response = client.get("/recommend/O001", headers=AUTH)
    assert response.status_code == 503  # never "let everyone in"
    assert client.get("/health").status_code in (200, 503)  # health stays reachable


def test_several_keys_allow_rotation(monkeypatch, client, app):
    monkeypatch.setenv("API_KEY", f"old-key-aaaaaaaaaaaaaaaa, {KEY}")
    use_runner(app, lambda q: fake_result())
    for key in ("old-key-aaaaaaaaaaaaaaaa", KEY):
        assert (
            client.post("/ask", json={"question": "hi"}, headers={"X-API-Key": key}).status_code
            == 200
        )
    assert (
        client.post("/ask", json={"question": "hi"}, headers={"X-API-Key": "third"}).status_code
        == 401
    )


def test_key_helpers():
    assert key_is_valid("abc", ["x", "abc"])
    assert not key_is_valid("abc", ["x", "abd"])
    assert not key_is_valid("", ["x"])
    label = key_id(KEY)
    assert len(label) == 8 and label not in KEY and KEY not in label


# --- /health -----------------------------------------------------------------------------------
@needs_data
def test_health_needs_no_key_and_reports_readiness(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"ok", "degraded"}
    assert body["database"] and body["model"]
    assert set(body) == {"status", "version", "database", "model", "policy_index", "llm_provider"}


def test_health_is_degraded_without_an_llm_and_unavailable_without_data(monkeypatch, client):
    monkeypatch.setattr("src.api.health.database_ok", lambda: True)
    monkeypatch.setattr("src.api.health.model_ok", lambda: True)
    monkeypatch.setenv("LLM_PROVIDER", "none")
    assert client.get("/health").json()["status"] == "degraded"
    monkeypatch.setattr("src.api.health.database_ok", lambda: False)
    response = client.get("/health")
    assert response.status_code == 503 and response.json()["status"] == "unavailable"


# --- /ask --------------------------------------------------------------------------------------
def test_ask_returns_answer_tools_usage_and_cost(client, app):
    use_runner(app, lambda question: fake_result(question=question))
    response = client.post("/ask", json={"question": "  who dropped?  "}, headers=AUTH)
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Cafe 115 dropped the most."
    assert body["tool_calls"] == [
        {
            "name": "query_sales",
            "args": {"question": "q"},
            "ok": True,
            "seconds": 0.4,
            "sql": "SELECT 1",
        }
    ]
    assert body["usage"]["llm_calls"] == 3
    assert body["usage"]["prompt_tokens"] == 5000
    assert body["usage"]["cost_usd"] == pytest.approx((5000 * 0.40 + 500 * 1.60) / 1e6)
    assert body["request_id"] == response.headers["x-request-id"]


@pytest.mark.parametrize("question", ["", "   ", "x" * 1001])
def test_ask_rejects_empty_and_oversized_questions(client, question):
    assert client.post("/ask", json={"question": question}, headers=AUTH).status_code == 422
    assert client.post("/ask", json={}, headers=AUTH).status_code == 422


def test_ask_passes_the_stripped_question_to_the_agent(client, app):
    seen = []
    use_runner(app, lambda q: seen.append(q) or fake_result())
    client.post("/ask", json={"question": "  hello  "}, headers=AUTH)
    assert seen == ["hello"]


def rate_limit_error() -> openai.RateLimitError:
    request = httpx.Request("POST", "https://example.test")
    return openai.RateLimitError("slow", response=httpx.Response(429, request=request), body=None)


def test_llm_failures_map_to_clean_http_errors(client, app):
    def broken(question):
        raise AgentError("Could not reach the LLM.")

    use_runner(app, broken)
    response = client.post("/ask", json={"question": "hi"}, headers=AUTH)
    assert response.status_code == 502 and "Could not reach the LLM" in response.json()["detail"]

    def limited(question):
        raise AgentError("rate limit") from rate_limit_error()

    use_runner(app, limited)
    response = client.post("/ask", json={"question": "hi"}, headers=AUTH)
    assert response.status_code == 429 and response.headers["retry-after"] == "30"


def test_unexpected_errors_return_500_without_leaking_details(client, app, caplog):
    def crash(question):
        raise RuntimeError("secret internal detail")

    use_runner(app, crash)
    with caplog.at_level(logging.INFO):
        response = client.post("/ask", json={"question": "hi"}, headers=AUTH)
    assert response.status_code == 500
    assert "secret internal detail" not in response.text
    assert response.json()["request_id"] == response.headers["x-request-id"]
    assert any(r.exc_info for r in caplog.records)  # the traceback is in the server log
    assert request_logs(caplog)[-1]["status"] == 500


class ScriptedModel:
    """A stand-in LLM: returns prepared messages in order (see tests/test_agent.py)."""

    def __init__(self, *replies):
        self.replies = list(replies)

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        return self.replies.pop(0)


@needs_data
def test_ask_end_to_end_with_real_tools_and_a_scripted_llm(client, monkeypatch, caplog):
    """Real agent graph + real forecast tool; only the LLM is scripted."""
    usage = {"input_tokens": 1000, "output_tokens": 100, "total_tokens": 1100}
    model = ScriptedModel(
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "forecast_demand",
                    "args": {"outlet_id": "O001", "sku_id": "S01", "weeks": 2},
                    "id": "c1",
                }
            ],
            usage_metadata=usage,
        ),
        AIMessage(content="About 166 units over two weeks.", usage_metadata=usage),
    )
    monkeypatch.setattr("src.agent.graph.llm.get_chat_model", lambda: model)
    with caplog.at_level(logging.INFO):
        response = client.post("/ask", json={"question": "forecast O001 S01"}, headers=AUTH)
    body = response.json()
    assert response.status_code == 200, body
    assert body["answer"] == "About 166 units over two weeks."
    assert [(t["name"], t["ok"]) for t in body["tool_calls"]] == [("forecast_demand", True)]
    assert body["usage"] == {
        "llm_calls": 2,
        "prompt_tokens": 2000,
        "completion_tokens": 200,
        "cost_usd": pytest.approx((2000 * 0.40 + 200 * 1.60) / 1e6, abs=1e-6),
    }
    log = request_logs(caplog)[-1]
    assert (log["prompt_tokens"], log["completion_tokens"]) == (2000, 200)
    assert log["tools"] == ["forecast_demand"]


# --- /forecast and /recommend -------------------------------------------------------------------
@needs_data
def test_forecast_matches_the_tool(client):
    response = client.post(
        "/forecast", json={"outlet_id": "O001", "sku_id": "S01", "weeks": 3}, headers=AUTH
    )
    assert response.status_code == 200
    assert response.json() == json.loads(forecast_demand("O001", "S01", 3).model_dump_json())


@needs_data
def test_forecast_backtest_date_is_passed_through(client):
    body = {"outlet_id": "O001", "sku_id": "S01", "weeks": 2, "as_of": "2025-09-22"}
    response = client.post("/forecast", json=body, headers=AUTH)
    assert response.status_code == 200
    assert response.json()["origin_week"] == "2025-09-22"


@pytest.mark.parametrize(
    "body",
    [
        {"outlet_id": "O001", "sku_id": "S01", "weeks": 0},
        {"outlet_id": "O001", "sku_id": "S01", "weeks": 9},
        {"outlet_id": "O 001; DROP", "sku_id": "S01"},
        {"outlet_id": "O001"},
        {"outlet_id": "O001", "sku_id": "S01", "as_of": "not-a-date"},
    ],
)
def test_forecast_validates_input(client, body):
    assert client.post("/forecast", json=body, headers=AUTH).status_code == 422


@needs_data
def test_unknown_ids_are_404(client):
    unknown_outlet = client.post(
        "/forecast", json={"outlet_id": "O999", "sku_id": "S01"}, headers=AUTH
    )
    assert unknown_outlet.status_code == 404
    assert "Unknown outlet_id" in unknown_outlet.json()["detail"]
    unknown_sku = client.post(
        "/forecast", json={"outlet_id": "O001", "sku_id": "S99"}, headers=AUTH
    )
    assert unknown_sku.status_code == 404
    assert client.get("/recommend/O999", headers=AUTH).status_code == 404


@needs_data
def test_recommend_matches_the_tool(client):
    response = client.get("/recommend/O010", headers=AUTH)
    assert response.status_code == 200
    assert response.json() == json.loads(recommend_order("O010").model_dump_json())


def test_recommend_rejects_malformed_ids(client):
    assert client.get("/recommend/O%20001", headers=AUTH).status_code == 422
    assert client.get("/recommend/" + "O" * 40, headers=AUTH).status_code == 422


def test_missing_model_is_503(client, monkeypatch):
    monkeypatch.setattr("src.forecast.predict.MODELS_DIR", MODELS_DIR / "does-not-exist")
    from src.forecast import predict

    predict.load_model.cache_clear()
    try:
        response = client.get("/recommend/O010", headers=AUTH)
    finally:
        predict.load_model.cache_clear()
    assert response.status_code == 503
    assert "Forecast model not found" in response.json()["detail"]


# --- request logging ---------------------------------------------------------------------------
def test_every_request_logs_one_structured_line_with_latency_and_cost(client, app, caplog):
    use_runner(app, lambda q: fake_result())
    with caplog.at_level(logging.DEBUG):
        client.post("/ask", json={"question": "who dropped?"}, headers=AUTH)
        client.get("/recommend/O%20001", headers=AUTH)  # a 422
    ask, bad = request_logs(caplog)
    assert ask["method"] == "POST" and ask["path"] == "/ask" and ask["status"] == 200
    assert ask["latency_ms"] >= 0
    assert (ask["llm_calls"], ask["prompt_tokens"], ask["completion_tokens"]) == (3, 5000, 500)
    assert ask["cost_usd"] == pytest.approx(0.0028, abs=1e-6)
    assert ask["tools"] == ["query_sales"] and ask["api_key_id"] == key_id(KEY)
    assert bad["status"] == 422 and bad["cost_usd"] == 0.0


def test_logs_never_contain_the_key_or_the_question(client, app, caplog):
    use_runner(app, lambda q: fake_result())
    with caplog.at_level(logging.DEBUG):
        client.post("/ask", json={"question": "confidential question text"}, headers=AUTH)
        client.post("/ask", json={"question": "x"}, headers={"X-API-Key": "wrong-key-value"})
    text = " ".join(JsonFormatter().format(r) for r in caplog.records if r.name.startswith("fmcg"))
    for secret in (KEY, "wrong-key-value", "confidential question text"):
        assert secret not in text
    assert request_logs(caplog)[0]["question_chars"] == len("confidential question text")


def test_failed_auth_is_logged_without_a_key_id(client, caplog):
    with caplog.at_level(logging.INFO):
        client.get("/recommend/O001")
    entry = request_logs(caplog)[-1]
    assert entry["status"] == 401 and "api_key_id" not in entry


def test_routes_are_logged_by_template_not_by_value(client, caplog):
    with caplog.at_level(logging.INFO):
        client.get("/recommend/O001", headers=AUTH)
        client.get("/recommend/O002", headers=AUTH)
    assert {entry["path"] for entry in request_logs(caplog)} == {"/recommend/{outlet_id}"}


def test_health_checks_do_not_spam_the_info_log(client, caplog):
    with caplog.at_level(logging.INFO):
        client.get("/health")
    assert request_logs(caplog) == []


def test_request_ids_are_echoed_when_safe_and_replaced_when_not(client):
    ok = client.get("/health", headers={"X-Request-ID": "demo-123"})
    assert ok.headers["x-request-id"] == "demo-123"
    for bad in ["has space", "x" * 100, "semi;colon"]:
        replaced = client.get("/health", headers={"X-Request-ID": bad})
        assert replaced.headers["x-request-id"] != bad
    generated = client.get("/health")
    assert len(generated.headers["x-request-id"]) == 16
    assert float(generated.headers["x-response-time-ms"]) >= 0


def test_clean_request_id():
    assert clean_request_id("abc-123_x.y") == "abc-123_x.y"
    assert clean_request_id("a\nb") is None and clean_request_id(None) is None


def test_json_formatter_outputs_parseable_lines_with_extra_fields_and_exceptions():
    logger = logging.getLogger("fmcg.test")
    try:
        raise ValueError("boom")
    except ValueError:
        record = logger.makeRecord(
            "fmcg.test", logging.ERROR, __file__, 1, "failed %s", ("x",), logging.sys.exc_info(),
            extra={"fields": {"request_id": "r1", "cost_usd": 0.5}},
        )  # fmt: skip
    line = JsonFormatter().format(record)
    data = json.loads(line)
    assert "\n" not in line
    assert data["message"] == "failed x" and data["level"] == "ERROR"
    assert data["request_id"] == "r1" and data["cost_usd"] == 0.5
    assert "ValueError: boom" in data["exception"]


# --- OpenAPI -----------------------------------------------------------------------------------
def test_openapi_documents_the_endpoints_and_the_api_key_scheme(client):
    spec = client.get("/openapi.json").json()
    assert set(spec["paths"]) == {"/ask", "/forecast", "/recommend/{outlet_id}", "/health"}
    scheme = next(iter(spec["components"]["securitySchemes"].values()))
    assert scheme == {
        "type": "apiKey",
        "in": "header",
        "name": "X-API-Key",
        "description": "Shared secret set in the API_KEY env var.",
    }
    assert "security" in spec["paths"]["/ask"]["post"]
    assert "security" not in spec["paths"]["/health"]["get"]
    assert {"401", "404", "422", "503"} <= set(spec["paths"]["/ask"]["post"]["responses"])
