"""The remote demo script, run in-process against the real FastAPI app with a canned agent."""

import os

import httpx
import pytest
import sqlalchemy as sa
from fastapi.testclient import TestClient

from scripts.demo_remote import STEPS, check_auth, run_demo, wait_until_ready
from src.agent.graph import AgentResult
from src.api.main import create_app, get_agent_runner
from src.config import DB_URL, MODELS_DIR

KEY = "demo-test-key-0123456789"
DB_PATH = sa.engine.make_url(DB_URL).database
needs_data = pytest.mark.skipif(
    not (DB_PATH and os.path.exists(DB_PATH) and (MODELS_DIR / "model_meta.json").exists()),
    reason="needs data/sales.db and a trained model",
)

GOOD_ANSWERS = {
    "Forecast": "About 357 units over 4 weeks; typical error is about 34%.",
    "order": "O010 should order Coca-Cola Zero 330ml can first.",
    "promotion 2": "Uplift was 35.4% (95% confidence interval 27.0% to 45.4%).",
    "30% discount": "No: above 25% needs Commercial Director approval; request 3 weeks ahead.",
    "Alexandria": "Alexandria Cafe 115 dropped the most.",
    "Pepsi": "I don't know: the data has no competitor information.",
}


def canned_agent(overrides: dict[str, str] | None = None):
    answers = GOOD_ANSWERS | (overrides or {})

    def run(question: str) -> AgentResult:
        answer = next(text for key, text in answers.items() if key in question)
        return AgentResult(
            question=question,
            answer=answer,
            llm_calls=2,
            prompt_tokens=4000,
            completion_tokens=100,
            seconds=1.0,
        )

    return run


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("API_KEY", KEY)
    app = create_app()
    return TestClient(app), app


@needs_data
def test_demo_passes_against_a_healthy_api(client):
    test_client, app = client
    app.dependency_overrides[get_agent_runner] = lambda: canned_agent()
    lines: list[str] = []
    results = run_demo(test_client, KEY, out=lines.append)
    assert all(r["ok"] for r in results), "\n".join(lines)
    assert len(results) == len(STEPS) + 1  # every step plus the auth check
    text = "\n".join(lines)
    assert "checks passed" in text and "estimated total cost" in text
    assert KEY not in text  # the key is never printed


@needs_data
def test_demo_flags_the_bad_answers_we_fixed_earlier(client):
    """A wrong lead time and an invented market share must be reported as failures."""
    test_client, app = client
    bad = {
        "30% discount": "Promotions must be requested with at least 4 weeks between them.",
        "Pepsi": "Our market share versus Pepsi in Cairo was 100%.",
    }
    app.dependency_overrides[get_agent_runner] = lambda: canned_agent(bad)
    lines: list[str] = []
    results = run_demo(test_client, KEY, out=lines.append)
    failed = {r["name"] for r in results if not r["ok"]}
    assert failed == {"Agent: policy (RAG)", "Agent: unanswerable (must say it does not know)"}
    assert any("forbidden" in line for line in lines)


def test_auth_check_passes_only_when_the_api_refuses_unauthenticated_calls(client, monkeypatch):
    test_client, _ = client
    assert check_auth(test_client, lambda line: None)
    monkeypatch.delenv("API_KEY")  # an API with no key configured answers 503, not 401
    assert not check_auth(test_client, lambda line: None)


def test_wait_until_ready_survives_a_cold_start():
    class ColdStartClient:
        """Connection errors and gateway errors first, then healthy, like a waking replica."""

        def __init__(self):
            self.calls = 0

        def get(self, path):
            self.calls += 1
            if self.calls == 1:
                raise httpx.ConnectError("connection refused")
            if self.calls == 2:
                return httpx.Response(503, json={"status": "unavailable"})
            return httpx.Response(200, json={"status": "ok"})

    cold = ColdStartClient()
    sleeps: list[float] = []
    lines: list[str] = []
    wait_until_ready(cold, lines.append, sleep=sleeps.append)
    assert cold.calls == 3 and sleeps == [5, 5]
    assert any("Ready after" in line for line in lines)


def test_wait_until_ready_gives_up_eventually():
    class Dead:
        def get(self, path):
            raise httpx.ConnectError("down")

    with pytest.raises(TimeoutError):
        wait_until_ready(Dead(), lambda line: None, timeout=0, sleep=lambda s: None)
