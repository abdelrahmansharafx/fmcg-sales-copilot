"""Opt-in smoke test against the real LLM: `uv run pytest -m live`."""

import pytest

from src.agent import run_agent
from src.config import load_llm_settings

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(load_llm_settings().provider == "none", reason="no LLM configured"),
]


def test_agent_answers_a_data_question_with_a_tool_call():
    result = run_agent("How many outlets are there in total?")
    assert [c.name for c in result.tool_calls] == ["query_sales"]
    assert "300" in result.answer


def test_agent_says_it_does_not_know_about_competitors():
    result = run_agent("What was our market share versus Pepsi in Cairo last quarter?")
    assert "don't know" in result.answer.lower() or "do not know" in result.answer.lower()
