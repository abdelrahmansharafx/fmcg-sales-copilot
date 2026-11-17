"""Regression tests for two failures found in Phase 4.

1. Pepsi market share: the agent reported "100% share" because the database has no competitor
   data. It must say it does not know.
2. Combined policy question ("30% discount ... and how far in advance to request"): retrieval
   missed the lead-time section, so the agent mixed up the 4-week gap between promotions with
   the 3-week lead time.

The offline tests check the deterministic parts (retrieval, prompt, check logic). The `live`
tests run the real agent; run them with `uv run pytest -m live`.
"""

import os

import pytest
import sqlalchemy as sa

from evals.scoring import check_regex, load_questions
from src.config import DB_URL, MODELS_DIR, load_llm_settings

DB_PATH = sa.engine.make_url(DB_URL).database
needs_data = pytest.mark.skipif(
    not (DB_PATH and os.path.exists(DB_PATH) and (MODELS_DIR / "model_meta.json").exists()),
    reason="needs data/sales.db and a trained model",
)
needs_llm = pytest.mark.skipif(load_llm_settings().provider == "none", reason="no LLM configured")

COMBINED_QUESTIONS = [
    "Can I run a 30% discount on Sprite 1L PET and how far in advance "
    "do I have to request a promotion?",
    "Can I run a 30% discount on Sprite 1L and how far in advance "
    "do I have to request a promotion?",
    "What discount can I give and what is the lead time to submit "
    "a promotion request, and who approves it?",
]


# --- 2. combined policy question -------------------------------------------------------------
@pytest.mark.parametrize("question", COMBINED_QUESTIONS)
def test_combined_question_retrieves_both_the_discount_and_the_lead_time_sections(question):
    from src.tools import search_policies

    sections = [p.section for p in search_policies(question).passages]  # default top_k
    assert any("Allowed promotion mechanics" in s for s in sections)
    assert any("Planning and lead time" in s for s in sections), sections


@needs_llm
@needs_data
@pytest.mark.live
def test_agent_gets_the_lead_time_right():
    from src.agent import run_agent

    answer = run_agent(COMBINED_QUESTIONS[1]).answer
    question = next(q for q in load_questions() if q.id == "reg-02")
    regex = next(c for c in question.checks if c["type"] == "regex")
    assert check_regex(answer, regex).passed, answer  # says 3 weeks, never "4 weeks to request"
    assert "commercial director" in answer.lower()  # 30% is only possible with that approval


# --- 1. Pepsi / competitor data ----------------------------------------------------------------
@needs_data
def test_prompt_tells_the_agent_there_is_no_competitor_data():
    from src.agent.graph import system_prompt

    prompt = system_prompt().lower()
    assert "no competitor sales" in prompt
    assert "i don't know" in prompt


@needs_llm
@needs_data
@pytest.mark.live
def test_agent_does_not_invent_a_market_share():
    from src.agent import run_agent

    answer = run_agent("What was our market share versus Pepsi in Cairo last quarter?").answer
    question = next(q for q in load_questions() if q.id == "reg-01")
    regex = next(c for c in question.checks if c["type"] == "regex")
    assert check_regex(answer, regex).passed, answer  # no percentage figure
    assert any(
        p in answer.lower() for p in ["don't know", "do not know", "cannot", "no competitor"]
    )


# --- 3. period windows in generated SQL ---------------------------------------------------------
def test_sql_prompt_defines_gap_free_period_windows():
    """The LLM once wrote the previous window as BETWEEN -56 AND -29 days, which silently drops
    the week at exactly -28 days and compares non-adjacent periods. The prompt must give exact,
    half-open conditions and forbid BETWEEN for week windows."""
    from src.tools.query_sales import SYSTEM_PROMPT

    prompt = SYSTEM_PROMPT.format(schema="", latest_week="2025-12-22", limit=200)
    assert "week_start > date('2025-12-22', '-56 days')" in prompt
    assert "week_start <= date('2025-12-22', '-28 days')" in prompt
    assert "Never use BETWEEN for week windows" in prompt
