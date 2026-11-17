"""Tests for the eval harness itself (scoring, judge parsing, report). No LLM calls."""

import json
import os
from types import SimpleNamespace

import httpx
import openai
import pytest
import sqlalchemy as sa

from evals.judge import Judge, parse_json, quote_in_passages
from evals.report import render_report, summary_table
from evals.retry import is_rate_limit, with_retries
from evals.run_evals import QuestionResult, retrieved_passages, score_answer
from evals.scoring import (
    CheckResult,
    Score,
    check_contains,
    check_numeric,
    check_regex,
    expected_items,
    expected_number,
    extract_numbers,
    load_questions,
    tool_choice_ok,
)
from src.config import DB_URL, MODELS_DIR
from src.tools import TOOLS

DB_PATH = sa.engine.make_url(DB_URL).database
needs_data = pytest.mark.skipif(
    not (DB_PATH and os.path.exists(DB_PATH) and (MODELS_DIR / "model_meta.json").exists()),
    reason="needs data/sales.db and a trained model",
)


# --- the question set -----------------------------------------------------------------------------
def test_question_set_is_well_formed():
    questions = load_questions()
    assert len(questions) >= 25
    assert len({q.id for q in questions}) == len(questions)
    tool_names = {t.__name__ for t in TOOLS}
    for q in questions:
        assert set(q.expected_tools) <= tool_names, q.id
        assert set(q.allowed_extra_tools) <= tool_names, q.id
        assert q.checks, q.id
        assert {c["type"] for c in q.checks} <= {"numeric", "contains", "regex", "judge"}, q.id
    assert {q.id for q in questions if q.regression} == {"reg-01", "reg-02", "reg-03"}
    assert {q.cohort for q in questions} == {"dev", "fresh"}


@needs_data
def test_every_expected_answer_can_be_computed():
    """If a truth_sql / truth_tool is broken, the eval would silently mis-score the agent."""
    for q in load_questions():
        for check in q.checks:
            if check["type"] == "numeric":
                assert expected_number(check) is not None, q.id
            elif check["type"] == "contains":
                assert expected_items(check), q.id


@needs_data
def test_known_expected_values():
    by_id = {q.id: q for q in load_questions()}
    assert expected_number(by_id["sql-01"].checks[0]) == 40  # outlets in Alexandria
    assert expected_items(by_id["sql-03"].checks[0]) == [["Cairo"]]
    assert expected_items(by_id["sql-05"].checks[0]) == [["Orange"]]
    assert any("O115" in alt for alt in expected_items(by_id["sql-07"].checks[0])[0])


# --- deterministic checks -------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("There are 1,234 units.", [1234]),
        ("Uplift was 35.4% (CI 27.0% to 45.4%)", [35.4, 27.0, 45.4]),
        ("Revenue was EGP 98.4 million", [98_400_000]),
        ("About 450k units", [450_000]),
        ("A change of -4.3%", [-4.3]),
        ("Down −4.3%", [-4.3]),
        ("As of 2025-12-22 outlet O001 sold SKU S01: 80 units", [80]),
        ("3 weeks", [3]),
    ],
)
def test_extract_numbers(text, expected):
    assert extract_numbers(text) == pytest.approx(expected)


def test_numeric_check_tolerance():
    check = {"value": 1000, "rel_tol": 0.01}
    assert check_numeric("about 1,005 units", check).passed
    assert not check_numeric("about 1,050 units", check).passed
    assert not check_numeric("no numbers here", check).passed
    assert check_numeric("exactly 300", {"value": 300, "rel_tol": 0}).passed
    assert not check_numeric("301", {"value": 300, "rel_tol": 0}).passed


def test_numeric_check_sign_insensitive_only_when_asked():
    assert check_numeric(
        "fell by 4.3%", {"value": -4.3, "abs_tol": 0.1, "sign_insensitive": True}
    ).passed
    assert not check_numeric("fell by 4.3%", {"value": -4.3, "abs_tol": 0.1}).passed


def test_contains_check_accepts_alternatives():
    check = {"items": [["O115", "Alexandria Cafe 115"], ["Cairo"]]}
    assert check_contains("The outlet Alexandria Cafe 115 in cairo", check).passed
    result = check_contains("Only O115 here", check)
    assert not result.passed and "Cairo" in result.detail


def test_tool_choice_rules():
    assert tool_choice_ok(["a", "b"], ["a", "b"], [])
    assert tool_choice_ok(["a", "x"], ["a"], ["x"])  # an allowed helper tool is fine
    assert not tool_choice_ok(["a"], ["a", "b"], [])  # missing expected tool
    assert not tool_choice_ok(["a", "x"], ["a"], [])  # unexpected extra tool
    assert tool_choice_ok([], [], [])  # a refusal with no tools
    assert tool_choice_ok(["x"], [], ["x"])


# --- regression: the two bugs we fixed must stay caught ------------------------------------------
def _check(question_id: str, answer: str) -> bool:
    q = next(q for q in load_questions() if q.id == question_id)
    regex = next(c for c in q.checks if c["type"] == "regex")
    return check_regex(answer, regex).passed


def test_lead_time_regex_catches_the_original_wrong_answer():
    wrong = (
        "You cannot run a 30% discount on Sprite 1L PET. Regarding timing, promotions must be "
        "requested with at least 4 weeks between the end of one promotion and the "
        "start of the next."
    )
    assert not _check("reg-02", wrong)


@pytest.mark.parametrize(
    "answer",
    [
        "Promotion requests must be submitted at least 3 weeks before the start date.",
        "Submit at least 3 weeks before the start; also leave at least 4 weeks between promotions.",
        "Requests are due three weeks ahead, and promotions need at least 4 weeks between them.",
    ],
)
def test_lead_time_regex_accepts_correct_answers(answer):
    assert _check("reg-02", answer)


def test_pepsi_regex_rejects_an_invented_share_and_accepts_dont_know():
    assert not _check("reg-01", "Our market share versus Pepsi in Cairo was 100%.")
    assert _check("reg-01", "I don't know. The database has no competitor data.")


# --- judge ----------------------------------------------------------------------------------------
def test_parse_json_tolerates_surrounding_text():
    assert parse_json('Sure! {"correct": true, "reason": "ok"} done') == {
        "correct": True,
        "reason": "ok",
    }
    with pytest.raises(ValueError):
        parse_json("no json")


def fake_judge(reply: str) -> Judge:
    return Judge(complete=lambda messages, json_mode=False: reply)


def routed_judge(correct_reply: str, grounded_reply: str) -> Judge:
    """A fake judge that answers correctness and groundedness prompts differently."""

    def complete(messages, json_mode=False):
        return (
            grounded_reply if "grounded in retrieved" in messages[0]["content"] else correct_reply
        )

    return Judge(complete=complete)


def facts(*items, contradictions=()) -> str:
    """A correctness-judge reply: (fact, present, evidence) triples."""
    return json.dumps(
        {
            "facts": [{"fact": f, "present": p, "evidence": e} for f, p, e in items],
            "contradictions": list(contradictions),
        }
    )


ANSWER = "Requests must be submitted at least 3 weeks before the start. Total: 186 units."


def test_correctness_needs_every_fact_with_a_quote_found_in_the_answer():
    ok = fake_judge(facts(("lead time 3 weeks", True, "at least 3 weeks before the start")))
    assert ok.correctness("q", ANSWER, "ref", ["lead time 3 weeks"]).passed

    missing = fake_judge(facts(("lead time 3 weeks", False, "")))
    verdict = missing.correctness("q", ANSWER, "ref", ["lead time 3 weeks"])
    assert not verdict.passed and "missing: lead time 3 weeks" in verdict.detail

    # The judge claims the fact is present but quotes text the answer does not contain.
    invented = fake_judge(facts(("lead time 3 weeks", True, "submitted 3 weeks in advance")))
    verdict = invented.correctness("q", ANSWER, "ref", ["lead time 3 weeks"])
    assert not verdict.passed and "evidence not found" in verdict.detail


def test_extra_information_does_not_fail_an_answer_but_contradictions_do():
    reply = facts(("lead time 3 weeks", True, "3 weeks before the start"))
    assert fake_judge(reply).correctness("q", ANSWER, "ref", ["lead time 3 weeks"]).passed

    contradicted = facts(
        ("lead time 3 weeks", True, "3 weeks before the start"), contradictions=["says 4 weeks"]
    )
    verdict = fake_judge(contradicted).correctness("q", ANSWER, "ref", ["lead time 3 weeks"])
    assert not verdict.passed and "contradiction: says 4 weeks" in verdict.detail


def test_judge_cannot_skip_key_facts_and_garbage_counts_as_failure():
    one_of_two = fake_judge(facts(("a", True, "3 weeks")))
    verdict = one_of_two.correctness("q", ANSWER, "ref", ["a", "b"])
    assert not verdict.passed and "skipped" in verdict.detail
    verdict = fake_judge("I think it is fine").correctness("q", ANSWER, "ref", ["a"])
    assert not verdict.passed and "judge error" in verdict.detail


def claims(*items) -> str:
    return json.dumps(
        {"claims": [{"claim": c, "supported": s, "evidence": e} for c, s, e in items]}
    )


def test_groundedness_requires_every_claim_supported_with_a_real_quote():
    passages = "Requests must be submitted at least 3 weeks before the start date."
    good = fake_judge(
        claims(("3 week lead time", True, "submitted at least 3 weeks before the start"))
    )
    assert good.groundedness("q", "a", passages).passed

    unsupported = fake_judge(claims(("4 week lead time", False, "")))
    verdict = unsupported.groundedness("q", "a", passages)
    assert not verdict.passed and "4 week lead time" in verdict.detail

    # The judge says "supported" but its quote is not in the passages: do not trust it.
    invented = fake_judge(claims(("5 week lead time", True, "must be submitted 5 weeks ahead")))
    verdict = invented.groundedness("q", "a", passages)
    assert not verdict.passed and "evidence not found" in verdict.detail

    assert fake_judge('{"claims": []}').groundedness("q", "a", passages).passed  # nothing to check
    assert not fake_judge("nonsense").groundedness("q", "a", passages).passed


def test_quote_matching_ignores_markdown_and_spacing():
    passages = "| hypermarket | Daily | 45 days credit | 20 cases |"
    assert quote_in_passages("hypermarket Daily 45 days credit", passages)
    assert not quote_in_passages("hypermarket weekly 30 days credit", passages)


# --- scoring one answer ---------------------------------------------------------------------------
def agent_result(answer: str, calls: list[tuple[str, str]]):
    records = [SimpleNamespace(name=n, ok=True, output=o) for n, o in calls]
    return SimpleNamespace(answer=answer, tool_calls=records)


POLICY_Q = next(q for q in load_questions() if q.id == "pol-03")
PASSAGES = json.dumps(
    {"passages": [{"source": "promotions_policy.md", "section": "4. Lead time", "text": "3 weeks"}]}
)


def test_score_answer_for_a_grounded_policy_answer():
    result = agent_result("At least 3 weeks before the start.", [("search_policies", PASSAGES)])
    judge = routed_judge(
        facts(("at least 3 weeks before the start date", True, "3 weeks before the start")),
        claims(("3 weeks", True, "3 weeks")),
    )
    score = score_answer(POLICY_Q, result, judge)
    assert score.tool_ok and score.correct and score.grounded


def test_policy_answer_without_search_is_not_grounded():
    result = agent_result("At least 3 weeks before the start.", [])
    score = score_answer(POLICY_Q, result, routed_judge(facts(), claims()))
    assert not score.tool_ok
    assert score.grounded is False


def test_no_judge_mode_skips_free_text_checks():
    result = agent_result("At least 3 weeks before the start.", [("search_policies", PASSAGES)])
    score = score_answer(POLICY_Q, result, judge=None)
    assert [c.passed for c in score.checks] == [True, None]  # regex ran, judge skipped
    assert score.correct is True  # decided by the checks that ran
    assert score.grounded is None


def test_retrieved_passages_are_extracted_for_the_judge():
    text = retrieved_passages(agent_result("a", [("search_policies", PASSAGES)]))
    assert "promotions_policy.md" in text and "3 weeks" in text


# --- report ---------------------------------------------------------------------------------------
def make_result(qid, seconds, tokens, cost, passed=True):
    q = next(q for q in load_questions() if q.id == qid)
    score = Score(tool_ok=passed, checks=[CheckResult("numeric", passed, "detail")])
    return QuestionResult(
        question=q, answer="an answer", called_tools=["query_sales"], score=score,
        seconds=seconds, prompt_tokens=tokens, completion_tokens=0, cost_usd=cost, llm_calls=2,
        model_name="test-model",
    )  # fmt: skip


def test_report_contains_cost_latency_and_regressions():
    results = [
        make_result("sql-01", 2.0, 4000, 0.002),
        make_result("sql-02", 4.0, 6000, 0.004),
        make_result("reg-02", 6.0, 2000, 0.001, passed=False),
    ]
    report = render_report(results, judge=None)
    assert "Average latency per question | 4.0s" in report
    assert "Average estimated cost per question | $0.0023" in report
    assert "| 4,000 + 0 = 4,000 |" in report
    assert "`reg-02`" in report and "FAIL" in report
    assert "## Failures" in report and "reg-02" in report.split("## Failures")[1]
    assert "judge" in report.lower()
    assert "Original questions (used while tuning)" in report
    table = summary_table(results)
    assert "**All**" in table and "sql" in table and "policy" in table


def test_failed_agent_run_counts_as_failed():
    q = next(q for q in load_questions() if q.id == "sql-01")
    assert not QuestionResult(question=q, error="AgentError: boom").passed


# --- rate limits ----------------------------------------------------------------------------------
def rate_limit_error() -> openai.RateLimitError:
    request = httpx.Request("POST", "https://example.test")
    return openai.RateLimitError(
        "slow down", response=httpx.Response(429, request=request), body=None
    )


def test_with_retries_waits_and_retries_on_rate_limit():
    calls, waits = [], []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise rate_limit_error()
        return "ok"

    assert with_retries(flaky, delay=10, sleep=waits.append) == "ok"
    assert len(calls) == 3
    assert waits == [10, 20]  # backs off a little longer each time


def test_with_retries_gives_up_and_does_not_retry_other_errors():
    def always():
        raise rate_limit_error()

    with pytest.raises(openai.RateLimitError):
        with_retries(always, attempts=2, sleep=lambda s: None)

    calls = []

    def broken():
        calls.append(1)
        raise ValueError("a real bug")

    with pytest.raises(ValueError):
        with_retries(broken, sleep=lambda s: None)
    assert len(calls) == 1


def test_wrapped_rate_limit_is_recognised():
    wrapped = RuntimeError("agent failed")
    wrapped.__cause__ = rate_limit_error()
    assert is_rate_limit(wrapped)
    assert not is_rate_limit(RuntimeError("other"))


def test_judge_outage_skips_the_check_instead_of_failing_it(monkeypatch):
    waits = []
    monkeypatch.setattr("evals.retry.time.sleep", waits.append)  # never really wait in a test

    def down(messages, json_mode=False):
        raise rate_limit_error()

    judge = Judge(complete=down)
    verdict = judge.correctness("q", "a", "ref", ["fact"])
    grounded = judge.groundedness("q", "a", "passages")
    assert verdict.passed is None and "judge unavailable" in verdict.detail
    assert grounded.passed is None
    assert waits  # it did try to back off before giving up


def test_report_warns_when_judge_checks_were_skipped():
    q = next(q for q in load_questions() if q.id == "pol-03")
    score = Score(
        tool_ok=True, checks=[CheckResult("judge", None, "judge unavailable: RateLimitError")]
    )
    result = QuestionResult(
        question=q, called_tools=["search_policies"], score=score, model_name="m"
    )
    assert "judge check(s) were skipped" in render_report([result], judge=None)
