"""Run the agent on every question in evals/questions.jsonl and write results/evals.md.

    uv run python -m evals.run_evals                      # all questions, with the LLM judge
    uv run python -m evals.run_evals --ids sql-01,reg-02  # a few questions
    uv run python -m evals.run_evals --category policy
    uv run python -m evals.run_evals --no-judge           # deterministic checks only (cheaper)

Exit code is 1 if a regression question fails, so CI can use it as a gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass

from evals.judge import Judge
from evals.report import render_report, summary_table
from evals.retry import with_retries
from evals.scoring import (
    CheckResult,
    EvalSetupError,
    Question,
    Score,
    check_contains,
    check_numeric,
    check_regex,
    load_questions,
    tool_choice_ok,
)
from src.agent import run_agent
from src.config import RESULTS_DIR


@dataclass
class QuestionResult:
    question: Question
    answer: str = ""
    called_tools: list[str] | None = None
    score: Score | None = None
    error: str = ""
    seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    llm_calls: int = 0
    model_name: str = ""

    @property
    def passed(self) -> bool:
        """Tool choice right, answer correct, and grounded when it is a RAG question."""
        s = self.score
        if self.error or s is None:
            return False
        return s.tool_ok and s.correct is not False and s.grounded is not False


def retrieved_passages(agent_result) -> str:
    """The passages search_policies returned, as text for the groundedness judge."""
    blocks = []
    for call in agent_result.tool_calls:
        if call.name != "search_policies" or not call.ok:
            continue
        try:
            for p in json.loads(call.output)["passages"]:
                blocks.append(f"[{p['source']} | {p['section']}]\n{p['text']}")
        except (json.JSONDecodeError, KeyError):  # output was truncated: use the raw text
            blocks.append(call.output)
    return "\n\n".join(blocks)


def score_answer(q: Question, agent_result, judge: Judge | None) -> Score:
    """Score tool choice, every check, and (for policy questions) groundedness."""
    called = [c.name for c in agent_result.tool_calls]
    score = Score(tool_ok=tool_choice_ok(called, q.expected_tools, q.allowed_extra_tools))
    answer = agent_result.answer

    for check in q.checks:
        kind = check["type"]
        try:
            if kind == "numeric":
                result = check_numeric(answer, check)
            elif kind == "contains":
                result = check_contains(answer, check)
            elif kind == "regex":
                result = check_regex(answer, check)
            elif judge is None:
                result = CheckResult("judge", None, "skipped (--no-judge)")
            else:
                verdict = judge.correctness(
                    q.question, answer, check["reference"], check.get("key_facts", [])
                )
                result = CheckResult("judge", verdict.passed, verdict.detail)
        except EvalSetupError as error:
            result = CheckResult(kind, False, f"EVAL SETUP ERROR: {error}")
        score.checks.append(result)

    if "search_policies" in q.expected_tools:  # RAG answer: is it supported by what was retrieved?
        passages = retrieved_passages(agent_result)
        if not passages:
            score.grounded, score.grounded_detail = False, "search_policies returned no passages"
        elif judge is not None:
            verdict = judge.groundedness(q.question, answer, passages)
            score.grounded, score.grounded_detail = verdict.passed, verdict.detail
    return score


def evaluate_one(q: Question, judge: Judge | None) -> QuestionResult:
    try:
        agent_result = with_retries(
            lambda: run_agent(q.question),
            on_retry=lambda n: print(f"(rate limited, retry {n}) ", end="", flush=True),
        )
    except Exception as error:  # noqa: BLE001 - one bad question must not stop the run
        return QuestionResult(q, error=f"{type(error).__name__}: {error}", called_tools=[])
    return QuestionResult(
        question=q,
        answer=agent_result.answer,
        called_tools=[c.name for c in agent_result.tool_calls],
        score=score_answer(q, agent_result, judge),
        seconds=agent_result.seconds,
        prompt_tokens=agent_result.prompt_tokens,
        completion_tokens=agent_result.completion_tokens,
        cost_usd=agent_result.cost_usd,
        llm_calls=agent_result.llm_calls,
        model_name=agent_result.model_name,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the FMCG Sales Copilot.")
    parser.add_argument("--ids", help="comma-separated question ids")
    parser.add_argument("--category", help="only this category")
    parser.add_argument("--no-judge", action="store_true", help="skip the LLM judge")
    parser.add_argument("--out", default=str(RESULTS_DIR / "evals.md"))
    args = parser.parse_args()

    questions = load_questions()
    if args.ids:
        wanted = set(args.ids.split(","))
        questions = [q for q in questions if q.id in wanted]
    if args.category:
        questions = [q for q in questions if q.category == args.category]
    if not questions:
        print("No questions selected.", file=sys.stderr)
        return 2

    judge = None if args.no_judge else Judge()
    results = []
    for i, q in enumerate(questions, 1):
        print(f"[{i:>2}/{len(questions)}] {q.id:<7} ", end="", flush=True)
        result = evaluate_one(q, judge)
        results.append(result)
        status = "PASS" if result.passed else "FAIL"
        print(f"{status}  {result.seconds:>5.1f}s  tools={result.called_tools}  {result.error}")

    report = render_report(results, judge)
    out = RESULTS_DIR / "evals.md" if args.out == str(RESULTS_DIR / "evals.md") else None
    path = out or __import__("pathlib").Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report)
    raw = [asdict(r) | {"passed": r.passed} for r in results]
    path.with_suffix(".raw.json").write_text(json.dumps(raw, indent=2, default=str))

    print("\n" + summary_table(results))
    print(f"\nReport written to {path}")
    return 1 if any(r.question.regression and not r.passed for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
