"""Turn eval results into the markdown report (results/evals.md) and a console summary."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median

import numpy as np

from evals.judge import Judge
from src import llm
from src.config import llm_prices_per_million, load_llm_settings


def pct(values: list[bool]) -> str:
    """'83% (5/6)' for a list of booleans, or 'n/a' when there is nothing to score."""
    if not values:
        return "n/a"
    return f"{100 * sum(values) / len(values):.0f}% ({sum(values)}/{len(values)})"


def _rows(results, key) -> dict[str, list]:
    groups: dict[str, list] = {}
    for r in results:
        groups.setdefault(key(r), []).append(r)
    return groups


def summarise(group: list) -> list[str]:
    """One table row of metrics for a group of results."""
    scored = [r for r in group if r.score is not None]
    tool = [r.score.tool_ok for r in scored]
    correct = [r.score.correct for r in scored if r.score.correct is not None]
    grounded = [r.score.grounded for r in scored if r.score.grounded is not None]
    return [
        str(len(group)),
        pct(tool),
        pct(correct),
        pct(grounded),
        pct([r.passed for r in group]),
        f"{mean(r.seconds for r in group):.1f}s",
        f"{mean(r.prompt_tokens + r.completion_tokens for r in group):,.0f}",
        f"${mean(r.cost_usd for r in group):.4f}",
    ]


HEADER = [
    "Questions",
    "Tool choice",
    "Answer correct",
    "Grounded (RAG)",
    "All passed",
    "Avg latency",
    "Avg tokens",
    "Avg cost",
]


def table(first_column: str, rows: list[tuple[str, list[str]]]) -> str:
    lines = [
        "| " + " | ".join([first_column, *HEADER]) + " |",
        "|" + "---|" * (len(HEADER) + 1),
    ]
    lines += ["| " + " | ".join([name, *cells]) + " |" for name, cells in rows]
    return "\n".join(lines)


def summary_table(results) -> str:
    rows = [("**All**", summarise(results))]
    for category, group in sorted(_rows(results, lambda r: r.question.category).items()):
        rows.append((category, summarise(group)))
    return table("Category", rows)


def cohort_table(results) -> str:
    """Original questions vs. the fresh variants written after the first run (never tuned on)."""
    names = {
        "dev": "Original questions (used while tuning)",
        "fresh": "Fresh variants (not tuned on)",
    }
    groups = _rows(results, lambda r: r.question.cohort)
    rows = [(names.get(c, c), summarise(g)) for c, g in sorted(groups.items())]
    return table("Question set", rows)


def skipped_note(results) -> str:
    """A visible warning when judge checks could not run, so skipped never looks like passed."""
    skipped = sum(
        1
        for r in results
        if r.score
        for c in r.score.checks
        if c.passed is None and "unavailable" in c.detail
    )
    skipped += sum(
        1
        for r in results
        if r.score
        and r.score.grounded is None
        and r.score.grounded_detail.startswith("judge unavailable")
    )
    if not skipped:
        return ""
    return f"\n> **Warning:** {skipped} judge check(s) were skipped because the LLM API was unavailable (rate limit). Re-run for a complete result.\n"


def failures_section(results) -> str:
    failed = [r for r in results if not r.passed]
    if not failed:
        return "No failures."
    lines = []
    for r in failed:
        reasons = []
        if r.error:
            reasons.append(f"error: {r.error}")
        elif r.score:
            if not r.score.tool_ok:
                q = r.question
                reasons.append(
                    f"tools used {r.called_tools}, expected {q.expected_tools}"
                    f" (+ optional {q.allowed_extra_tools})"
                )
            reasons += [f"{c.type}: {c.detail}" for c in r.score.checks if c.passed is False]
            if r.score.grounded is False:
                reasons.append(f"not grounded: {r.score.grounded_detail}")
        answer = " ".join(r.answer.split())[:300]
        lines.append(f"- **{r.question.id}** ({r.question.category}): " + "; ".join(reasons))
        lines.append(f"  - Q: {r.question.question}")
        lines.append(f"  - A: {answer}")
    return "\n".join(lines)


def regression_section(results) -> str:
    regs = [r for r in results if r.question.regression]
    if not regs:
        return "No regression questions were selected."
    lines = ["| Question | Result |", "|---|---|"]
    for r in regs:
        lines.append(
            f"| `{r.question.id}`: {r.question.question} | {'PASS' if r.passed else 'FAIL'} |"
        )
    return "\n".join(lines)


def detail_table(results) -> str:
    lines = [
        "| ID | Tools called | Tool | Correct | Grounded | Latency | Tokens | Cost |",
        "|---|---|---|---|---|---|---|---|",
    ]
    mark = {True: "yes", False: "NO", None: "-"}
    for r in results:
        s = r.score
        tools = ", ".join(r.called_tools or []) or "none"
        lines.append(
            f"| {r.question.id} | {tools} | {mark[s.tool_ok] if s else 'NO'} | "
            f"{mark[s.correct] if s else 'NO'} | {mark[s.grounded] if s else '-'} | "
            f"{r.seconds:.1f}s | {r.prompt_tokens + r.completion_tokens:,} | ${r.cost_usd:.4f} |"
        )
    return "\n".join(lines)


def cost_latency_section(results, judge: Judge | None) -> str:
    seconds = [r.seconds for r in results]
    prompt = [r.prompt_tokens for r in results]
    completion = [r.completion_tokens for r in results]
    cost = [r.cost_usd for r in results]
    price_in, price_out = llm_prices_per_million()
    lines = [
        "| Metric | Value |",
        "|---|---|",
        f"| Average latency per question | {mean(seconds):.1f}s |",
        f"| Median / 95th percentile latency | {median(seconds):.1f}s / {np.percentile(seconds, 95):.1f}s |",
        f"| Average tokens per question (prompt + completion) | {mean(prompt):,.0f} + {mean(completion):,.0f} = {mean(prompt) + mean(completion):,.0f} |",
        f"| Average estimated cost per question | ${mean(cost):.4f} |",
        f"| Most expensive question | {max(results, key=lambda r: r.cost_usd).question.id} (${max(cost):.4f}) |",
        f"| Total estimated cost of the {len(results)} questions | ${sum(cost):.3f} |",
    ]
    if judge is not None:
        judge_cost = llm.estimate_cost_usd(judge.usage.prompt_tokens, judge.usage.completion_tokens)
        lines.append(
            f"| Judge overhead (eval only, not a per-question product cost) | "
            f"{judge.usage.calls} calls, {judge.usage.prompt_tokens + judge.usage.completion_tokens:,} tokens, ${judge_cost:.3f} |"
        )
    lines += [
        "",
        f"Cost is an **estimate**: tokens x list price of ${price_in:.2f} / ${price_out:.2f} per 1M "
        "input / output tokens (override with `LLM_PRICE_INPUT_PER_1M` / `LLM_PRICE_OUTPUT_PER_1M`). "
        "Tokens include every LLM call of the question: the agent's own calls plus the text-to-SQL "
        "calls made inside `query_sales`. Latency is wall-clock, one question at a time, and includes "
        "network, tools and the LLM.",
    ]
    return "\n".join(lines)


def history() -> str:
    """The hand-written run history (evals/HISTORY.md), included so earlier results are not lost."""
    path = Path(__file__).parent / "HISTORY.md"
    return path.read_text().strip() if path.exists() else "(no history file)"


def render_report(results, judge: Judge | None) -> str:
    settings = load_llm_settings()
    model = next((r.model_name for r in results if r.model_name), "unknown")
    stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    judge_note = (
        "LLM judge: on (same model as the agent)."
        if judge
        else "LLM judge: OFF, free-text checks skipped."
    )
    return f"""# Evaluation results

Run on {stamp} with provider `{settings.provider}`, deployment `{settings.chat_model}` (model `{model}`),
{len(results)} questions, temperature 0. {judge_note}

## Summary

{summary_table(results)}
{skipped_note(results)}
- **Tool choice**: the tools called include every expected tool and nothing outside the expected/allowed set.
- **Answer correct**: all checks of the question pass. Numbers and names are checked deterministically against values computed from the database or by calling the tool directly; free text is graded by an LLM judge against key facts.
- **Grounded (RAG)**: policy answers only claim what the retrieved passages say (LLM judge). A policy answer where `search_policies` was not used counts as not grounded.
- **All passed**: tool choice, correctness and groundedness all OK.

## Original vs fresh questions

The prompt and the judge were improved after the first run exposed failures on the original questions. The fresh variants were written before the re-run and never used for tuning, so they show whether the fixes generalise.

{cohort_table(results)}

## Cost and latency

{cost_latency_section(results, judge)}

## Regression cases

Failures we hit earlier in development and fixed; they must keep passing.

{regression_section(results)}

## Failures

{failures_section(results)}

## Per-question detail

{detail_table(results)}

## History of this eval

{history()}

## Method and limitations

- Expected answers for data questions are computed when the eval runs (`truth_sql`, or a direct call of the tool), so they cannot go stale if the data is regenerated. Tool-based questions check that the agent reports the tool's numbers faithfully; the tools themselves are unit-tested separately.
- The judge is an LLM from the same family as the agent, so it can be lenient; regex checks back it up where the key fact is a simple phrase.
- One run per question at temperature 0. LLM output can still vary slightly between runs, and 30 questions is a small sample: treat a 1-2 question change as noise.
- Number matching reads any number in the answer; a coincidental match is possible for small expected values.
- Not measured yet: whether every number in an answer appears in a tool result (a general "no invented numbers" check).
"""
