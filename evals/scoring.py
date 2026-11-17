"""Scoring helpers for the evals: questions, truth values, deterministic checks.

Every question lists `checks`. A question is CORRECT when all its checks pass:
  numeric   a number in the answer matches the expected value within a tolerance
  contains  every expected item (any of its alternative spellings) appears in the answer
  regex     all `all` patterns match and no `none` pattern matches
  judge     an LLM compares the answer with a reference and key facts (free text)
Expected values are never typed in by hand for data questions: they are computed from the
database (`truth_sql`) or by calling the tool directly (`truth_tool`) when the eval runs.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.tools import TOOLS
from src.tools.sql_guard import run_select

QUESTIONS_PATH = Path(__file__).parent / "questions.jsonl"
TOOLS_BY_NAME = {tool.__name__: tool for tool in TOOLS}


class EvalSetupError(ValueError):
    """A question's expected answer could not be computed (a bug in the eval set, not the agent)."""


@dataclass
class Question:
    id: str
    category: str
    question: str
    expected_tools: list[str]
    allowed_extra_tools: list[str]
    checks: list[dict]
    regression: bool = False
    cohort: str = "dev"  # "fresh" = written after the first run, never used for tuning


@dataclass
class CheckResult:
    type: str
    passed: bool | None  # None = skipped (judge disabled)
    detail: str


@dataclass
class Score:
    tool_ok: bool
    checks: list[CheckResult] = field(default_factory=list)
    grounded: bool | None = None  # only for questions that should use search_policies
    grounded_detail: str = ""

    @property
    def correct(self) -> bool | None:
        """True if every check that ran passed; None if no check could run."""
        ran = [c for c in self.checks if c.passed is not None]
        return all(c.passed for c in ran) if ran else None


def load_questions(path: Path = QUESTIONS_PATH) -> list[Question]:
    questions = [Question(**json.loads(line)) for line in path.read_text().splitlines() if line]
    ids = [q.id for q in questions]
    if len(ids) != len(set(ids)):
        raise EvalSetupError("Duplicate question ids in questions.jsonl")
    return questions


# --- tool choice ------------------------------------------------------------------------------
def tool_choice_ok(called: list[str], expected: list[str], allowed_extra: list[str]) -> bool:
    """Correct when every expected tool was used and nothing outside expected + allowed was."""
    used = set(called)
    return set(expected) <= used <= set(expected) | set(allowed_extra)


# --- truth values -----------------------------------------------------------------------------
def get_path(data: Any, path: str) -> Any:
    """Follow a dotted path such as 'lines.0.sku_name' through dicts and lists."""
    for part in path.split("."):
        data = data[int(part)] if isinstance(data, list) else data[part]
    return data


def _sql_rows(sql: str) -> list[list]:
    try:
        return run_select(sql).rows
    except Exception as error:  # noqa: BLE001
        raise EvalSetupError(f"truth_sql failed: {error}") from error


_tool_cache: dict[str, dict] = {}


def tool_truth(spec: dict) -> dict:
    """Call a tool directly (the reference the agent's answer is compared with)."""
    args = dict(spec["args"])
    if "args_sql" in spec:  # extra arguments looked up in the database (column name = arg name)
        try:
            out = run_select(spec["args_sql"])
        except Exception as error:  # noqa: BLE001
            raise EvalSetupError(f"args_sql failed: {error}") from error
        if not out.rows:
            raise EvalSetupError("args_sql returned no rows")
        args[out.columns[0]] = out.rows[0][0]
    key = json.dumps([spec["name"], args], sort_keys=True)
    if key not in _tool_cache:
        _tool_cache[key] = TOOLS_BY_NAME[spec["name"]](**args).model_dump(mode="json")
    return _tool_cache[key]


def expected_number(check: dict) -> float:
    if "value" in check:
        return float(check["value"])
    if "truth_sql" in check:
        rows = _sql_rows(check["truth_sql"])
        if not rows or rows[0][0] is None:
            raise EvalSetupError("truth_sql returned no value")
        return float(rows[0][0])
    spec = check["truth_tool"]
    return float(get_path(tool_truth(spec), spec["field"]))


def expected_items(check: dict) -> list[list[str]]:
    """Items that must appear; each item is a list of acceptable spellings."""
    items = [list(item) for item in check.get("items", [])]
    if "truth_sql" in check:
        for row in _sql_rows(check["truth_sql"]):
            items.append([str(cell) for cell in row if cell is not None])
    if "truth_tool" in check:
        spec = check["truth_tool"]
        data = tool_truth(spec)
        items.append([str(get_path(data, f)) for f in spec["fields"]])
    if not items:
        raise EvalSetupError("contains check has no expected items")
    return items


# --- deterministic checks ---------------------------------------------------------------------
_SCALES = {
    "k": 1e3, "thousand": 1e3,
    "m": 1e6, "mn": 1e6, "million": 1e6,
    "b": 1e9, "bn": 1e9, "billion": 1e9,
}  # fmt: skip
_NUMBER = re.compile(
    r"(?<![\w.])(-?\d[\d,]*(?:\.\d+)?)\s*(thousand|million|billion|bn|mn|k|m|b)?\b", re.I
)


def extract_numbers(text: str) -> list[float]:
    """All numbers in the text, understanding 1,234 / 12.5% / 3.2 million / 450k."""
    text = text.replace("−", "-")  # unicode minus
    text = re.sub(r"\d{4}-\d{2}-\d{2}", " ", text)  # dates are not quantities
    text = re.sub(r"\b[A-Za-z]\d{2,4}\b", " ", text)  # ids such as O001 / S01
    numbers = []
    for raw, scale in _NUMBER.findall(text):
        value = float(raw.replace(",", ""))
        numbers.append(value * _SCALES.get(scale.lower(), 1) if scale else value)
    return numbers


def check_numeric(answer: str, check: dict) -> CheckResult:
    expected = expected_number(check)
    rel_tol, abs_tol = check.get("rel_tol", 0.01), check.get("abs_tol", 0.0)
    allowed = max(abs(expected) * rel_tol, abs_tol)
    flip = check.get("sign_insensitive", False)  # "down 4.3%" and "-4.3%" are the same answer
    found = extract_numbers(answer)
    hit = [
        n
        for n in found
        if abs(n - expected) <= allowed or (flip and abs(abs(n) - abs(expected)) <= allowed)
    ]
    detail = f"expected {expected:,.2f} (tolerance {allowed:,.2f}); " + (
        f"matched {hit[0]:,.2f}" if hit else f"answer numbers: {[round(n, 2) for n in found[:6]]}"
    )
    return CheckResult("numeric", bool(hit), detail)


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower())


def check_contains(answer: str, check: dict) -> CheckResult:
    text = _normalise(answer)
    missing = [
        item for item in expected_items(check) if not any(_normalise(a) in text for a in item)
    ]
    detail = "all expected items found" if not missing else f"missing: {missing}"
    return CheckResult("contains", not missing, detail)


def check_regex(answer: str, check: dict) -> CheckResult:
    problems = [f"missing /{p}/" for p in check.get("all", []) if not re.search(p, answer, re.I)]
    problems += [
        f"forbidden /{p}/ matched" for p in check.get("none", []) if re.search(p, answer, re.I)
    ]
    return CheckResult("regex", not problems, "ok" if not problems else "; ".join(problems))
