"""Tool 1: answer a sales question by letting the LLM write SQL, then running it safely."""

from __future__ import annotations

import re
from collections.abc import Callable
from functools import lru_cache

import sqlalchemy as sa

from src import llm
from src.config import DB_URL
from src.errors import ToolError
from src.tools.schemas import SalesQueryResult
from src.tools.sql_guard import MAX_ROWS, run_select

SYSTEM_PROMPT = """You write SQLite queries for a beverage sales database. Reply with ONE \
SELECT statement and nothing else (no explanation, no markdown).

{schema}

Rules:
- Read-only: a single SELECT (CTEs with WITH are fine). Use only the tables above.
- Time: weeks are identified by week_start (a Monday, format YYYY-MM-DD). The latest week in \
the data is {latest_week}. Measure every relative period back from that date, NEVER from \
today's date:
    "last week"  = week_start = '{latest_week}'
    "last month" = the last 4 weeks (week_start > date('{latest_week}', '-28 days'))
    "the month before" = the 4 weeks before that (between -56 and -28 days)
    "last quarter" = 13 weeks, "last year" = 52 weeks.
- units = 0 usually means a stock-out, not zero demand. revenue_egp = units * unit_price_egp.
- Compare periods with SUM(units) (or revenue_egp), and use ROUND for percentages.
- Keep result sets small: aggregate, ORDER BY the interesting column, and add LIMIT {limit} \
or less.
"""


@lru_cache
def describe_schema(db_url: str = DB_URL) -> tuple[str, str]:
    """Return (schema text, latest week) built from the live database, so it never goes stale."""
    engine = sa.create_engine(db_url)
    inspector = sa.inspect(engine)
    lines = ["Tables:"]
    for table in inspector.get_table_names():
        columns = ", ".join(f"{c['name']} {c['type']}" for c in inspector.get_columns(table))
        lines.append(f"- {table}({columns})")

    with engine.connect() as conn:

        def distinct(column: str, table: str) -> str:
            rows = conn.execute(sa.text(f"SELECT DISTINCT {column} FROM {table} ORDER BY 1"))
            return ", ".join(str(r[0]) for r in rows)

        first, latest = conn.execute(
            sa.text("SELECT MIN(week_start), MAX(week_start) FROM sales_weekly")
        ).one()
        lines += [
            "",
            "Keys: sales_weekly.outlet_id -> outlets.outlet_id, sales_weekly.sku_id -> "
            "skus.sku_id, sales_weekly.promo_id -> promotions.promo_id (NULL = no promotion), "
            "promotions.sku_id -> skus.sku_id. Primary key of sales_weekly = (week_start, "
            "outlet_id, sku_id).",
            f"outlets.channel values: {distinct('channel', 'outlets')}",
            f"outlets.region values: {distinct('region', 'outlets')}",
            f"skus.brand values: {distinct('brand', 'skus')}",
            f"skus.pack_size values: {distinct('pack_size', 'skus')}",
            "promotions.channel is NULL when the promotion ran in all channels.",
            f"Data covers weeks {first} to {latest}.",
        ]
    return "\n".join(lines), latest


def extract_sql(reply: str) -> str:
    """Pull the SQL out of an LLM reply, tolerating ```sql fences and chatter."""
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", reply, flags=re.DOTALL | re.IGNORECASE)
    return (fenced.group(1) if fenced else reply).strip()


def answer_question(question: str, complete: Callable[[list[dict]], str]) -> SalesQueryResult:
    """Text-to-SQL with one repair attempt: if the SQL fails, show the LLM the error once."""
    if not question.strip():
        raise ToolError("The question is empty.")
    schema, latest_week = describe_schema()
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT.format(schema=schema, latest_week=latest_week, limit=MAX_ROWS),
        },
        {"role": "user", "content": question},
    ]

    sql = extract_sql(complete(messages))
    try:
        output = run_select(sql)
    except ToolError as first_error:
        messages += [
            {"role": "assistant", "content": sql},
            {"role": "user", "content": f"That failed: {first_error}\nReturn a corrected query."},
        ]
        sql = extract_sql(complete(messages))
        output = run_select(sql)  # a second failure is raised to the caller

    return SalesQueryResult(
        question=question,
        sql=sql,
        columns=output.columns,
        rows=output.rows,
        row_count=len(output.rows),
        truncated=output.truncated,
    )


def query_sales(question: str) -> SalesQueryResult:
    """Answer a question about historical sales by querying the sales database.

    Use this for facts about the past: totals, rankings, trends, comparisons between periods,
    regions, channels, outlets, SKUs or promotions. It converts the question to a read-only
    SQL query, runs it, and returns the SQL together with the rows.

    Args:
        question: The question in plain English, e.g. "Which Alexandria outlets dropped
            sales last month?". Include the metric, period and filters you care about.
    """
    return answer_question(question, llm.complete)
