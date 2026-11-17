"""Strict, read-only SQL execution for the query_sales tool.

The LLM writes SQL, so we treat that SQL as untrusted input. Four independent layers:

1. VALIDATE   exactly one statement, and it must be a SELECT (or WITH ... SELECT / UNION),
              touching only known tables, with no dangerous functions.
2. READ-ONLY  the database is opened with SQLite's `mode=ro` plus `PRAGMA query_only`, so
              even if validation had a bug the database itself refuses to write.
3. ROW LIMIT  we fetch at most MAX_ROWS rows, whatever the SQL says.
4. TIMEOUT    a progress handler aborts the query after TIMEOUT_SECONDS (runaway joins).

For Postgres you would replace layer 2 with a dedicated read-only database role.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import sqlalchemy as sa
import sqlglot
from sqlglot import exp

from src.config import DB_URL
from src.errors import ToolError

MAX_ROWS = 200
TIMEOUT_SECONDS = 5.0
ALLOWED_TABLES = {"outlets", "skus", "promotions", "sales_weekly"}
BLOCKED_FUNCTIONS = {"load_extension", "readfile", "writefile", "edit", "fts3_tokenizer"}

# Statement types that must never appear anywhere in the parsed query.
_FORBIDDEN_NODE_NAMES = (
    "Insert Update Delete Drop Create Alter Command Pragma Attach Detach Transaction "
    "Commit Rollback Merge Set Into"
).split()
FORBIDDEN_NODES = tuple(getattr(exp, name) for name in _FORBIDDEN_NODE_NAMES if hasattr(exp, name))


class UnsafeSQLError(ToolError):
    """The SQL is not a single, safe, read-only SELECT."""


class QueryTimeoutError(ToolError):
    """The query ran longer than the allowed time."""


class QueryError(ToolError):
    """The database rejected the query (unknown column, syntax error, ...)."""


@dataclass
class QueryOutput:
    columns: list[str]
    rows: list[list]
    truncated: bool  # True if there were more than max_rows rows


def validate_select(sql: str) -> str:
    """Return the SQL (without a trailing semicolon) if it is a single safe SELECT, else raise."""
    if not sql or not sql.strip():
        raise UnsafeSQLError("Empty query.")
    try:
        statements = [s for s in sqlglot.parse(sql, read="sqlite") if s is not None]
    except sqlglot.errors.SqlglotError as error:
        raise UnsafeSQLError(f"Could not parse the SQL: {error}") from error

    if len(statements) != 1:
        raise UnsafeSQLError("Only a single SQL statement is allowed.")
    tree = statements[0]

    if not isinstance(tree, exp.Query):  # Select, Union, ... (WITH is part of a Select)
        raise UnsafeSQLError(f"Only SELECT queries are allowed, got {tree.key.upper()}.")
    if tree.find(*FORBIDDEN_NODES):
        raise UnsafeSQLError("The query contains a forbidden statement type.")

    for function in tree.find_all(exp.Func):
        name = (function.name or function.sql_name()).lower()
        if name in BLOCKED_FUNCTIONS:
            raise UnsafeSQLError(f"Function {name} is not allowed.")

    cte_names = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
    for table in tree.find_all(exp.Table):
        name = table.name.lower()
        if name not in ALLOWED_TABLES and name not in cte_names:
            allowed = ", ".join(sorted(ALLOWED_TABLES))
            raise UnsafeSQLError(f"Table '{table.name}' is not allowed. Use only: {allowed}.")
    return sql.strip().rstrip(";").strip()


def _db_path() -> Path:
    path = sa.engine.make_url(DB_URL).database
    if not path:
        raise ToolError("DATABASE_URL must point to a SQLite file for the SQL tool.")
    return Path(path)


def run_select(
    sql: str,
    *,
    max_rows: int = MAX_ROWS,
    timeout_seconds: float = TIMEOUT_SECONDS,
    db_path: Path | None = None,
) -> QueryOutput:
    """Validate `sql`, then run it on a read-only connection with a row limit and a timeout."""
    sql = validate_select(sql)
    path = Path(db_path or _db_path())
    if not path.exists():
        raise ToolError(
            f"Database not found at {path}. Run: uv run python scripts/generate_data.py"
        )

    connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)  # layer 2
    try:
        connection.execute("PRAGMA query_only = ON")
        deadline = time.monotonic() + timeout_seconds
        # SQLite calls this every 10,000 VM steps; returning 1 aborts the running query.
        connection.set_progress_handler(lambda: int(time.monotonic() > deadline), 10_000)
        cursor = connection.execute(sql)
        columns = [d[0] for d in cursor.description or []]
        rows = cursor.fetchmany(max_rows + 1)  # one extra row tells us if we truncated
    except sqlite3.OperationalError as error:
        if "interrupted" in str(error).lower():
            raise QueryTimeoutError(f"Query stopped after {timeout_seconds} seconds.") from error
        raise QueryError(f"SQL error: {error}") from error
    except sqlite3.Error as error:
        raise QueryError(f"SQL error: {error}") from error
    finally:
        connection.close()

    truncated = len(rows) > max_rows
    return QueryOutput(columns, [list(r) for r in rows[:max_rows]], truncated)
