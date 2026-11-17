"""The SQL guard must block everything except a single, read-only SELECT."""

import sqlite3

import pytest

from src.tools.sql_guard import (
    QueryError,
    QueryTimeoutError,
    UnsafeSQLError,
    run_select,
    validate_select,
)


@pytest.fixture()
def db(tmp_path):
    """A tiny database with the four allowed tables (the real one is not needed here)."""
    path = tmp_path / "test.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE outlets (outlet_id TEXT, region TEXT);
        CREATE TABLE skus (sku_id TEXT);
        CREATE TABLE promotions (promo_id INTEGER);
        CREATE TABLE sales_weekly (outlet_id TEXT, units INTEGER);
        INSERT INTO outlets VALUES ('O1', 'Cairo'), ('O2', 'Giza');
        """
    )
    conn.executemany("INSERT INTO sales_weekly VALUES ('O1', ?)", [(i,) for i in range(3000)])
    conn.commit()
    conn.close()
    return path


def table_exists(path, name: str) -> bool:
    conn = sqlite3.connect(path)
    found = conn.execute("SELECT 1 FROM sqlite_master WHERE name = ?", (name,)).fetchone()
    conn.close()
    return found is not None


# --- queries that must be blocked ---------------------------------------------------------------
@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE outlets",
        "DELETE FROM outlets",
        "DELETE FROM outlets WHERE region = 'Cairo'",
        "UPDATE outlets SET region = 'X'",
        "INSERT INTO outlets VALUES ('O9', 'Delta')",
        "CREATE TABLE evil (x INT)",
        "ALTER TABLE outlets ADD COLUMN x TEXT",
        "PRAGMA table_info(outlets)",
        "ATTACH DATABASE '/tmp/other.db' AS other",
        "VACUUM",
        "",
        "   ",
        "not sql at all ((",
    ],
)
def test_non_select_statements_are_blocked(sql):
    with pytest.raises(UnsafeSQLError):
        validate_select(sql)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 1; DROP TABLE outlets",
        "SELECT * FROM outlets; DELETE FROM outlets",
        "SELECT 1; SELECT 2",
        "SELECT * FROM outlets;;DROP TABLE skus;",
    ],
)
def test_multiple_statements_are_blocked(sql):
    with pytest.raises(UnsafeSQLError, match="single"):
        validate_select(sql)


def test_unknown_tables_and_dangerous_functions_are_blocked():
    with pytest.raises(UnsafeSQLError, match="not allowed"):
        validate_select("SELECT * FROM sqlite_master")
    with pytest.raises(UnsafeSQLError, match="not allowed"):
        validate_select("SELECT * FROM outlets o JOIN secrets s ON s.id = o.outlet_id")
    with pytest.raises(UnsafeSQLError, match="load_extension"):
        validate_select("SELECT load_extension('evil.so')")


def test_blocked_queries_do_not_change_the_database(db):
    for sql in ["DROP TABLE outlets", "DELETE FROM outlets", "SELECT 1; DROP TABLE outlets"]:
        with pytest.raises(UnsafeSQLError):
            run_select(sql, db_path=db)
    assert table_exists(db, "outlets")
    assert run_select("SELECT COUNT(*) FROM outlets", db_path=db).rows == [[2]]


def test_connection_is_read_only_even_if_validation_is_bypassed(db):
    """Layer 2 on its own: open the DB the way run_select does and try to write."""
    conn = sqlite3.connect(f"{db.resolve().as_uri()}?mode=ro", uri=True)
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        conn.execute("DELETE FROM outlets")
    conn.close()
    assert run_select("SELECT COUNT(*) FROM outlets", db_path=db).rows == [[2]]


# --- queries that must work -----------------------------------------------------------------------
@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM outlets",
        "SELECT region, COUNT(*) AS n FROM outlets GROUP BY region ORDER BY n DESC;",
        "WITH c AS (SELECT * FROM outlets) SELECT * FROM c",
        "SELECT outlet_id FROM outlets UNION SELECT outlet_id FROM sales_weekly",
        "SELECT 'DROP TABLE outlets' AS harmless_text",  # keywords inside strings are fine
    ],
)
def test_valid_selects_pass(sql):
    assert validate_select(sql)


def test_results_have_columns_and_rows(db):
    out = run_select("SELECT region, outlet_id FROM outlets ORDER BY outlet_id", db_path=db)
    assert out.columns == ["region", "outlet_id"]
    assert out.rows == [["Cairo", "O1"], ["Giza", "O2"]]
    assert not out.truncated


def test_row_limit_truncates_results(db):
    out = run_select("SELECT units FROM sales_weekly", max_rows=50, db_path=db)
    assert len(out.rows) == 50
    assert out.truncated


def test_timeout_stops_runaway_queries(db):
    heavy = "SELECT COUNT(*) FROM sales_weekly a, sales_weekly b, sales_weekly c"  # 27 billion rows
    with pytest.raises(QueryTimeoutError):
        run_select(heavy, timeout_seconds=0.3, db_path=db)


def test_sql_errors_are_reported_clearly(db):
    with pytest.raises(QueryError, match="no such column"):
        run_select("SELECT nope FROM outlets", db_path=db)
