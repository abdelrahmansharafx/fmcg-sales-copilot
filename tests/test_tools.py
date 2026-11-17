"""Tests for the five tools. They need data/sales.db and the trained model, which are built by
`scripts/generate_data.py` and `python -m src.forecast.train` (CI runs both first)."""

import math

import numpy as np
import pandas as pd
import pytest
import sqlalchemy as sa

from src.config import DB_URL, MODELS_DIR
from src.forecast.features import FEATURES, load_panel, make_supervised
from src.forecast.predict import forecast_outlet, get_engine, load_model
from src.tools import TOOLS, ToolError, forecast_demand, promo_impact, recommend_order
from src.tools import search_policies as search_policies_tool
from src.tools.promo_impact import bootstrap_uplift
from src.tools.query_sales import answer_question, extract_sql
from src.tools.recommend_order import SAFETY_STOCK_WEEKS, estimate_stock
from src.tools.sql_guard import UnsafeSQLError

DB_PATH = sa.engine.make_url(DB_URL).database
pytestmark = pytest.mark.skipif(
    not (
        DB_PATH
        and __import__("os").path.exists(DB_PATH)
        and (MODELS_DIR / "model_meta.json").exists()
    ),
    reason="needs data/sales.db and a trained model",
)


# --- query_sales (the LLM is replaced by a fake that returns scripted replies) -----------------
def scripted_llm(*replies: str):
    """A fake `complete` function that returns the given replies in order."""
    queue = list(replies)
    return lambda messages: queue.pop(0)


def test_extract_sql_handles_fences_and_plain_text():
    assert extract_sql("```sql\nSELECT 1\n```") == "SELECT 1"
    assert extract_sql("  SELECT 2  ") == "SELECT 2"


def test_query_sales_runs_the_generated_sql():
    result = answer_question(
        "How many outlets?", scripted_llm("```sql\nSELECT COUNT(*) AS n FROM outlets\n```")
    )
    assert result.sql == "SELECT COUNT(*) AS n FROM outlets"
    assert result.columns == ["n"]
    assert result.rows == [[300]]


def test_query_sales_repairs_a_failing_query_once():
    fake = scripted_llm("SELECT nope FROM outlets", "SELECT COUNT(*) FROM outlets")
    assert answer_question("How many outlets?", fake).rows == [[300]]


def test_query_sales_blocks_unsafe_sql_from_the_llm():
    with pytest.raises(UnsafeSQLError):
        answer_question("wipe it", scripted_llm("DROP TABLE outlets", "DELETE FROM outlets"))
    # The data is untouched.
    assert answer_question("count", scripted_llm("SELECT COUNT(*) FROM outlets")).rows == [[300]]


# --- forecast_demand ----------------------------------------------------------------------------
def test_inference_matches_the_training_pipeline():
    """Guards against train/serve skew: the same origin must give the same prediction."""
    origin = pd.Timestamp("2025-09-22")
    served = forecast_outlet("O001", weeks=4, as_of=origin).set_index(["sku_id", "horizon"])

    training_rows = make_supervised(load_panel(get_engine(), outlet_id="O001"))
    training_rows = training_rows[training_rows["origin_week"] == origin]
    model, _ = load_model()
    expected = pd.Series(
        model.predict(training_rows[FEATURES]),
        index=pd.MultiIndex.from_arrays(
            [training_rows["sku_id"].astype(str), training_rows["horizon"]],
            names=["sku_id", "horizon"],
        ),
    )
    common = served.index.intersection(expected.index)
    assert len(common) > 40
    np.testing.assert_allclose(
        served.loc[common, "forecast_units"], expected.loc[common], rtol=1e-6
    )


def test_forecast_demand_returns_the_requested_weeks():
    result = forecast_demand("O001", "S01", weeks=3)
    assert [w.horizon_weeks for w in result.weeks] == [1, 2, 3]
    assert result.total_units == pytest.approx(sum(w.units for w in result.weeks), abs=0.2)
    assert all(w.units >= 0 for w in result.weeks)
    assert result.typical_error_pct > 0


def test_backtest_forecast_is_in_the_right_ballpark():
    """Forecast 4 weeks from an old origin and compare with what really sold."""
    result = forecast_demand("O001", "S01", weeks=4, as_of="2025-09-22")
    with get_engine().connect() as conn:
        actual = conn.execute(
            sa.text(
                "SELECT SUM(units) FROM sales_weekly WHERE outlet_id='O001' AND sku_id='S01' "
                "AND week_start > '2025-09-22' AND week_start <= '2025-10-20'"
            )
        ).scalar()
    assert abs(result.total_units - actual) / actual < 0.6  # weekly noise is ~35%


def test_forecast_demand_errors_are_clear():
    with pytest.raises(ToolError, match="Unknown outlet_id"):
        forecast_demand("O999", "S01")
    with pytest.raises(ToolError, match="Unknown sku_id"):
        forecast_demand("O001", "S99")
    with pytest.raises(ToolError, match="weeks must be between"):
        forecast_demand("O001", "S01", weeks=9)


def test_forecast_demand_rejects_sku_the_outlet_does_not_stock():
    with get_engine().connect() as conn:
        sku = conn.execute(
            sa.text(
                "SELECT sku_id FROM skus WHERE sku_id NOT IN "
                "(SELECT sku_id FROM sales_weekly WHERE outlet_id = 'O002')"
            )
        ).scalar()
    assert sku is not None
    with pytest.raises(ToolError, match="does not stock"):
        forecast_demand("O002", sku)


# --- recommend_order ----------------------------------------------------------------------------
def test_estimate_stock_rules():
    assert estimate_stock(avg_demand=100, last_week_units=0, safety_weeks=0.5) == 0  # stock-out
    assert estimate_stock(100, 60, 0.5) == pytest.approx(100 * 1.5 - 60)
    assert estimate_stock(100, 400, 0.5) == 0  # never negative
    assert estimate_stock(0, 0, 0.5) == 0


def test_recommend_order_follows_its_formula():
    rec = recommend_order("O001")
    assert rec.channel == "hypermarket"
    assert rec.total_order_units == sum(line.order_units for line in rec.lines)
    assert [line.order_units for line in rec.lines] == sorted(
        (line.order_units for line in rec.lines), reverse=True
    )
    for line in rec.lines:
        raw = line.forecast_units + line.safety_stock_units - line.estimated_stock_units
        assert line.order_units == pytest.approx(max(0, math.ceil(raw)), abs=1)  # rounding of parts
        assert (
            line.safety_stock_units >= SAFETY_STOCK_WEEKS["hypermarket"] * line.forecast_units - 0.1
        )
        assert "forecast" in line.explanation


def test_recommend_order_unknown_outlet():
    with pytest.raises(ToolError, match="Unknown outlet_id"):
        recommend_order("O999")


# --- promo_impact -------------------------------------------------------------------------------
def test_bootstrap_uplift_recovers_a_known_effect():
    rng = np.random.default_rng(1)
    base = rng.uniform(20, 60, size=100)
    low, high = bootstrap_uplift(base * 2.0, base)  # promo doubles volume exactly
    assert low == pytest.approx(1.0) and high == pytest.approx(1.0)
    noisy_low, noisy_high = bootstrap_uplift(base * rng.uniform(1.0, 3.0, size=100), base)
    assert noisy_low < noisy_high


def test_promo_impact_is_positive_with_a_sensible_interval():
    result = promo_impact("S01", 2)  # a 15% discount in all channels
    assert result.ci_low_pct <= result.uplift_pct <= result.ci_high_pct
    assert result.uplift_pct > 0
    assert result.outlets_analysed > 50
    # Simulated elasticities are 1.5-2.5, so the implied elasticity should be in that region.
    assert 1.2 < result.implied_price_elasticity < 3.0


def test_promo_impact_validates_inputs():
    with pytest.raises(ToolError, match="Unknown promo_id"):
        promo_impact("S01", 99999)
    with pytest.raises(ToolError, match="is for SKU S01, not S02"):
        promo_impact("S02", 1)
    with pytest.raises(ToolError, match="Unknown sku_id"):
        promo_impact("S99", 1)


# --- search_policies ----------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("question", "expected_source"),
    [
        ("What is the maximum discount without director approval?", "promotions_policy.md"),
        ("What are the payment terms for hypermarkets?", "trade_terms_by_channel.md"),
        ("When did list prices increase?", "pricing_guidelines.md"),
        ("What should the first order of a new outlet look like?", "new_outlet_onboarding.md"),
    ],
)
def test_search_policies_finds_the_right_document(question, expected_source):
    result = search_policies_tool(question, top_k=3)
    assert result.passages[0].source == expected_source
    assert all(0 <= p.score <= 1 for p in result.passages)
    assert all(p.text and p.section for p in result.passages)


def test_search_policies_rejects_empty_question():
    with pytest.raises(ToolError):
        search_policies_tool("   ")


# --- registry -----------------------------------------------------------------------------------
def test_every_tool_has_a_description_for_the_llm():
    assert [t.__name__ for t in TOOLS] == [
        "query_sales",
        "forecast_demand",
        "recommend_order",
        "promo_impact",
        "search_policies",
    ]
    for tool in TOOLS:
        assert tool.__doc__ and "Args:" in tool.__doc__
