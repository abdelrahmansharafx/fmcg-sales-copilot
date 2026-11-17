"""Tests for forecast feature engineering, the time split and the metrics."""

import numpy as np
import pandas as pd
import pytest

from scripts import generate_data as gd
from src.forecast import features as ft
from src.forecast.evaluate import mape, rmse, wape

N_WEEKS = 70


def make_panel(n_weeks: int = N_WEEKS, seed: int = 0) -> pd.DataFrame:
    """Two outlet x SKU pairs with random units, one promo week and a price rise."""
    rng = np.random.default_rng(seed)
    weeks = pd.date_range("2024-01-01", periods=n_weeks, freq="W-MON")
    frames = []
    for outlet_id, sku_id in [("O1", "S1"), ("O2", "S1")]:
        frames.append(
            pd.DataFrame(
                {
                    "week_start": weeks,
                    "outlet_id": outlet_id,
                    "sku_id": sku_id,
                    "units": rng.integers(0, 50, size=n_weeks),
                    "unit_price_egp": 20.0,
                    "discount_pct": 0,
                    "channel": "kiosk",
                    "region": "Cairo",
                    "pack_type": "can",
                }
            )
        )
    panel = pd.concat(frames, ignore_index=True)
    panel.loc[panel["week_start"] == weeks[30], ["discount_pct", "unit_price_egp"]] = [20, 16.0]
    return panel


def row(data: pd.DataFrame, outlet: str, origin: str, horizon: int) -> pd.Series:
    mask = (
        (data["outlet_id"] == outlet)
        & (data["origin_week"] == pd.Timestamp(origin))
        & (data["horizon"] == horizon)
    )
    assert mask.sum() == 1
    return data[mask].iloc[0]


def test_lags_and_rolling_means_look_backwards():
    panel = make_panel()
    data = ft.make_supervised(panel)
    units = panel[panel["outlet_id"] == "O1"].set_index("week_start")["units"]
    r = row(data, "O1", "2024-06-03", 1)  # origin week index 22
    origin = pd.Timestamp("2024-06-03")
    assert r["lag_0"] == units[origin]
    assert r["lag_1"] == units[origin - pd.Timedelta(weeks=1)]
    assert r["lag_7"] == units[origin - pd.Timedelta(weeks=7)]
    assert r["mean_4"] == pytest.approx(units[origin - pd.Timedelta(weeks=3) : origin].mean())


def test_target_is_h_weeks_after_origin():
    panel = make_panel()
    data = ft.make_supervised(panel)
    units = panel[panel["outlet_id"] == "O1"].set_index("week_start")["units"]
    for h in ft.HORIZONS:
        r = row(data, "O1", "2024-06-03", h)
        target_week = pd.Timestamp("2024-06-03") + pd.Timedelta(weeks=h)
        assert r["target_week"] == target_week
        assert r["target_units"] == units[target_week]


def test_features_never_use_the_future():
    """Leakage test: rewriting every week AFTER the origin must not change its features."""
    panel = make_panel()
    origin = pd.Timestamp("2024-06-03")
    altered = panel.copy()
    after = altered["week_start"] > origin
    altered.loc[after, "units"] = altered.loc[after, "units"] + 1000

    before_rows = ft.make_supervised(panel)
    after_rows = ft.make_supervised(altered)
    history_features = [
        c for c in ft.NUMERIC_FEATURES if c.startswith(("lag_", "mean_")) or c.endswith("_share")
    ]
    for h in ft.HORIZONS:
        a, b = row(before_rows, "O1", "2024-06-03", h), row(after_rows, "O1", "2024-06-03", h)
        assert a[history_features].equals(b[history_features])


def test_seasonal_naive_is_same_week_last_year():
    panel = make_panel()
    data = ft.make_supervised(panel)
    units = panel[panel["outlet_id"] == "O1"].set_index("week_start")["units"]
    r = row(data, "O1", "2025-02-17", 2)  # target week 2025-03-03
    assert r["seasonal_naive"] == units[pd.Timestamp("2025-03-03") - pd.Timedelta(weeks=52)]
    assert np.isnan(row(data, "O1", "2024-06-03", 1)["seasonal_naive"])  # no history yet


def test_promo_and_price_features_describe_the_target_week():
    data = ft.make_supervised(make_panel())
    r = row(data, "O1", "2024-07-15", 2)  # target week index 30 is the promo week
    assert r["target_promo_flag"] == 1
    assert r["target_discount_pct"] == 20
    assert r["price_change_vs_origin"] == pytest.approx(16 / 20 - 1)


def test_missing_weeks_are_rejected():
    panel = make_panel().drop(index=10)  # punches a hole in one pair's weekly series
    with pytest.raises(ValueError):
        ft.add_history_features(panel)


def test_time_split_is_chronological_and_has_no_overlap():
    data = ft.make_supervised(make_panel())
    train, val, test = ft.time_split(data, val_weeks=6, test_weeks=6)
    assert train["target_week"].max() < val["target_week"].min()
    assert val["target_week"].max() < test["target_week"].min()
    assert len(train) + len(val) + len(test) == len(data)
    # Every feature is built from weeks up to the origin, which is before the target.
    assert (data["origin_week"] < data["target_week"]).all()


def test_ramadan_share_matches_the_data_generator():
    weeks = pd.Series(gd.WEEKS)
    assert ft.ramadan_share(weeks) == pytest.approx(gd.ramadan_share())


def test_metrics_on_a_small_example():
    actual = np.array([10.0, 0.0, 20.0])
    predicted = np.array([12.0, 3.0, 15.0])
    assert wape(actual, predicted) == pytest.approx((2 + 3 + 5) / 30 * 100)
    assert mape(actual, predicted) == pytest.approx((0.2 + 0.25) / 2 * 100)  # the 0 row is skipped
    assert rmse(actual, predicted) == pytest.approx(np.sqrt((4 + 9 + 25) / 3))
