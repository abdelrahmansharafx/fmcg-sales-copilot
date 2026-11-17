"""Feature engineering for the 4-week-ahead demand forecast (per outlet x SKU).

Vocabulary used everywhere in this project
------------------------------------------
* origin week  : the last week whose sales we already know ("today").
* target week  : the week we predict = origin week + horizon (1..4 weeks).
* One training row = one (outlet, SKU, origin week, horizon). A single LightGBM model learns
  all four horizons, with `horizon` as a feature (the "direct" multi-horizon strategy: no
  recursive predictions that feed errors back in).

Two kinds of features, and why leakage is the main risk
-------------------------------------------------------
1. HISTORY features (lags, rolling means): computed from sales up to and including the origin
   week only. Every rolling window looks backwards, never forwards.
2. TARGET-WEEK features (week of year, Ramadan, promo, price): properties of the week we
   predict that are genuinely known in advance - the calendar is fixed, price lists are
   announced 30 days ahead and promotions are planned >= 3 weeks ahead (see docs/). We use
   them as inputs but NEVER use the target week's units.

   Caveat: promotions need 3 weeks of lead time, so for horizon 4 the promo plan may not be
   final yet. We treat the promotion calendar as known; say so when you present the results.

LEAKAGE = any information in the features that would not be available at the origin week in
real life. It makes offline metrics look great and production results disappointing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import sqlalchemy as sa

KEY = ["outlet_id", "sku_id"]
HORIZONS = (1, 2, 3, 4)
LAGS = (0, 1, 2, 3, 7)  # lag_0 = origin week itself, lag_1 = one week before it, ...
ROLLING_WINDOWS = (4, 8, 13)
SEASONAL_LAG_WEEKS = 52

# Real Ramadan dates (first day, last day). 2026 is included so Phase 3 can forecast ahead.
# Keep in sync with scripts/generate_data.py (a test checks this).
RAMADAN_PERIODS = [
    ("2024-03-11", "2024-04-09"),
    ("2025-03-01", "2025-03-29"),
    ("2026-02-18", "2026-03-19"),
]

CATEGORICAL_FEATURES = ["channel", "region", "sku_id", "pack_type"]
NUMERIC_FEATURES = (
    [f"lag_{k}" for k in LAGS]
    + [f"mean_{w}" for w in ROLLING_WINDOWS]
    + ["zero_share_8", "promo_share_4", "history_weeks"]
    + ["horizon", "target_week_of_year", "target_ramadan_share"]
    + ["target_promo_flag", "target_discount_pct", "target_price", "price_change_vs_origin"]
)
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

PANEL_SQL = """
SELECT s.week_start, s.outlet_id, s.sku_id, s.units, s.unit_price_egp,
       COALESCE(p.discount_pct, 0) AS discount_pct,
       o.channel, o.region, k.pack_type
FROM sales_weekly s
JOIN outlets o ON o.outlet_id = s.outlet_id
JOIN skus k ON k.sku_id = s.sku_id
LEFT JOIN promotions p ON p.promo_id = s.promo_id
"""


def load_panel(engine: sa.Engine) -> pd.DataFrame:
    """Load the weekly sales panel (one row per outlet x SKU x week) from the database."""
    return pd.read_sql(PANEL_SQL, engine, parse_dates=["week_start"])


def ramadan_share(week_start: pd.Series) -> np.ndarray:
    """Fraction of each Monday-to-Sunday week (0..1) that falls inside Ramadan."""
    start = week_start.to_numpy(dtype="datetime64[D]")
    share = np.zeros(len(start))
    for first, last in RAMADAN_PERIODS:
        overlap_days = (
            np.minimum(start + 6, np.datetime64(last)) - np.maximum(start, np.datetime64(first))
        ).astype(int) + 1
        share += np.clip(overlap_days, 0, 7) / 7
    return share


def _check_weekly_contiguous(panel: pd.DataFrame) -> None:
    """Shifting by N rows only means 'N weeks' if every pair has one row per week, no gaps."""
    gaps = panel.groupby(KEY)["week_start"].diff().dropna()
    if not (gaps == pd.Timedelta(days=7)).all():
        raise ValueError("Panel has missing or duplicated weeks for some outlet x SKU pair")


def _rolling_mean(panel: pd.DataFrame, column: str, window: int) -> pd.Series:
    """Mean of `column` over the last `window` weeks per pair, INCLUDING the current row.

    Rolling windows only look backwards, so a value at week t uses weeks t-window+1 .. t.
    """
    rolled = panel.groupby(KEY)[column].rolling(window, min_periods=1).mean()
    return rolled.reset_index(level=KEY, drop=True)  # back to the panel's row index


def add_history_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Add lags, rolling means and the seasonal-naive value, all known at the row's own week."""
    panel = panel.sort_values([*KEY, "week_start"]).reset_index(drop=True)
    _check_weekly_contiguous(panel)

    by_pair = panel.groupby(KEY)
    for k in LAGS:
        panel[f"lag_{k}"] = by_pair["units"].shift(k)  # shift(k) = value k weeks earlier
    for w in ROLLING_WINDOWS:
        panel[f"mean_{w}"] = _rolling_mean(panel, "units", w)

    # Zero units are mostly stock-outs; their recent frequency tells the model about availability.
    panel["is_zero"] = (panel["units"] == 0).astype(float)
    panel["is_promo"] = (panel["discount_pct"] > 0).astype(float)
    panel["zero_share_8"] = _rolling_mean(panel, "is_zero", 8)
    panel["promo_share_4"] = _rolling_mean(panel, "is_promo", 4)
    panel["history_weeks"] = by_pair.cumcount() + 1  # young outlets have little history

    # Baseline: what sold in this same week last year (known long before the target week).
    panel["seasonal_naive"] = by_pair["units"].shift(SEASONAL_LAG_WEEKS)
    return panel


def make_supervised(panel: pd.DataFrame) -> pd.DataFrame:
    """Turn the weekly panel into (origin, horizon) training rows with features and a target."""
    panel = add_history_features(panel).rename(columns={"week_start": "origin_week"})
    by_pair = panel.groupby(KEY)

    # Columns we look up from h weeks AHEAD: shift(-h) pulls the future row up to the origin row.
    # Only used as the target or as target-week covariates, never as history.
    ahead = {
        "origin_week": "target_week",
        "units": "target_units",
        "unit_price_egp": "target_price",
        "discount_pct": "target_discount_pct",
        "seasonal_naive": "seasonal_naive",
    }
    frames = []
    for h in HORIZONS:
        frame = panel.copy()
        frame["horizon"] = h
        for source, name in ahead.items():
            frame[name] = by_pair[source].shift(-h)
        frames.append(frame.dropna(subset=["target_units"]))  # the last h weeks have no target
    data = pd.concat(frames, ignore_index=True)

    data["target_week_of_year"] = data["target_week"].dt.isocalendar().week.astype(int)
    data["target_ramadan_share"] = ramadan_share(data["target_week"])
    data["target_promo_flag"] = (data["target_discount_pct"] > 0).astype(int)
    data["price_change_vs_origin"] = data["target_price"] / data["unit_price_egp"] - 1

    data = data.drop(columns=["is_zero", "is_promo"])
    for column in CATEGORICAL_FEATURES:
        data[column] = data[column].astype("category")
    return data


def time_split(
    data: pd.DataFrame, val_weeks: int = 13, test_weeks: int = 13
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by TARGET WEEK into train / validation / test, in chronological order.

    Why not a random split? Sales are a time series. With a random split, a training row for
    week 50 and a test row for week 49 of the same outlet sit side by side, and the lag
    features of one literally contain the other's target. The model would be graded on weeks
    it has effectively already seen (and on the same seasonal pattern), so metrics come out
    optimistically low. Real use is "train on the past, predict the future", so we evaluate
    exactly like that:

        |------------- train -------------|-- validation --|---- test ----|
        oldest                                                       newest

    * train      : fit the model.
    * validation : early stopping / choosing settings. May be looked at many times.
    * test       : the final, untouched exam. Looked at ONCE, after all decisions are made.

    Rows are assigned by target week. A row's features only use weeks <= its origin week, which
    is always before its own target week, so no training row contains anything from the
    validation or test period.
    """
    last_week = data["target_week"].max()
    test_start = last_week - pd.Timedelta(weeks=test_weeks - 1)
    val_start = test_start - pd.Timedelta(weeks=val_weeks)

    train = data[data["target_week"] < val_start]
    val = data[(data["target_week"] >= val_start) & (data["target_week"] < test_start)]
    test = data[data["target_week"] >= test_start]
    return train, val, test
