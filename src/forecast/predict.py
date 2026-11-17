"""Use the trained model to forecast the weeks AFTER the last week of data.

Training rows have the target week's price and promo from the data. For the future we have to
assume them, using only what is known at the origin week:
  * promotions: those already planned in the promotions table for the target week;
  * price: the current list price (promo discount removed), with no new list-price increase.
Both assumptions are returned to the caller so they can be shown to the user.
"""

from __future__ import annotations

import json
from functools import lru_cache

import joblib
import lightgbm as lgb
import pandas as pd
import sqlalchemy as sa

from src.config import DB_URL, MODELS_DIR
from src.errors import NotFoundError, NotReadyError, ToolError
from src.forecast.features import (
    CATEGORICAL_FEATURES,
    FEATURES,
    HORIZONS,
    add_history_features,
    load_panel,
    ramadan_share,
)


@lru_cache
def get_engine() -> sa.Engine:
    return sa.create_engine(DB_URL)


@lru_cache
def load_model() -> tuple[lgb.LGBMRegressor, dict]:
    """Load the trained model and its metadata (feature list, category levels, test metrics)."""
    model_path, meta_path = MODELS_DIR / "lgbm_forecast.joblib", MODELS_DIR / "model_meta.json"
    if not model_path.exists() or not meta_path.exists():
        raise NotReadyError("Forecast model not found. Run: uv run python -m src.forecast.train")
    return joblib.load(model_path), json.loads(meta_path.read_text())


def latest_week(engine: sa.Engine) -> pd.Timestamp:
    """The most recent week with sales in the database."""
    with engine.connect() as conn:
        return pd.Timestamp(
            conn.execute(sa.text("SELECT MAX(week_start) FROM sales_weekly")).scalar()
        )


def resolve_origin(engine: sa.Engine, as_of: str | pd.Timestamp | None) -> pd.Timestamp:
    """The origin week: the Monday of `as_of`, capped at the last week with data."""
    last = latest_week(engine)
    if as_of is None:
        return last
    day = pd.Timestamp(as_of)
    monday = day - pd.Timedelta(days=day.weekday())
    return min(monday, last)


def planned_discount(rows: pd.DataFrame, engine: sa.Engine) -> pd.Series:
    """Discount % already planned for each row's SKU, channel and target week (0 if none)."""
    promos = pd.read_sql(
        "SELECT sku_id, channel AS promo_channel, start_week, end_week, discount_pct "
        "FROM promotions",
        engine,
        parse_dates=["start_week", "end_week"],
    ).rename(columns={"discount_pct": "planned_discount"})
    merged = rows[["sku_id", "channel", "target_week"]].reset_index().merge(promos, on="sku_id")
    in_window = (merged["start_week"] <= merged["target_week"]) & (
        merged["target_week"] <= merged["end_week"]
    )
    in_channel = merged["promo_channel"].isna() | (merged["promo_channel"] == merged["channel"])
    best = merged[in_window & in_channel].groupby("index")["planned_discount"].max()
    return best.reindex(rows.index).fillna(0)


def forecast_outlet(
    outlet_id: str,
    weeks: int = 4,
    as_of: str | pd.Timestamp | None = None,
    sku_ids: list[str] | None = None,
) -> pd.DataFrame:
    """Forecast units for every SKU the outlet stocks (or only `sku_ids`), 1..weeks ahead.

    `as_of` lets you forecast from a past week (a backtest): only sales up to that week are used.
    Returns one row per SKU x horizon, with the origin-week features the callers also need.
    """
    if not 1 <= weeks <= max(HORIZONS):
        raise ToolError(f"weeks must be between 1 and {max(HORIZONS)} (the model's horizon).")
    engine = get_engine()
    model, meta = load_model()
    origin = resolve_origin(engine, as_of)

    panel = load_panel(engine, outlet_id=outlet_id, up_to=origin)
    if panel.empty:
        raise NotFoundError(f"No sales found for outlet '{outlet_id}' up to {origin.date()}.")
    if sku_ids is not None:
        panel = panel[panel["sku_id"].isin(sku_ids)]
    history = add_history_features(panel)
    now = history[history["week_start"] == origin].copy()  # SKUs selling at the origin week
    if now.empty:
        raise NotFoundError(
            f"Outlet '{outlet_id}' has no sales rows in the week of {origin.date()}."
        )

    rows = []
    for horizon in range(1, weeks + 1):
        step = now.assign(horizon=horizon, origin_week=origin)
        step["target_week"] = origin + pd.Timedelta(weeks=horizon)
        rows.append(step)
    data = pd.concat(rows, ignore_index=True)

    data["target_discount_pct"] = planned_discount(data, engine)
    list_price = data["unit_price_egp"] / (1 - data["discount_pct"] / 100)  # remove current promo
    # Prices are recorded in EGP with 2 decimals; round the same way so features match training.
    data["target_price"] = (list_price * (1 - data["target_discount_pct"] / 100)).round(2)
    data["target_promo_flag"] = (data["target_discount_pct"] > 0).astype(int)
    data["price_change_vs_origin"] = data["target_price"] / data["unit_price_egp"] - 1
    data["target_week_of_year"] = data["target_week"].dt.isocalendar().week.astype(int)
    data["target_ramadan_share"] = ramadan_share(data["target_week"])

    for column in CATEGORICAL_FEATURES:  # same category codes as in training
        levels = meta["categorical_levels"][column]
        unknown = set(data[column]) - set(levels)
        if unknown:
            raise ToolError(f"{column} {sorted(unknown)} was not in the training data.")
        data[column] = pd.Categorical(data[column], categories=levels)

    data["forecast_units"] = model.predict(data[FEATURES]).clip(min=0)
    data["outlet_id"] = outlet_id
    return data
