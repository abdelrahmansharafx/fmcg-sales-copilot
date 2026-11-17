"""Tool 2: forecast weekly units for one outlet x SKU with the trained LightGBM model."""

from __future__ import annotations

from src.errors import ToolError
from src.forecast.predict import forecast_outlet, load_model
from src.tools.catalog import outlet_channel, require_sku
from src.tools.schemas import ForecastResult, WeekForecast


def forecast_demand(
    outlet_id: str, sku_id: str, weeks: int = 4, as_of: str | None = None
) -> ForecastResult:
    """Forecast expected weekly unit sales of one SKU at one outlet for the next 1-4 weeks.

    Use this for questions about FUTURE demand. The forecast comes from a machine-learning
    model trained on past sales, promotions, prices, seasonality and Ramadan. It is an
    estimate: the response includes the model's typical error.

    Args:
        outlet_id: Outlet id such as "O001".
        sku_id: SKU id such as "S01".
        weeks: How many weeks ahead, 1 to 4.
        as_of: Optional date (YYYY-MM-DD). Forecast as if today were this date, using only
            sales up to then. Leave empty to forecast from the latest data.
    """
    outlet_channel(outlet_id)  # validates the id with a clear message
    sku_name = require_sku(sku_id)
    try:
        data = forecast_outlet(outlet_id, weeks=weeks, as_of=as_of, sku_ids=[sku_id])
    except ToolError as error:
        if "no sales rows" in str(error):
            raise ToolError(
                f"Outlet '{outlet_id}' does not stock {sku_id} ({sku_name})."
            ) from error
        raise
    if data.empty:
        raise ToolError(f"Outlet '{outlet_id}' does not stock {sku_id} ({sku_name}).")

    _, meta = load_model()
    history_weeks = int(data["history_weeks"].iloc[0])
    assumptions = [
        "Uses the promotions already planned in the promotions table; none assumed otherwise.",
        "Assumes the current list price (no new price increase).",
    ]
    if history_weeks < 26:
        assumptions.append(
            f"Only {history_weeks} weeks of history: this forecast is less reliable."
        )

    forecasts = [
        WeekForecast(
            week_start=row.target_week.date(),
            horizon_weeks=int(row.horizon),
            units=round(float(row.forecast_units), 1),
        )
        for row in data.itertuples()
    ]
    return ForecastResult(
        outlet_id=outlet_id,
        sku_id=sku_id,
        sku_name=sku_name,
        origin_week=data["origin_week"].iloc[0].date(),
        weeks=forecasts,
        total_units=round(sum(f.units for f in forecasts), 1),
        typical_error_pct=meta["test_wape_pct"],
        assumptions=assumptions,
    )
