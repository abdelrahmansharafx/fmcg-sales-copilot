"""Tool 3: suggest next week's order per SKU for an outlet.

    order = forecast + safety stock - estimated stock on hand

There is no stock column in the data, so stock on hand is ESTIMATED (see estimate_stock).
Replace that function with real stock data when it is available.
"""

from __future__ import annotations

import math

from src.forecast.predict import forecast_outlet
from src.tools.catalog import outlet_channel, sku_names
from src.tools.schemas import OrderLine, OrderRecommendation

# Weeks of forecast demand to hold as safety stock, per channel (docs/trade_terms_by_channel.md).
SAFETY_STOCK_WEEKS = {
    "kiosk": 0.5,
    "supermarket": 0.75,
    "restaurant_cafe": 0.5,
    "hypermarket": 1.0,
}
PEAK_EXTRA_WEEKS = 0.25  # added during Ramadan weeks and promotion weeks (same document)


def estimate_stock(avg_demand: float, last_week_units: float, safety_weeks: float) -> float:
    """Estimate units on hand now, assuming a simple 'order-up-to' habit.

    We assume the outlet started last week topped up to its recent average weekly demand
    plus its normal safety stock, then sold `last_week_units`. Zero sales in a week that
    normally sells is treated as a likely stock-out: nothing on hand.
    """
    if last_week_units == 0 and avg_demand >= 1:
        return 0.0
    topped_up_level = avg_demand * (1 + safety_weeks)
    return max(0.0, topped_up_level - last_week_units)


def recommend_order(outlet_id: str) -> OrderRecommendation:
    """Recommend how many units of each SKU an outlet should order for next week.

    Use this when a rep or manager asks what an outlet should order or restock. For every SKU
    the outlet stocks it combines the demand forecast, a safety stock that depends on the
    channel (and is higher in Ramadan or promo weeks) and an estimate of current stock. It
    returns one line per SKU with a short explanation, largest orders first.

    Args:
        outlet_id: Outlet id such as "O001".
    """
    channel = outlet_channel(outlet_id)
    names = sku_names()
    data = forecast_outlet(outlet_id, weeks=1)  # next week only

    lines = []
    for row in data.itertuples():
        peak = row.target_promo_flag == 1 or row.target_ramadan_share > 0
        base_weeks = SAFETY_STOCK_WEEKS[channel]
        safety_weeks = base_weeks + (PEAK_EXTRA_WEEKS if peak else 0)

        forecast = float(row.forecast_units)
        safety = safety_weeks * forecast
        stock = estimate_stock(float(row.mean_4), float(row.lag_0), base_weeks)
        order = max(0, math.ceil(forecast + safety - stock))

        reason = (
            f"forecast {forecast:.0f} + safety {safety:.0f} ({safety_weeks:g} wk"
            f"{', promo/Ramadan week' if peak else ''}) - est. stock {stock:.0f} = {order}"
        )
        if row.lag_0 == 0 and row.mean_4 >= 1:
            reason += "; zero sales last week, likely a stock-out"
        lines.append(
            OrderLine(
                sku_id=row.sku_id,
                sku_name=names[row.sku_id],
                forecast_units=round(forecast, 1),
                safety_stock_units=round(safety, 1),
                estimated_stock_units=round(stock, 1),
                order_units=order,
                explanation=reason,
            )
        )

    lines.sort(key=lambda line: line.order_units, reverse=True)
    return OrderRecommendation(
        outlet_id=outlet_id,
        channel=channel,
        forecast_origin_week=data["origin_week"].iloc[0].date(),
        order_for_week=data["target_week"].iloc[0].date(),
        lines=lines,
        total_order_units=sum(line.order_units for line in lines),
        assumptions=[
            "Stock on hand is estimated from recent sales (no stock data): the outlet is assumed "
            "to top up to its 4-week average demand plus normal safety stock each week.",
            f"Safety stock for {channel}: {SAFETY_STOCK_WEEKS[channel]:g} weeks of forecast "
            f"demand (+{PEAK_EXTRA_WEEKS:g} in Ramadan/promo weeks), per the trade terms policy.",
            "Quantities are in units (bottles/cans), not cases.",
        ],
    )
