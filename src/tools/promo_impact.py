"""Tool 4: how much extra volume did a promotion sell? (promo weeks vs comparable normal weeks)

Method, in plain words
----------------------
For every outlet that stocked the SKU during the promotion:
  * promo average    = average weekly units in the promotion weeks;
  * baseline average = average weekly units in normal weeks (no promotion) in the 4 weeks
                       BEFORE and the 4 weeks AFTER. Using both sides cancels most of the
                       seasonal drift and trend between the promo and its baseline.
Then uplift = total promo average / total baseline average - 1 across outlets.
The 95% confidence interval comes from a bootstrap: resample outlets with replacement many
times, recompute the uplift each time, and take the 2.5th and 97.5th percentiles.
Outlets are the resampling unit because weeks of the same outlet are not independent.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import sqlalchemy as sa

from src.errors import ToolError
from src.forecast.features import RAMADAN_PERIODS
from src.forecast.predict import get_engine
from src.tools.catalog import require_sku
from src.tools.schemas import PromoImpact

BASELINE_WEEKS = 4  # weeks before and after the promotion
BOOTSTRAP_DRAWS = 2000
MIN_BASELINE_WEEKS = 2  # an outlet needs at least this many baseline weeks to be used


def bootstrap_uplift(promo: np.ndarray, base: np.ndarray, seed: int = 0) -> tuple[float, float]:
    """95% interval of sum(promo)/sum(base) - 1 when resampling outlets with replacement."""
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(promo), size=(BOOTSTRAP_DRAWS, len(promo)))
    ratios = promo[draws].sum(axis=1) / base[draws].sum(axis=1) - 1
    low, high = np.percentile(ratios, [2.5, 97.5])
    return float(low), float(high)


def overlaps_ramadan(first_week: pd.Timestamp, last_week: pd.Timestamp) -> bool:
    return any(
        first_week <= pd.Timestamp(end) and pd.Timestamp(start) <= last_week + pd.Timedelta(days=6)
        for start, end in RAMADAN_PERIODS
    )


def promo_impact(sku_id: str, promo_id: int) -> PromoImpact:
    """Estimate the sales uplift of one promotion, with a 95% confidence interval.

    Use this to judge how well a past promotion worked (for example "did promo 12 on S01
    pay off?"). It compares sales in the promotion weeks with comparable normal weeks around
    them at the same outlets and returns the uplift %, its uncertainty and the extra units.

    Args:
        sku_id: SKU id such as "S01". Must be the SKU the promotion was for.
        promo_id: Promotion id (integer) from the promotions table.
    """
    sku_name = require_sku(sku_id)
    engine = get_engine()
    with engine.connect() as conn:
        promo = (
            conn.execute(sa.text("SELECT * FROM promotions WHERE promo_id = :id"), {"id": promo_id})
            .mappings()
            .first()
        )
    if promo is None:
        raise ToolError(f"Unknown promo_id {promo_id}.")
    if promo["sku_id"] != sku_id:
        raise ToolError(f"Promotion {promo_id} is for SKU {promo['sku_id']}, not {sku_id}.")

    start, end = pd.Timestamp(promo["start_week"]), pd.Timestamp(promo["end_week"])
    window_start = start - pd.Timedelta(weeks=BASELINE_WEEKS)
    window_end = end + pd.Timedelta(weeks=BASELINE_WEEKS)
    sales = pd.read_sql(
        sa.text(
            "SELECT s.outlet_id, s.week_start, s.units, s.promo_id "
            "FROM sales_weekly s JOIN outlets o ON o.outlet_id = s.outlet_id "
            "WHERE s.sku_id = :sku AND s.week_start BETWEEN :a AND :b "
            "AND (:channel IS NULL OR o.channel = :channel)"
        ),
        engine,
        params={
            "sku": sku_id,
            "a": window_start.date().isoformat(),
            "b": window_end.date().isoformat(),
            "channel": promo["channel"],
        },
        parse_dates=["week_start"],
    )

    # Stock-outs show as zero units and would drag either side down, so zeros are excluded
    # from both the promo and the baseline weeks.
    sales = sales[sales["units"] > 0]
    in_promo = sales["promo_id"] == promo_id
    in_baseline = sales["promo_id"].isna() & (
        (sales["week_start"] < start) | (sales["week_start"] > end)
    )
    per_outlet = pd.DataFrame(
        {
            "promo_avg": sales[in_promo].groupby("outlet_id")["units"].mean(),
            "base_avg": sales[in_baseline].groupby("outlet_id")["units"].mean(),
            "base_weeks": sales[in_baseline].groupby("outlet_id")["units"].count(),
            "promo_weeks": sales[in_promo].groupby("outlet_id")["units"].count(),
        }
    ).dropna()
    per_outlet = per_outlet[per_outlet["base_weeks"] >= MIN_BASELINE_WEEKS]
    if len(per_outlet) < 5:
        raise ToolError(
            f"Not enough outlets with comparable baseline weeks to measure promo {promo_id} "
            f"({len(per_outlet)} found, need 5)."
        )

    promo_avg, base_avg = per_outlet["promo_avg"].to_numpy(), per_outlet["base_avg"].to_numpy()
    uplift = promo_avg.sum() / base_avg.sum() - 1
    low, high = bootstrap_uplift(promo_avg, base_avg)
    n_weeks = int((end - start).days // 7 + 1)
    incremental = float(((promo_avg - base_avg) * per_outlet["promo_weeks"]).sum())

    discount = int(promo["discount_pct"])
    elasticity = (
        round(math.log(1 + uplift) / -math.log(1 - discount / 100), 2) if uplift > -1 else None
    )
    caveats = [
        "Observational estimate: baseline weeks are normal weeks 4 weeks either side of the promo.",
        "Zero-sales weeks (mostly stock-outs) are excluded from both sides.",
        "Does not measure cannibalisation of other SKUs or a post-promo dip.",
    ]
    if overlaps_ramadan(start, end):
        caveats.append("The promo overlaps Ramadan, so part of the uplift may be seasonal.")
    if low <= 0 <= high:
        caveats.append("The confidence interval includes 0: the uplift is not clearly positive.")

    return PromoImpact(
        promo_id=promo_id,
        sku_id=sku_id,
        sku_name=sku_name,
        channel_scope=promo["channel"] or "all",
        discount_pct=discount,
        promo_weeks=n_weeks,
        outlets_analysed=len(per_outlet),
        uplift_pct=round(uplift * 100, 1),
        ci_low_pct=round(low * 100, 1),
        ci_high_pct=round(high * 100, 1),
        incremental_units=round(incremental, 0),
        implied_price_elasticity=elasticity,
        method=(
            f"Promo weeks vs the same outlets' normal weeks ({BASELINE_WEEKS} before and after); "
            f"95% bootstrap CI over outlets ({BOOTSTRAP_DRAWS} resamples)."
        ),
        caveats=caveats,
    )
