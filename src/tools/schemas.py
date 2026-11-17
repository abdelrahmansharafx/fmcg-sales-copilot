"""Typed results returned by the tools (pydantic, so the API and MCP server can reuse them)."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel


class SalesQueryResult(BaseModel):
    question: str
    sql: str  # the SQL that was executed, so answers can be audited
    columns: list[str]
    rows: list[list]
    row_count: int
    truncated: bool  # True if more rows existed than the row limit


class WeekForecast(BaseModel):
    week_start: date
    horizon_weeks: int
    units: float


class ForecastResult(BaseModel):
    outlet_id: str
    sku_id: str
    sku_name: str
    origin_week: date  # last week of actual sales used
    weeks: list[WeekForecast]
    total_units: float
    typical_error_pct: float  # WAPE of the model on the held-out test period
    assumptions: list[str]


class OrderLine(BaseModel):
    sku_id: str
    sku_name: str
    forecast_units: float
    safety_stock_units: float
    estimated_stock_units: float
    order_units: int
    explanation: str


class OrderRecommendation(BaseModel):
    outlet_id: str
    channel: str
    forecast_origin_week: date
    order_for_week: date
    lines: list[OrderLine]
    total_order_units: int
    assumptions: list[str]


class PromoImpact(BaseModel):
    promo_id: int
    sku_id: str
    sku_name: str
    channel_scope: str  # a channel name or "all"
    discount_pct: int
    promo_weeks: int
    outlets_analysed: int
    uplift_pct: float  # % more units than the baseline
    ci_low_pct: float  # 95% confidence interval for uplift_pct
    ci_high_pct: float
    incremental_units: float
    implied_price_elasticity: float | None
    method: str
    caveats: list[str]


class PolicyPassage(BaseModel):
    source: str  # file name, e.g. promotions_policy.md
    section: str  # heading inside the file
    text: str
    score: float  # similarity, 0..1 (higher = more relevant)


class PolicySearchResult(BaseModel):
    question: str
    passages: list[PolicyPassage]
