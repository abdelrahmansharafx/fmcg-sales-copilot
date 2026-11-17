"""The five tools. Plain, typed Python functions: the agent (Phase 4), the API (Phase 6) and
the MCP server (Phase 6) all import them from here, so the logic exists exactly once."""

from src.errors import ToolError
from src.tools.forecast_demand import forecast_demand
from src.tools.promo_impact import promo_impact
from src.tools.query_sales import query_sales
from src.tools.recommend_order import recommend_order
from src.tools.search_policies import search_policies

TOOLS = [query_sales, forecast_demand, recommend_order, promo_impact, search_policies]

__all__ = [
    "TOOLS",
    "ToolError",
    "forecast_demand",
    "promo_impact",
    "query_sales",
    "recommend_order",
    "search_policies",
]
