"""MCP server: exposes the five sales-copilot tools to any MCP client (e.g. Claude Desktop).

    uv run python -m src.mcp_server        # speaks MCP over stdio; a client starts it for you

The tools are the same plain functions the agent and the API use (src/tools), so there is no
duplicated logic. Over stdio the process's STDOUT IS THE PROTOCOL CHANNEL: never print() here,
and send all logging to stderr.

Note: query_sales calls the server's own LLM (from .env) to write SQL; the other four tools need
no LLM at all.
"""

from __future__ import annotations

import functools
import logging
import sys
from collections.abc import Callable

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError as McpToolError
from mcp.types import ToolAnnotations

from src.api.logging_setup import configure_logging
from src.errors import ToolError
from src.tools import TOOLS

logger = logging.getLogger("fmcg.mcp")

# Every tool only reads data: clients can show them as safe and need less confirmation.
READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False, open_world_hint=False)

INSTRUCTIONS = """\
Tools for a beverage sales copilot (synthetic Egypt data: 300 outlets, 25 SKUs, weekly sales).

- Ids: outlets look like O001..O300, SKUs like S01..S25. Resolve names to ids with query_sales.
- Use query_sales for facts about PAST sales; forecast_demand for FUTURE units of one outlet and
  SKU (1-4 weeks); recommend_order for what an outlet should order next week; promo_impact for how
  well a past promotion worked; search_policies for company rules (promotions, trade terms,
  pricing, onboarding).
- Time: measure relative periods ("last month" = the last 4 weeks) back from the LATEST WEEK IN THE
  DATA ({latest_week}), not from today's date.
- Units are packs (cans/bottles), currency is EGP. The data holds only our own sales: there is no
  competitor, market-share, cost, margin or stock-level data, so say so instead of guessing.
- Forecasts and order recommendations are estimates: report the stated typical error / assumptions.
"""


def _latest_week() -> str:
    """The last week of data, for the instructions; a placeholder if the database is missing."""
    try:
        from src.forecast.predict import get_engine, latest_week

        return latest_week(get_engine()).date().isoformat()
    except Exception:  # noqa: BLE001 - the server should still start so the error is visible
        return "unknown (database not found)"


def expose(function: Callable) -> Callable:
    """Wrap a tool so its expected failures reach the model as readable error messages.

    MCP treats only its own ToolError as an anticipated failure whose message the model may read
    ("Unknown outlet_id 'O999'..." lets Claude correct itself). Any other exception counts as a
    crash: the client sees only a generic message and the traceback stays in our logs.
    functools.wraps keeps the name, docstring and type hints, which become the tool's schema.
    """

    @functools.wraps(function)
    def tool(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except ToolError as error:
            raise McpToolError(str(error)) from error

    return tool


def create_server() -> MCPServer:
    server = MCPServer(
        name="fmcg-sales-copilot",
        title="FMCG Sales Copilot",
        instructions=INSTRUCTIONS.format(latest_week=_latest_week()),
    )
    for tool in TOOLS:
        # The function's name, type hints and docstring become the tool's name, schema and
        # description; the pydantic return type becomes its structured output schema.
        server.tool(annotations=READ_ONLY)(expose(tool))
    return server


def main() -> None:
    configure_logging(stream=sys.stderr)  # stdout is reserved for the MCP protocol
    logger.info("starting MCP server on stdio")
    create_server().run(transport="stdio")


if __name__ == "__main__":
    main()
