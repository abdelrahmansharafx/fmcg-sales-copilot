"""MCP server tests: tool metadata, error translation, and a real stdio round trip."""

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest
import sqlalchemy as sa
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.server.mcpserver.exceptions import ToolError as McpToolError

from scripts.claude_desktop_config import SERVER_NAME, build_config, problems
from src.config import DB_URL, MODELS_DIR, ROOT
from src.errors import NotFoundError, ToolError
from src.mcp_server import create_server, expose
from src.tools import TOOLS, forecast_demand

DB_PATH = sa.engine.make_url(DB_URL).database
needs_data = pytest.mark.skipif(
    not (DB_PATH and os.path.exists(DB_PATH) and (MODELS_DIR / "model_meta.json").exists()),
    reason="needs data/sales.db and a trained model",
)
TOOL_NAMES = [
    "query_sales",
    "forecast_demand",
    "recommend_order",
    "promo_impact",
    "search_policies",
]


# --- tool metadata ----------------------------------------------------------------------------
def test_server_exposes_the_same_five_tools_as_the_agent():
    tools = asyncio.run(create_server().list_tools())
    assert [t.name for t in tools] == TOOL_NAMES == [t.__name__ for t in TOOLS]


def test_every_tool_has_a_description_a_schema_and_is_marked_read_only():
    for tool in asyncio.run(create_server().list_tools()):
        assert tool.description and "Args:" in tool.description, tool.name
        assert tool.input_schema["properties"], tool.name
        assert tool.output_schema, tool.name  # pydantic return types become structured output
        assert tool.annotations.read_only_hint is True, tool.name
        assert tool.annotations.destructive_hint is False, tool.name


def test_schemas_keep_the_types_and_required_arguments():
    tools = {t.name: t for t in asyncio.run(create_server().list_tools())}
    forecast = tools["forecast_demand"].input_schema
    assert set(forecast["properties"]) == {"outlet_id", "sku_id", "weeks", "as_of"}
    assert set(forecast["required"]) == {"outlet_id", "sku_id"}
    assert forecast["properties"]["weeks"]["type"] == "integer"
    assert tools["promo_impact"].input_schema["properties"]["promo_id"]["type"] == "integer"
    assert tools["query_sales"].input_schema["required"] == ["question"]


def test_server_instructions_describe_the_data_conventions():
    server = create_server()
    assert "LATEST WEEK" in server.instructions and "competitor" in server.instructions


# --- error translation ------------------------------------------------------------------------
def test_expected_tool_errors_become_readable_mcp_errors():
    def lookup(outlet_id: str) -> str:
        """Look up an outlet.

        Args:
            outlet_id: Outlet id.
        """
        raise NotFoundError(f"Unknown outlet_id '{outlet_id}'.")

    wrapped = expose(lookup)
    assert wrapped.__name__ == "lookup" and wrapped.__doc__ == lookup.__doc__
    with pytest.raises(McpToolError, match="Unknown outlet_id 'O999'"):
        wrapped("O999")
    with pytest.raises(McpToolError):  # the base ToolError is an expected failure too
        expose(lambda: (_ for _ in ()).throw(ToolError("bad input")))()


def test_real_crashes_are_not_translated():
    """A bug must stay a crash: generic message to the client, traceback only in the log."""

    def buggy() -> str:
        raise RuntimeError("internal detail")

    with pytest.raises(RuntimeError):
        expose(buggy)()


# --- Claude Desktop config helper ---------------------------------------------------------------
def test_claude_desktop_config_uses_absolute_paths_and_this_module():
    entry = build_config(ROOT)["mcpServers"][SERVER_NAME]
    assert Path(entry["command"]).is_absolute()
    assert entry["command"].endswith(".venv/bin/python")
    assert entry["args"] == ["-m", "src.mcp_server"]
    assert entry["env"]["PYTHONPATH"] == str(ROOT)
    json.dumps(build_config(ROOT))  # must be valid JSON to paste into the config file


def test_config_helper_reports_missing_pieces(tmp_path):
    messages = " ".join(problems(tmp_path))
    for expected in ("uv sync", "generate_data.py", "src.forecast.train", ".env"):
        assert expected in messages


# --- a real stdio round trip --------------------------------------------------------------------
def run_session(steps):
    """Start the server exactly as Claude Desktop would (own process, stdio, other directory)."""
    entry = build_config(ROOT)["mcpServers"][SERVER_NAME]
    params = StdioServerParameters(
        command=sys.executable,  # the test's own interpreter: it has the project's packages
        args=entry["args"],
        env={**os.environ, **entry["env"]},
        cwd="/",  # NOT the project folder: the server must find its files by itself
    )

    async def main():
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                init = await session.initialize()
                return await steps(session, init)

    return asyncio.run(asyncio.wait_for(main(), timeout=120))


@needs_data
def test_stdio_server_lists_tools_and_answers_calls():
    async def steps(session, init):
        names = [t.name for t in (await session.list_tools()).tools]
        forecast = await session.call_tool(
            "forecast_demand", {"outlet_id": "O001", "sku_id": "S01", "weeks": 2}
        )
        policy = await session.call_tool(
            "search_policies", {"question": "How far in advance must a promotion be requested?"}
        )
        missing = await session.call_tool("recommend_order", {"outlet_id": "O999"})
        return init, names, forecast, policy, missing

    init, names, forecast, policy, missing = run_session(steps)

    assert init.server_info.name == SERVER_NAME and "2025-12-22" in init.instructions
    assert names == TOOL_NAMES
    assert forecast.is_error is False
    expected = forecast_demand("O001", "S01", 2)
    assert forecast.structured_content["total_units"] == expected.total_units
    assert policy.structured_content["passages"][0]["source"] == "promotions_policy.md"
    # An expected failure comes back as a readable error the model can act on.
    assert missing.is_error is True
    assert "Unknown outlet_id 'O999'" in missing.content[0].text
