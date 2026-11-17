# Connect the MCP server to Claude Desktop

The MCP server exposes the same five tools as the API and the agent: `query_sales`,
`forecast_demand`, `recommend_order`, `promo_impact`, `search_policies`. Claude Desktop starts the
server itself over stdio; you do not run it by hand.

## One-off setup (macOS)

1. **Build the project** (skip what already exists):
   ```bash
   uv sync
   uv run python scripts/generate_data.py      # data/sales.db
   uv run python -m src.forecast.train         # models/
   ```
   `query_sales` also needs your LLM settings in `.env` (it asks the LLM to write SQL). The other
   four tools need no LLM.

2. **Print your config block** (absolute paths, already correct for this folder):
   ```bash
   uv run python scripts/claude_desktop_config.py
   ```
   It warns on stderr if something is missing (venv, database, model, `.env`).

3. **Open Claude Desktop's config file**: Claude Desktop > *Settings* > *Developer* > *Edit Config*.
   This opens `~/Library/Application Support/Claude/claude_desktop_config.json`.

4. **Paste the block.** If the file is empty or has no `mcpServers`, use the printed JSON as is. If it
   already has other servers, add only the `"fmcg-sales-copilot": { ... }` entry inside the existing
   `mcpServers` object (keep the JSON valid: commas between entries).

5. **Quit Claude Desktop completely** (`Cmd+Q`, not just close the window) and open it again.

6. **Check it connected.** In a new chat, open the tools menu (the sliders icon in the message box, or
   *Settings* > *Developer*): `fmcg-sales-copilot` should be listed as running with 5 tools. Labels vary
   a little between Claude Desktop versions.

## Demo script

Ask these in a new chat (Claude asks permission the first time it uses a tool: choose *Allow*):

| # | Prompt | Tool used | What you should see |
|---|---|---|---|
| 1 | "Forecast Coca-Cola Original 330ml can at outlet O001 for the next 4 weeks." | `forecast_demand` | about 357 units, typical error about 34% |
| 2 | "What should outlet O010 order next week? Show the top 5 lines." | `recommend_order` | 186 units in total; Coca-Cola Zero 330ml can first (54) |
| 3 | "Did promotion 2 on Coca-Cola Original 330ml can pay off?" | `promo_impact` | +35.4% uplift (95% CI 27.0% to 45.4%) |
| 4 | "Can I run a 30% discount, and how far ahead must I request a promotion?" | `search_policies` | above 25% needs Commercial Director approval; request at least 3 weeks ahead |
| 5 | "Which Alexandria outlet had the largest percentage drop in units over the last 4 weeks vs the 4 before?" | `query_sales` | Alexandria Cafe 115, about -34% (1,393 to 913 units) |
| 6 | "What was our market share versus Pepsi last quarter?" | none | it should say the data has no competitor information |

Tip for a live demo: have the numbers above open in another window, and point out that the SQL,
forecast and policy answers come from tools running on your machine, not from the model's memory.

## If it does not show up

- **Logs**: `~/Library/Logs/Claude/mcp-server-fmcg-sales-copilot.log` (this server's stderr, JSON lines)
  and `~/Library/Logs/Claude/mcp.log`.
- **Test the server without Claude Desktop** (should print the 5 tool names):
  ```bash
  uv run python -c "import asyncio; from src.mcp_server import create_server; print([t.name for t in asyncio.run(create_server().list_tools())])"
  ```
- **"spawn ENOENT" / server disconnected**: a path in the config is wrong. Re-run step 2 and paste again.
  The config deliberately uses the project's own `.venv/bin/python` (desktop apps often cannot find
  `uv` on their PATH). If you moved the folder, re-run step 2.
- **A tool answers with an error about the database or model**: build them (step 1).
- **`query_sales` says no LLM is configured**: the server reads `.env` from the project folder; check the
  Azure variables there.
- Invalid JSON in the config file is the most common cause of "nothing happens": paste it into
  `python3 -m json.tool` to check.

## Same server in Claude Code (optional, no restart needed)

```bash
claude mcp add fmcg-sales-copilot --env PYTHONPATH="$PWD" -- "$PWD/.venv/bin/python" -m src.mcp_server
```

## Security notes

- All five tools are read-only (they are declared so in the MCP tool annotations). `query_sales` can
  only run a single SELECT on a read-only connection with a row limit and timeout.
- stdio servers run with your user's permissions on your machine and have no network listener.
