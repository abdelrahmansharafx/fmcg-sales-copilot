"""Print the Claude Desktop config entry for this project's MCP server (with absolute paths).

    uv run python scripts/claude_desktop_config.py

Paste the printed "mcpServers" block into Claude Desktop's config file
(see guides/mcp_claude_desktop.md).
It only prints: it never edits your Claude Desktop config.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SERVER_NAME = "fmcg-sales-copilot"


def build_config(root: Path = ROOT) -> dict:
    """The mcpServers entry. We use the project's own Python so no PATH lookup (uv, python) is
    needed: desktop apps start with a minimal PATH and often cannot find uv."""
    python = root / ".venv" / "bin" / "python"
    return {
        "mcpServers": {
            SERVER_NAME: {
                "command": str(python),
                "args": ["-m", "src.mcp_server"],
                "env": {"PYTHONPATH": str(root)},  # so `src` is importable from any directory
            }
        }
    }


def problems(root: Path = ROOT) -> list[str]:
    """Things that would stop the server from starting; shown as warnings."""
    found = []
    if not (root / ".venv" / "bin" / "python").exists():
        found.append("No .venv found. Run: uv sync")
    if not (root / "data" / "sales.db").exists():
        found.append("No database. Run: uv run python scripts/generate_data.py")
    if not (root / "models" / "lgbm_forecast.joblib").exists():
        found.append("No trained model. Run: uv run python -m src.forecast.train")
    if not (root / ".env").exists():
        found.append("No .env: query_sales needs the LLM settings (the other 4 tools do not).")
    return found


if __name__ == "__main__":
    print(json.dumps(build_config(), indent=2))
    for message in problems():
        print(f"WARNING: {message}", file=sys.stderr)
