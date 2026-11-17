"""Command line: uv run python -m src.agent "which Alexandria outlets dropped sales last month?"."""

from __future__ import annotations

import argparse
import json
import sys

from src.agent.graph import AgentError, run_agent
from src.errors import ToolError


def show_call(call) -> None:
    """Print one tool call in a compact, readable form."""
    args = ", ".join(f"{k}={v!r}" for k, v in call.args.items())
    status = "ok" if call.ok else "FAILED"
    print(f"  -> {call.name}({args})  [{status}, {call.seconds}s]")
    try:
        data = json.loads(call.output)
    except json.JSONDecodeError:
        data = {}
    if call.name == "query_sales" and call.ok and isinstance(data, dict):
        print(f"     SQL: {' '.join(data.get('sql', '').split())}")
        print(f"     rows returned: {data.get('row_count')}")
    elif not call.ok:
        print(f"     {call.output}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Ask the FMCG Sales Copilot a question.")
    parser.add_argument("question", help="your question, in quotes")
    parser.add_argument("--quiet", action="store_true", help="print only the answer")
    parser.add_argument("--max-steps", type=int, default=6, help="max rounds of tool use")
    args = parser.parse_args()

    try:
        result = run_agent(args.question, max_steps=args.max_steps)
    except (AgentError, ToolError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    if args.quiet:
        print(result.answer)
        return 0
    print(f"Question: {result.question}\n")
    if result.tool_calls:
        print("Tool calls:")
        for call in result.tool_calls:
            show_call(call)
        print()
    print(f"Answer:\n{result.answer}\n")
    note = " (step limit reached)" if result.hit_step_limit else ""
    print(
        f"[{len(result.tool_calls)} tool calls, {result.llm_calls} LLM calls, "
        f"{result.total_tokens:,} tokens, ~${result.cost_usd:.4f}, {result.seconds}s]{note}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
