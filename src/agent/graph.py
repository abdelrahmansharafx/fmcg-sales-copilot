"""The LangGraph agent: an LLM that can call the five tools in a loop.

          +-------+   tool calls?   +-------+
 START -> | agent | --------------> | tools |
          +-------+ <-------------- +-------+
              | no tool calls            (results go back to the agent)
              v
             END   (the final answer)

* `agent` node: the LLM reads the conversation and either answers or asks for tools.
* `tools` node: runs the requested tools. Failures become text the LLM can read and recover
  from ("ERROR: unknown outlet_id ...") instead of crashing the run.
* Step limit: after MAX_STEPS rounds of tool use the agent is called WITHOUT tools and told to
  answer with what it has, so a confused model cannot loop forever or burn tokens.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict

import openai
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from src import llm
from src.agent.prompt import SYSTEM_PROMPT
from src.errors import ToolError

logger = logging.getLogger(__name__)

MAX_STEPS = 6  # max rounds of tool use per question
MAX_TOOL_CHARS = 8000  # tool output shown to the LLM is cut to this size

BUDGET_MESSAGE = (
    "You have used all your tool calls for this question. Do not call tools again. Answer now "
    "using only the tool results above, and say clearly what you could not find out."
)


class AgentError(RuntimeError):
    """The agent could not run (LLM unreachable, bad credentials, ...). Message is user-facing."""


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    tool_rounds: int  # how many times the tools node has run


@dataclass
class ToolCallRecord:
    name: str
    args: dict[str, Any]
    ok: bool
    output: str  # what the LLM saw (possibly truncated)
    seconds: float


@dataclass
class AgentResult:
    question: str
    answer: str
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    hit_step_limit: bool = False
    model_name: str = ""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def cost_usd(self) -> float:
        """Estimated cost of this question, including LLM calls made inside tools."""
        return llm.estimate_cost_usd(self.prompt_tokens, self.completion_tokens)


def as_langchain_tool(function: Callable) -> StructuredTool:
    """Wrap a plain tool function so the LLM sees its name, description and typed arguments."""
    return StructuredTool.from_function(function, parse_docstring=True)


def format_tool_output(output: Any) -> str:
    """Serialise a tool result (pydantic model or plain value) for the LLM, with a size cap."""
    text = output.model_dump_json() if hasattr(output, "model_dump_json") else json.dumps(output)
    if len(text) > MAX_TOOL_CHARS:
        text = (
            text[:MAX_TOOL_CHARS]
            + f"... [TRUNCATED, {len(text) - MAX_TOOL_CHARS} more characters. "
            "Ask a narrower question, e.g. aggregate or add a LIMIT.]"
        )
    return text


def build_graph(model, tools: Sequence[StructuredTool], max_steps: int = MAX_STEPS):
    """Assemble and compile the agent graph. `model` needs .bind_tools() and .invoke()."""
    tools_by_name = {tool.name: tool for tool in tools}
    model_with_tools = model.bind_tools(list(tools))

    def agent(state: AgentState) -> dict:
        if state["tool_rounds"] >= max_steps:  # budget used up: force a final answer
            reply = model.invoke([*state["messages"], SystemMessage(content=BUDGET_MESSAGE)])
        else:
            reply = model_with_tools.invoke(state["messages"])
        return {"messages": [reply]}

    def run_tools(state: AgentState) -> dict:
        results = []
        for call in state["messages"][-1].tool_calls:
            results.append(execute_tool_call(call, tools_by_name))
        return {"messages": results, "tool_rounds": state["tool_rounds"] + 1}

    def route(state: AgentState) -> str:
        last = state["messages"][-1]
        wants_tools = isinstance(last, AIMessage) and last.tool_calls
        return "tools" if wants_tools and state["tool_rounds"] < max_steps else END

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent)
    graph.add_node("tools", run_tools)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()


def execute_tool_call(call: dict, tools_by_name: dict[str, StructuredTool]) -> ToolMessage:
    """Run one tool call. Any failure is returned to the LLM as text starting with 'ERROR:'."""
    name, args = call["name"], call["args"]
    started = time.perf_counter()
    ok = True
    if name not in tools_by_name:
        ok, content = False, f"ERROR: unknown tool '{name}'. Available: {sorted(tools_by_name)}."
    else:
        try:
            content = format_tool_output(tools_by_name[name].invoke(args))
        except ToolError as error:  # expected problems with a clear message
            ok, content = False, f"ERROR: {error}"
        except Exception as error:  # noqa: BLE001 - never let a tool crash the whole run
            logger.exception("tool %s crashed", name)
            ok, content = False, f"ERROR: {name} failed unexpectedly ({type(error).__name__})."
    message = ToolMessage(content=content, tool_call_id=call["id"], name=name)
    message.additional_kwargs["record"] = ToolCallRecord(
        name, args, ok, content, round(time.perf_counter() - started, 2)
    )
    return message


def system_prompt() -> str:
    """The system prompt with the latest data week and the SKU catalogue filled in."""
    from src.forecast.predict import get_engine, latest_week
    from src.tools.catalog import brand_names, sku_names

    catalogue = "\n".join(f"{sku_id}: {name}" for sku_id, name in sku_names().items())
    return SYSTEM_PROMPT.format(
        latest_week=latest_week(get_engine()).date().isoformat(),
        brands=", ".join(brand_names()),
        sku_catalogue=catalogue,
    )


def run_agent(
    question: str,
    *,
    model=None,
    tools: Sequence[Callable] | None = None,
    max_steps: int = MAX_STEPS,
    system: str | None = None,
) -> AgentResult:
    """Answer one question. By default uses the configured LLM and the five real tools."""
    if not question.strip():
        raise AgentError("The question is empty.")
    if tools is None:
        from src.tools import TOOLS

        tools = TOOLS
    graph = build_graph(
        model or llm.get_chat_model(), [as_langchain_tool(t) for t in tools], max_steps
    )

    started = time.perf_counter()
    initial = {
        "messages": [
            SystemMessage(content=system if system is not None else system_prompt()),
            HumanMessage(content=question),
        ],
        "tool_rounds": 0,
    }
    try:
        # The recursion limit is a hard backstop in case the graph itself ever loops.
        with llm.track_usage() as tool_usage:  # LLM calls made inside tools (text-to-SQL)
            final = graph.invoke(initial, {"recursion_limit": 2 * max_steps + 6})
    except openai.AuthenticationError as error:
        raise AgentError("The LLM rejected the credentials. Check the API key in .env.") from error
    except openai.RateLimitError as error:
        raise AgentError("The LLM rate limit was hit. Wait a moment and try again.") from error
    except openai.APIConnectionError as error:
        raise AgentError("Could not reach the LLM. Check the endpoint and your network.") from error
    except openai.APIStatusError as error:
        raise AgentError(
            f"The LLM returned an error ({error.status_code}): {error.message}"
        ) from error

    result = summarise(question, final, max_steps, time.perf_counter() - started)
    result.llm_calls += tool_usage.calls
    result.prompt_tokens += tool_usage.prompt_tokens
    result.completion_tokens += tool_usage.completion_tokens
    return result


def summarise(question: str, final: dict, max_steps: int, seconds: float) -> AgentResult:
    """Collect the answer, tool-call trace and token usage from the finished graph state."""
    messages = final["messages"]
    result = AgentResult(question=question, answer="", seconds=round(seconds, 2))
    for message in messages:
        if isinstance(message, ToolMessage):
            result.tool_calls.append(message.additional_kwargs["record"])
        elif isinstance(message, AIMessage):
            result.llm_calls += 1
            result.model_name = result.model_name or message.response_metadata.get("model_name", "")
            usage = message.usage_metadata or {}
            result.prompt_tokens += usage.get("input_tokens", 0)
            result.completion_tokens += usage.get("output_tokens", 0)
    last = messages[-1]
    result.answer = (last.content if isinstance(last, AIMessage) else "") or (
        "I could not produce an answer."
    )
    result.hit_step_limit = final["tool_rounds"] >= max_steps
    return result
