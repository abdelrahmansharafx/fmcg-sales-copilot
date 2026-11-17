"""Agent logic tests with a scripted fake LLM and fake tools (no network, no database)."""

import pytest
from langchain_core.messages import AIMessage

from src.agent.graph import (
    AgentError,
    as_langchain_tool,
    execute_tool_call,
    format_tool_output,
    run_agent,
)
from src.errors import ToolError


class ScriptedModel:
    """Stands in for the LLM: returns the prepared replies in order, and records what it saw."""

    def __init__(self, *replies: AIMessage):
        self.replies = list(replies)
        self.seen: list[list] = []
        self.tools_bound = False

    def bind_tools(self, tools):
        bound = ScriptedModel.__new__(ScriptedModel)
        bound.__dict__ = self.__dict__  # share the script and the log
        bound.tools_bound = True
        return bound

    def invoke(self, messages):
        self.seen.append(list(messages))
        return self.replies.pop(0)


def call(name: str, args: dict, call_id: str = "c1") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def final(text: str) -> AIMessage:
    return AIMessage(
        content=text, usage_metadata={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}
    )


# Fake tools (plain functions with the same docstring style as the real ones).
def add_numbers(a: int, b: int) -> int:
    """Add two numbers.

    Args:
        a: First number.
        b: Second number.
    """
    return a + b


def lookup_outlet(outlet_id: str) -> dict:
    """Look up an outlet.

    Args:
        outlet_id: Outlet id such as "O001".
    """
    if outlet_id != "O001":
        raise ToolError(f"Unknown outlet_id '{outlet_id}'.")
    return {"outlet_id": "O001", "region": "Cairo"}


def crash(x: str) -> str:
    """Always crashes.

    Args:
        x: Anything.
    """
    raise RuntimeError("secret internal detail")


TOOLS = [add_numbers, lookup_outlet, crash]


def run(model: ScriptedModel, question: str = "q", **kwargs):
    return run_agent(question, model=model, tools=TOOLS, system="You are a test.", **kwargs)


def test_agent_calls_a_tool_then_answers():
    model = ScriptedModel(call("add_numbers", {"a": 2, "b": 3}), final("The sum is 5."))
    result = run(model)
    assert result.answer == "The sum is 5."
    assert [(c.name, c.args, c.ok) for c in result.tool_calls] == [
        ("add_numbers", {"a": 2, "b": 3}, True)
    ]
    assert result.tool_calls[0].output == "5"
    # The model's second turn saw the tool result.
    assert any(getattr(m, "content", None) == "5" for m in model.seen[1])
    assert result.llm_calls == 2
    assert (result.prompt_tokens, result.completion_tokens) == (100, 20)


def test_answer_without_tools_makes_no_tool_calls():
    result = run(ScriptedModel(final("Hello!")))
    assert result.tool_calls == []
    assert result.answer == "Hello!"


def test_tool_errors_are_given_to_the_model_which_can_recover():
    model = ScriptedModel(
        call("lookup_outlet", {"outlet_id": "O999"}, "c1"),
        call("lookup_outlet", {"outlet_id": "O001"}, "c2"),
        final("O001 is in Cairo."),
    )
    result = run(model)
    assert [c.ok for c in result.tool_calls] == [False, True]
    assert result.tool_calls[0].output == "ERROR: Unknown outlet_id 'O999'."
    assert result.answer == "O001 is in Cairo."


def test_unknown_tool_and_crashing_tool_do_not_stop_the_run():
    model = ScriptedModel(
        call("does_not_exist", {}, "c1"), call("crash", {"x": "a"}, "c2"), final("done")
    )
    result = run(model)
    assert result.answer == "done"
    assert "unknown tool" in result.tool_calls[0].output
    assert result.tool_calls[1].output.startswith("ERROR: crash failed unexpectedly")
    assert "secret internal detail" not in result.tool_calls[1].output  # no internals leaked


def test_step_limit_forces_a_final_answer_without_tools():
    looping = [call("add_numbers", {"a": 1, "b": 1}, f"c{i}") for i in range(10)]
    model = ScriptedModel(*looping[:3], final("Best effort answer."))
    result = run(model, max_steps=3)
    assert result.hit_step_limit
    assert len(result.tool_calls) == 3  # exactly max_steps rounds of tools
    assert result.answer == "Best effort answer."
    # The last model call was told to stop using tools.
    assert "used all your tool calls" in model.seen[-1][-1].content


def test_step_limit_still_ends_if_the_model_ignores_the_instruction():
    """Even if the model keeps asking for tools after the budget, the graph must end."""
    model = ScriptedModel(*[call("add_numbers", {"a": 1, "b": 1}, f"c{i}") for i in range(10)])
    result = run(model, max_steps=2)
    assert result.hit_step_limit
    assert len(result.tool_calls) == 2
    assert result.answer  # a non-empty fallback text


def test_empty_question_is_rejected():
    with pytest.raises(AgentError):
        run_agent("  ", model=ScriptedModel(), tools=TOOLS, system="x")


def test_tool_output_is_truncated_for_the_llm():
    big = format_tool_output({"rows": ["x" * 100] * 500})
    assert len(big) < 8500
    assert "TRUNCATED" in big


def test_real_tools_become_llm_tools_with_descriptions():
    from src.tools import TOOLS as REAL_TOOLS

    wrapped = [as_langchain_tool(t) for t in REAL_TOOLS]
    assert [t.name for t in wrapped] == [t.__name__ for t in REAL_TOOLS]
    for tool in wrapped:
        assert tool.description and tool.args  # description and typed arguments
    assert set(wrapped[1].args) == {"outlet_id", "sku_id", "weeks", "as_of"}


def test_execute_tool_call_returns_error_text_not_exceptions():
    tools = {t.name: t for t in map(as_langchain_tool, TOOLS)}
    message = execute_tool_call(
        {"name": "lookup_outlet", "args": {"outlet_id": "X"}, "id": "1"}, tools
    )
    assert message.content.startswith("ERROR:")


def test_system_prompt_is_filled_with_live_facts():
    from src.agent.graph import system_prompt

    prompt = system_prompt()
    assert "{" not in prompt  # every placeholder was filled
    assert "2025-12-22" in prompt  # latest data week
    assert "S01: Coca-Cola Original 330ml can" in prompt
    assert "Coca-Cola" in prompt and "NO competitor sales" in prompt
