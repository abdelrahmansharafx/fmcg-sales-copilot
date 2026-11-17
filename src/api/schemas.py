"""Request and response models of the API (the tool results are reused from src.tools.schemas)."""

from __future__ import annotations

from datetime import date
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

# Ids are short codes like O001 / S01. Restricting the characters keeps junk out of logs and SQL.
ID_PATTERN = r"^[A-Za-z0-9_-]{1,20}$"
IdStr = Annotated[str, StringConstraints(pattern=ID_PATTERN)]


class AskRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(
        min_length=1,
        max_length=1000,
        examples=["Which Alexandria outlets dropped sales last month?"],
    )


class ToolCallInfo(BaseModel):
    name: str
    args: dict
    ok: bool
    seconds: float
    sql: str | None = Field(None, description="The SQL that was run, for query_sales calls.")


class UsageInfo(BaseModel):
    llm_calls: int
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float = Field(description="Estimated from token counts and list prices.")


class AskResponse(BaseModel):
    request_id: str
    answer: str
    tool_calls: list[ToolCallInfo]
    usage: UsageInfo
    seconds: float
    hit_step_limit: bool = Field(description="True if the tool-call budget ran out.")


class ForecastRequest(BaseModel):
    outlet_id: IdStr = Field(examples=["O001"])
    sku_id: IdStr = Field(examples=["S01"])
    weeks: int = Field(4, ge=1, le=4, description="Weeks ahead (the model's horizon is 4).")
    as_of: date | None = Field(
        None, description="Forecast as if today were this date (a backtest). Default: latest data."
    )


class HealthResponse(BaseModel):
    status: str = Field(description="ok, degraded (no LLM configured) or unavailable")
    version: str
    database: bool
    model: bool
    policy_index: bool | None = Field(None, description="Warmed at startup; None = not checked.")
    llm_provider: str


class ErrorResponse(BaseModel):
    detail: str
    request_id: str | None = None
