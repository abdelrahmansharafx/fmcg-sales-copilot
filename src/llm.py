"""Thin wrappers around the OpenAI / Azure OpenAI SDK. Provider choice lives in config.py."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from openai import AzureOpenAI, OpenAI

from src.config import LLMSettings, llm_prices_per_million, load_llm_settings
from src.errors import ToolError


class LLMNotConfiguredError(ToolError):
    """Raised when a tool needs the LLM but no API key is configured."""


def get_client(settings: LLMSettings | None = None) -> OpenAI:
    """Return the OpenAI-compatible client for the configured provider.

    Azure normally uses the "v1" endpoint with the plain OpenAI client (no api-version);
    if AZURE_OPENAI_API_VERSION holds a real version we use the classic AzureOpenAI client.
    """
    settings = settings or load_llm_settings()
    if settings.provider == "azure":
        if settings.azure_api_version:
            return AzureOpenAI(
                api_key=settings.api_key,
                azure_endpoint=settings.azure_base_url.removesuffix("openai/v1/"),
                api_version=settings.azure_api_version,
            )
        return OpenAI(api_key=settings.api_key, base_url=settings.azure_base_url)
    if settings.provider == "openai":
        return OpenAI(api_key=settings.api_key)
    raise LLMNotConfiguredError(
        "No LLM configured. Set AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT, or OPENAI_API_KEY "
        "(see .env.example)."
    )


def get_chat_model(settings: LLMSettings | None = None):
    """LangChain chat model for the agent, using the same provider settings as get_client()."""
    from langchain_openai import AzureChatOpenAI, ChatOpenAI

    settings = settings or load_llm_settings()
    if settings.provider == "azure":
        if settings.azure_api_version:
            return AzureChatOpenAI(
                api_key=settings.api_key,
                azure_endpoint=settings.azure_base_url.removesuffix("openai/v1/"),
                api_version=settings.azure_api_version,
                azure_deployment=settings.chat_model,
                temperature=0,
            )
        return ChatOpenAI(
            api_key=settings.api_key,
            base_url=settings.azure_base_url,
            model=settings.chat_model,
            temperature=0,
        )
    if settings.provider == "openai":
        return ChatOpenAI(api_key=settings.api_key, model=settings.chat_model, temperature=0)
    raise LLMNotConfiguredError(
        "No LLM configured. Set AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT, or OPENAI_API_KEY."
    )


@dataclass
class Usage:
    """Tokens used by the LLM calls made through complete() while a meter is active."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


_meter: ContextVar[Usage | None] = ContextVar("llm_usage_meter", default=None)


@contextmanager
def track_usage() -> Iterator[Usage]:
    """Count the tokens of every complete() call inside the `with` block.

    The agent wraps a request in this so LLM calls made INSIDE tools (text-to-SQL) are billed
    to the request too, not just the agent's own calls.
    """
    usage = Usage()
    token = _meter.set(usage)
    try:
        yield usage
    finally:
        _meter.reset(token)


def estimate_cost_usd(prompt_tokens: int, completion_tokens: int) -> float:
    """Estimated USD cost of the given token counts (see llm_prices_per_million)."""
    price_in, price_out = llm_prices_per_million()
    return (prompt_tokens * price_in + completion_tokens * price_out) / 1_000_000


def complete(
    messages: list[dict[str, str]], temperature: float = 0.0, json_mode: bool = False
) -> str:
    """Send chat messages and return the reply text (temperature 0 = most deterministic).

    json_mode=True asks the model to reply with a single JSON object.
    """
    settings = load_llm_settings()
    options = {"response_format": {"type": "json_object"}} if json_mode else {}
    response = get_client(settings).chat.completions.create(
        model=settings.chat_model, messages=messages, temperature=temperature, **options
    )
    meter = _meter.get()
    if meter is not None and response.usage is not None:
        meter.calls += 1
        meter.prompt_tokens += response.usage.prompt_tokens
        meter.completion_tokens += response.usage.completion_tokens
    return response.choices[0].message.content or ""


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Turn texts into embedding vectors with the configured backend.

    "azure" / "openai": the provider's embedding model. "local": Chroma's bundled MiniLM model
    (runs offline on CPU after a one-off download), so RAG works without any API key.
    """
    settings = load_llm_settings()
    if settings.embedding_provider == "local":
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

        return [list(map(float, vector)) for vector in DefaultEmbeddingFunction()(texts)]

    client = get_client(settings)
    vectors: list[list[float]] = []
    for start in range(0, len(texts), 64):  # the API accepts batches of texts
        response = client.embeddings.create(
            model=settings.embedding_model, input=texts[start : start + 64]
        )
        vectors += [item.embedding for item in response.data]
    return vectors


def embedding_backend_name() -> str:
    """A short label for the embedding backend, used to keep vector indexes apart."""
    settings = load_llm_settings()
    if settings.embedding_provider == "local":
        return "local-minilm"
    return f"{settings.embedding_provider}-{settings.embedding_model}".replace("/", "_")


def main() -> None:
    """Connection test: `uv run python -m src.llm`."""
    settings = load_llm_settings()
    print(f"provider={settings.provider} model/deployment={settings.chat_model}")
    if settings.provider == "azure":
        print(
            f"base_url={settings.azure_base_url} api_version={settings.azure_api_version or 'v1'}"
        )
    print(f"embeddings={settings.embedding_provider}")
    response = get_client(settings).chat.completions.create(
        model=settings.chat_model,
        messages=[{"role": "user", "content": "Reply with exactly: connection ok"}],
    )
    usage = response.usage
    print("reply:", response.choices[0].message.content)
    print(f"tokens: prompt={usage.prompt_tokens} completion={usage.completion_tokens}")


if __name__ == "__main__":
    main()
