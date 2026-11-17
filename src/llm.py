"""Thin wrappers around the OpenAI / Azure OpenAI SDK. Provider choice lives in config.py."""

from __future__ import annotations

from openai import AzureOpenAI, OpenAI

from src.config import LLMSettings, load_llm_settings
from src.errors import ToolError


class LLMNotConfiguredError(ToolError):
    """Raised when a tool needs the LLM but no API key is configured."""


def get_client(settings: LLMSettings | None = None) -> OpenAI:
    """Return an OpenAI or Azure OpenAI client, depending on the configured provider."""
    settings = settings or load_llm_settings()
    if settings.provider == "azure":
        return AzureOpenAI(
            api_key=settings.api_key,
            azure_endpoint=settings.azure_endpoint,
            api_version=settings.azure_api_version,
        )
    if settings.provider == "openai":
        return OpenAI(api_key=settings.api_key)
    raise LLMNotConfiguredError(
        "No LLM configured. Set OPENAI_API_KEY, or AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT "
        "(see .env.example)."
    )


def complete(messages: list[dict[str, str]], temperature: float = 0.0) -> str:
    """Send chat messages and return the reply text (temperature 0 = most deterministic)."""
    settings = load_llm_settings()
    response = get_client(settings).chat.completions.create(
        model=settings.chat_model, messages=messages, temperature=temperature
    )
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
