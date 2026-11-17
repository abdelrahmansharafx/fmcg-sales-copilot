"""Tool 5: RAG over the policy documents in docs/ (chunk -> embed -> store -> retrieve).

Indexing (build_index):
  1. Read every docs/*.md file and split it into chunks, one per section (## heading). Long
     sections are split further on blank lines / table rows so chunks stay small and focused.
  2. Embed each chunk (src/llm.py chooses Azure OpenAI, OpenAI or the offline local model).
  3. Store text, vector and metadata (file, section) in a local Chroma collection.
  The collection name contains a hash of the documents and the embedding backend, so editing
  a document or switching embedding models automatically builds a fresh index.

Retrieval (search_policies): embed the question, return the top-k most similar chunks with
their source file and section. The agent writes the answer from these passages and cites them.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import chromadb

from src import llm
from src.config import CHROMA_DIR, DOCS_DIR
from src.errors import ToolError
from src.tools.schemas import PolicyPassage, PolicySearchResult

MAX_CHUNK_CHARS = 900
TOP_K = 6


@dataclass
class Chunk:
    source: str  # file name
    section: str  # heading
    text: str


def split_long(text: str, limit: int = MAX_CHUNK_CHARS) -> list[str]:
    """Split text into pieces under `limit` characters, breaking only at line boundaries."""
    pieces, current = [], ""
    for line in text.splitlines():
        if current and len(current) + len(line) + 1 > limit:
            pieces.append(current.strip())
            current = ""
        current += line + "\n"
    if current.strip():
        pieces.append(current.strip())
    return pieces


def chunk_document(path: Path) -> list[Chunk]:
    """One chunk per '## ' section (long ones are split); the document title is kept as context."""
    markdown = path.read_text()
    title = re.search(r"^# (.+)$", markdown, flags=re.MULTILINE)
    title_text = title.group(1) if title else path.stem
    sections = re.split(r"^## ", markdown, flags=re.MULTILINE)[1:]  # drop the preamble
    chunks = []
    for section in sections:
        heading, _, body = section.partition("\n")
        for piece in split_long(body.strip()):
            chunks.append(Chunk(path.name, f"{title_text} / {heading.strip()}", piece))
    return chunks


def load_chunks(docs_dir: Path = DOCS_DIR) -> list[Chunk]:
    files = sorted(docs_dir.glob("*.md"))
    if not files:
        raise ToolError(f"No policy documents found in {docs_dir}.")
    return [chunk for path in files for chunk in chunk_document(path)]


def _collection_name(chunks: list[Chunk]) -> str:
    digest = hashlib.sha256("".join(c.source + c.text for c in chunks).encode()).hexdigest()[:8]
    return f"policies_{llm.embedding_backend_name()}_{digest}".replace(".", "_")[:63]


@lru_cache
def build_index(docs_dir: Path = DOCS_DIR, chroma_dir: Path = CHROMA_DIR):
    """Create (or reuse) the Chroma collection holding the embedded policy chunks."""
    chunks = load_chunks(docs_dir)
    client = chromadb.PersistentClient(path=str(chroma_dir))
    name = _collection_name(chunks)
    existing = [c.name for c in client.list_collections()]
    if name in existing:
        return client.get_collection(name)

    for old in existing:  # remove indexes of older document versions / embedding models
        if old.startswith("policies_"):
            client.delete_collection(old)
    # We pass our own vectors, so Chroma needs no embedding function. Cosine distance suits
    # text embeddings (it compares direction, not length).
    collection = client.create_collection(
        name, embedding_function=None, metadata={"hnsw:space": "cosine"}
    )
    texts = [f"{c.section}\n{c.text}" for c in chunks]  # heading helps the embedding
    collection.add(
        ids=[f"{c.source}#{i}" for i, c in enumerate(chunks)],
        documents=[c.text for c in chunks],
        embeddings=llm.embed_texts(texts),
        metadatas=[{"source": c.source, "section": c.section} for c in chunks],
    )
    return collection


def search_policies(question: str, top_k: int = TOP_K) -> PolicySearchResult:
    """Search the company policy documents and return the most relevant passages with sources.

    Use this for questions about rules and policies: promotion rules (discount levels,
    duration, approvals), trade terms by channel (payment terms, rebates, delivery, safety
    stock), pricing guidelines and price-increase history, and new-outlet onboarding. Base
    the answer only on the returned passages and cite the source file. Ask ONE focused
    question per call: a question mixing two topics retrieves worse, so call it twice instead.

    Args:
        question: The policy question in plain English.
        top_k: How many passages to return (default 6).
    """
    if not question.strip():
        raise ToolError("The question is empty.")
    collection = build_index()
    query_vector = llm.embed_texts([question])
    result = collection.query(
        query_embeddings=query_vector,
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    passages = [
        PolicyPassage(
            source=meta["source"],
            section=meta["section"],
            text=document,
            score=round(1 - distance, 3),  # cosine distance -> similarity
        )
        for document, meta, distance in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0], strict=True
        )
    ]
    return PolicySearchResult(question=question, passages=passages)
