"""
scripts/ingest.py
=================
CLI entry point for loading, chunking, and indexing documents into ChromaDB.

Reads ``.txt`` and ``.md`` files from a source directory, splits them into
overlapping chunks, generates embeddings via a local Ollama model, and
persists the resulting vector store to disk for use by the RAG pipeline.

Entry point:
    uv run ingest-docs

Usage examples:
    uv run ingest-docs
    uv run ingest-docs --source src/llm_eval/datasets/raw --reset

Architecture:
    Embeddings : Ollama local (nomic-embed-text) — free, offline, fast
    Generation : Ollama Cloud (llama3.3:70b)     — heavy models without local hardware

Prerequisites:
    ollama pull nomic-embed-text   (274 MB, one-time download)

Configuration (resolved from ``src/llm_eval/shared/config.py``):
    CHROMA_PERSIST_DIR  : path where ChromaDB persists its SQLite + binary files
    CHROMA_COLLECTION   : collection name inside ChromaDB
    CHUNK_SIZE          : maximum token/character size per chunk
    CHUNK_OVERLAP       : overlap between consecutive chunks to preserve context
    OLLAMA_EMBED_MODEL  : embedding model name, e.g. ``"nomic-embed-text"``
    OLLAMA_BASE_URL     : Ollama daemon address, e.g. ``"http://localhost:11434"``

Dependencies:
    langchain-community, langchain-text-splitters, langchain-ollama,
    langchain-chroma, python-dotenv
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from src.llm_eval.shared.config import (
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    OLLAMA_EMBED_MODEL,
    OLLAMA_BASE_URL,
)


def get_embeddings() -> OllamaEmbeddings:

    """Instantiate the local Ollama embeddings model.

    Creates an ``OllamaEmbeddings`` instance pointed at the local Ollama
    daemon. This function is used both during ingestion (to generate and
    persist chunk embeddings) and at query time (to embed the user question
    before similarity search).

    Using the same model and base URL in both contexts ensures that query
    and document vectors live in the same embedding space — a requirement
    for meaningful cosine similarity scores.

    Returns:
        ``OllamaEmbeddings`` configured with ``OLLAMA_EMBED_MODEL`` and
        ``OLLAMA_BASE_URL`` from ``shared/config.py``.

    Note:
        The Ollama daemon must be running (``ollama serve``) and the model
        must be available locally (``ollama pull nomic-embed-text``) before
        calling this function.
    """

    return OllamaEmbeddings(
        model=OLLAMA_EMBED_MODEL,   # "nomic-embed-text" (config.py)
        base_url=OLLAMA_BASE_URL,   # "http://localhost:11434" (config.py)
    )


def ingest(source_dir: str | Path = "src/llm_eval/datasets/raw", reset: bool = False) -> int:

    """Load, chunk, embed, and persist documents into ChromaDB.

    Full ingestion pipeline:

    1. **Load** — discovers all ``.txt`` and ``.md`` files under ``source_dir``
       using two separate ``DirectoryLoader`` instances (one per extension).
       Glob patterns like ``**/*.{txt,md}`` are intentionally avoided because
       Python's ``pathlib`` does not expand brace expressions on Windows.

    2. **Split** — applies ``RecursiveCharacterTextSplitter`` with the
       separators ``["\\n\\n", "\\n", ". ", " ", ""]`` to produce overlapping
       chunks that respect natural text boundaries (paragraphs → sentences →
       words → characters).

    3. **Embed & persist** — generates embeddings for all chunks via the local
       Ollama model and writes them to the ChromaDB collection on disk.
       If ``reset=True``, the existing collection directory is deleted before
       writing so that stale chunks from previous runs are not retained.

    Args:
        source_dir: Path to the directory containing source documents.
                    Defaults to ``"src/llm_eval/datasets/raw"``.
        reset:      If ``True``, wipes the existing ChromaDB persist directory
                    before ingestion. Use when re-indexing updated documents.

    Returns:
        Total number of chunks stored in the ChromaDB collection after ingestion.

    Raises:
        FileNotFoundError: If ``source_dir`` does not exist.
        ValueError:        If no ``.txt`` or ``.md`` files are found under
                           ``source_dir``.
    """

    source_path = Path(source_dir)
    if not source_path.exists():
        raise FileNotFoundError(
            f"Document directory not found: {source_path.resolve()}\n"
            "Add .txt or .md files before ingesting."
        )

    print(f"\n[ingest] Loading documents from '{source_path}'...")
    start = time.monotonic()

    # Two separate loaders are required because glob brace expansion
    # (e.g. "**/*.{txt,md}") is not supported by pathlib on Windows.
    docs = []
    for pattern in ("**/*.txt", "**/*.md"):
        loader = DirectoryLoader(
            str(source_path),
            glob=pattern,
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            show_progress=False,
            use_multithreading=True,
        )
        docs.extend(loader.load())

    if not docs:
        raise ValueError(f"No .txt/.md documents found under '{source_path}'.")
    print(f" ✓ {len(docs)} document(s) loaded(s)")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f" ✓ {len(chunks)} chunks generated (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    persist_dir = Path(CHROMA_PERSIST_DIR)
    if reset and persist_dir.exists():
        shutil.rmtree(persist_dir)
        print(f" ✓ Collection '{CHROMA_COLLECTION}' removed (--reset)")

    print(f" Generating embeddings (Ollama local) and persisting in '{persist_dir}'...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=CHROMA_COLLECTION,
        persist_directory=str(persist_dir),
    )

    elapsed  = time.monotonic() - start
    n_chunks = vectorstore._collection.count()
    print(f" ✓ {n_chunks} Chunks in ChromaDB | {elapsed:.1f}s\n")
    return n_chunks


def get_vectorstore() -> Chroma:
    return Chroma(
        collection_name=CHROMA_COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_PERSIST_DIR,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest documents into ChromaDB.")
    parser.add_argument(
        "--source",
        default="src/llm_eval/datasets/raw",
        help="Folder with archiche .txt/.md (default: src/llm_eval/datasets/raw)",
    )
    parser.add_argument("--reset", action="store_true", help="Clean the collection before ingesting.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ingest(source_dir=args.source, reset=args.reset)


if __name__ == "__main__":
    main()
