"""
scripts/ingest.py
Entry point CLI: uv run ingest-docs

Usage:
uv run ingest-docs
uv run ingest-docs --source src/llm_eval/datasets/raw --reset
uv run ingest-docs --status

Hybrid Architecture:
Embeddings → Local Ollama (nomic-embed-text) — free, offline, fast
Generation → Cloud Ollama (llama3.3:70b)     — heavyweight models without hardware

Prerequisite:
ollama pull nomic-embed-text  (274MB, one-time)
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

# Resolve paths relative to the project root, not the current working directory
PROJECT_ROOT   = Path(__file__).parent.parent
DEFAULT_SOURCE = PROJECT_ROOT / "src" / "llm_eval" / "datasets" / "raw"


def get_embeddings() -> OllamaEmbeddings:
    """Local embeddings via Ollama daemon (http://localhost:11434)."""
    return OllamaEmbeddings(
        model=OLLAMA_EMBED_MODEL,
        base_url=OLLAMA_BASE_URL,
    )


def get_vectorstore() -> Chroma:
    """Return the existing vectorstore without re-indexing. Used by evaluate_rag.py."""
    persist_dir = Path(CHROMA_PERSIST_DIR)
    if not persist_dir.is_absolute():
        persist_dir = PROJECT_ROOT / persist_dir
    if not persist_dir.exists():
        raise FileNotFoundError(
            f"ChromaDB not found at: {persist_dir.resolve()}\n"
            "Run first: uv run ingest-docs"
        )
    return Chroma(
        collection_name=CHROMA_COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=str(persist_dir),
    )


def status() -> None:
    """Print information about the current ChromaDB collection."""
    try:
        vs = get_vectorstore()
        n  = vs._collection.count()
        print(f"\n ChromaDB status")
        print(f" ├─ persist_dir : {Path(CHROMA_PERSIST_DIR).resolve()}")
        print(f" ├─ collection  : {CHROMA_COLLECTION}")
        print(f" └─ chunks      : {n}\n")
    except FileNotFoundError as e:
        print(f"\n ⚠ {e}\n")


def ingest(source_dir: str | Path = DEFAULT_SOURCE, reset: bool = False) -> int:
    source_path = Path(source_dir)

    # Resolve relative paths from project root
    if not source_path.is_absolute():
        source_path = PROJECT_ROOT / source_path

    if not source_path.exists():
        raise FileNotFoundError(
            f"Document folder not found: {source_path.resolve()}\n"
            "Run first: uv run populate-raw-docs"
        )

    print(f"\n[ingest] Loading documents from '{source_path}'...")
    start = time.monotonic()

    # glob "**/*.{txt,md}" does not work on Windows — use two separate loaders
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
        raise ValueError(
            f"No .txt/.md files found in '{source_path}'.\n"
            "Run: uv run populate-raw-docs"
        )
    print(f" ✅ {len(docs)} document(s) loaded")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    # Drop empty/whitespace-only chunks — they cause HTTP 400 errors on Ollama
    chunks = [c for c in chunks if c.page_content.strip()]
    print(f" ✅ {len(chunks)} chunks generated (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    persist_dir = Path(CHROMA_PERSIST_DIR)

    # Resolve persist_dir relative to project root if needed
    if not persist_dir.is_absolute():
        persist_dir = PROJECT_ROOT / persist_dir

    if reset and persist_dir.exists():
        shutil.rmtree(persist_dir)
        print(f" ✅ Collection '{CHROMA_COLLECTION}' dropped (--reset)")

    print(f" Generating embeddings (Ollama local) and persisting to '{persist_dir}'...")

    embeddings  = get_embeddings()
    BATCH_SIZE  = 100
    vectorstore = None

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        pct   = min(i + BATCH_SIZE, len(chunks))
        print(f" [{pct:>5}/{len(chunks)}] embedding batch...", end="\r")

        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                collection_name=CHROMA_COLLECTION,
                persist_directory=str(persist_dir),
            )
        else:
            vectorstore.add_documents(batch)

    print()
    elapsed = time.monotonic() - start

    if vectorstore is None:
        raise RuntimeError("Ingest failed: no chunks were embedded.")

    n_chunks = vectorstore._collection.count()
    print(f" ✅ {n_chunks} chunks in ChromaDB | {elapsed:.1f}s")
    print(f" ✅ Next step: uv run evaluate-rag\n")
    return n_chunks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest documents into ChromaDB.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  uv run ingest-docs                        # index datasets/raw/\n"
            "  uv run ingest-docs --reset                # drop and re-index\n"
            "  uv run ingest-docs --source my/folder     # custom source\n"
            "  uv run ingest-docs --status               # inspect ChromaDB\n"
        ),
    )
    parser.add_argument(
        "--source",
        default=str(DEFAULT_SOURCE),
        help="Folder with .txt/.md files (default: datasets/raw/)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop the ChromaDB collection before re-indexing.",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print information about the current collection and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.status:
        status()
        return
    ingest(source_dir=args.source, reset=args.reset)


if __name__ == "__main__":
    main()