"""
src/llm_eval/shared/config.py
==============================
Global configuration constants for the llm-eval-suite project.

All values are resolved from environment variables with sensible defaults,
allowing the full stack to run out of the box without a ``.env`` file for
local development, while remaining fully configurable in CI/CD and
production environments.

Stack overview:
    - **Ollama Local**  : embeddings and local LLM inference (free, offline)
    - **Ollama Cloud**  : generation, RAG, and evaluation (free, hosted)
    - **ChromaDB**      : vector store for RAG document chunks (local, persistent)

Environment variable reference:

.. list-table::
   :header-rows: 1

   * - Variable
     - Default
     - Used by
   * - ``OLLAMA_BASE_URL``
     - ``http://localhost:11434``
     - ``ingest.py``, ``OllamaClient``
   * - ``OLLAMA_MODEL``
     - ``llama3.1:8b``
     - ``OllamaClient``
   * - ``OLLAMA_EMBED_MODEL``
     - ``nomic-embed-text``
     - ``ingest.py``, ``evaluate_rag.py``
   * - ``OLLAMA_CLOUD_MODEL``
     - ``llama3.3:70b``
     - ``OllamaCloudClient``, ``RAGPipeline``
   * - ``OLLAMA_API_KEY``
     - *(required for cloud)*
     - ``OllamaCloudClient``, ``RAGPipeline``
   * - ``CHROMA_PERSIST_DIR``
     - ``./chroma_db``
     - ``ingest.py``, ``RAGPipeline``
   * - ``CHUNK_SIZE``
     - ``1000``
     - ``ingest.py``
   * - ``CHUNK_OVERLAP``
     - ``200``
     - ``ingest.py``
   * - ``TOP_K_RETRIEVAL``
     - ``5``
     - All three RAG strategies

Warning:
    ``OLLAMA_CLOUD_API_KEY`` is read here at module import time and stored as
    a module-level constant. Code that needs to support ``monkeypatch.setenv``
    in tests (e.g. ``OllamaCloudClient``, ``RAGPipeline``) must read
    ``OLLAMA_API_KEY`` directly via ``os.getenv()`` at call time instead of
    importing this constant.

Prerequisites:
    - Ollama daemon running locally: https://ollama.com/download
    - Embedding model pulled: ``ollama pull nomic-embed-text``
    - Ollama Cloud API key: https://ollama.com/settings/keys
"""
import os


# ---------------------------------------------------------------------------
# Ollama Local — embeddings and local LLM dev/test
# ---------------------------------------------------------------------------

# Local daemon address. Override if Ollama is running in Docker or on a
# different host (e.g. OLLAMA_BASE_URL=http://ollama-service:11434).
OLLAMA_BASE_URL    = os.getenv("OLLAMA_BASE_URL",    "http://localhost:11434")

# Local generation model. Used by OllamaClient in run_benchmark.py --local.
# Pull with: ollama pull llama3.1:8b
OLLAMA_LLM_MODEL   = os.getenv("OLLAMA_MODEL",        "llama3.1:8b")

# Embedding model used by both ingest.py and evaluate_rag.py.
# Must be the SAME model in both contexts to avoid embedding space mismatch.
# Pull with: ollama pull nomic-embed-text  (274 MB, one-time download)
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL",  "nomic-embed-text")


# ---------------------------------------------------------------------------
# Ollama Cloud — generation, RAG, and evaluation
# ---------------------------------------------------------------------------

# Ollama Cloud OpenAI-compatible API endpoint. Not configurable via env var
# as it is a fixed infrastructure address for the Ollama Cloud service.
OLLAMA_CLOUD_BASE_URL = "https://ollama.com/api"

# Cloud generation model. llama3.3:70b provides strong reasoning quality
# without local hardware requirements. Override for cost/latency trade-offs.
OLLAMA_CLOUD_MODEL    = os.getenv("OLLAMA_CLOUD_MODEL", "llama3.3:70b")

# WARNING: this constant is frozen at module import time.
# Tests using monkeypatch.setenv("OLLAMA_API_KEY", ...) will NOT affect this
# value. Use os.getenv("OLLAMA_API_KEY") directly in code that must be
# patchable (OllamaCloudClient.__init__, RAGPipeline.__init__,
# _build_langchain_llm).
OLLAMA_CLOUD_API_KEY  = os.getenv("OLLAMA_API_KEY", "")


# ---------------------------------------------------------------------------
# ChromaDB — local vector store
# ---------------------------------------------------------------------------

# Directory where ChromaDB persists its SQLite database and binary index files.
# Delete this directory (or use ingest --reset) to rebuild the vector store.
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

# ChromaDB collection name. Changing this creates a new empty collection;
# the old collection remains on disk until CHROMA_PERSIST_DIR is cleared.
CHROMA_COLLECTION  = "llm_eval_docs"


# ---------------------------------------------------------------------------
# RAG chunking and retrieval
# ---------------------------------------------------------------------------

# Maximum characters per chunk. Larger values preserve more context per chunk
# but may exceed the LLM context window and dilute relevance scores.
CHUNK_SIZE      = int(os.getenv("CHUNK_SIZE",      "1000"))

# Character overlap between consecutive chunks. Prevents information loss at
# chunk boundaries by repeating the tail of each chunk at the start of the next.
CHUNK_OVERLAP   = int(os.getenv("CHUNK_OVERLAP",   "200"))

# Number of chunks retrieved per query across all three RAG strategies.
# RerankingRetriever fetches k*2 candidates before reranking down to TOP_K.
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "5"))


# ---------------------------------------------------------------------------
# LLM-as-judge evaluation rubric
# ---------------------------------------------------------------------------

# Shared prompt template used by OllamaClient.judge(), OllamaCloudClient.judge(),
# and GeminiClient.judge(). Centralising the rubric here ensures all three
# clients score responses on an identical 1–5 scale, making cross-client
# comparisons valid.
#
# {question}, {reference}, and {hypothesis} are filled in by each client's
# judge() method via str.format().
JUDGE_PROMPT_TEMPLATE = """You are an expert evaluator. Assess the Model Answer.

Question: {question}
Reference Answer: {reference}
Model Answer: {hypothesis}

Scoring rubric:
- 5: Fully correct and complete
- 4: Mostly correct, minor omissions
- 3: Partially correct, key gaps
- 2: Mostly wrong, some relevant content
- 1: Completely wrong or off-topic"""