"""
shared/config.py
Global project constants — Ollama stack only.
"""
import os


# Ollama Local (embeddings + dev) 
# Requires local daemon: https://ollama.com/download
# Embedding model: ollama pull nomic-embed-text
OLLAMA_BASE_URL   = os.getenv("OLLAMA_BASE_URL",   "http://localhost:11434")
OLLAMA_LLM_MODEL  = os.getenv("OLLAMA_MODEL",       "llama3.1:8b")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")


# ── Ollama Cloud (Generation + RAG + Evaluation) ──────────────────────────────────
# Key: https://ollama.com/settings/keys
OLLAMA_CLOUD_BASE_URL = "https://ollama.com/v1"
OLLAMA_CLOUD_MODEL    = os.getenv("OLLAMA_CLOUD_MODEL", "llama3.3:70b")
OLLAMA_CLOUD_API_KEY  = os.getenv("OLLAMA_API_KEY", "")


# ── ChromaDB ──────────────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
CHROMA_COLLECTION  = "llm_eval_docs"


# ── RAG ───────────────────────────────────────────────────────────────────────
CHUNK_SIZE      = int(os.getenv("CHUNK_SIZE",      "1000"))
CHUNK_OVERLAP   = int(os.getenv("CHUNK_OVERLAP",   "200"))
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "5"))


# ── Avaliação — LLM-as-judge ──────────────────────────────────────────────────
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