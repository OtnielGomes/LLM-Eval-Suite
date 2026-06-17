"""
tests/conftest.py
Fixtures shared between all tests.
No test makes a real API call — everything is mocked here.
Src layout structure:

src/llm_eval/shared/types.py, config.py, ratelimiter.py
src/llm_eval/benchmark/evaluator.py
src/llm_eval/clients/ollama_cloud_cliente.py
src/llm_eval/rag/rag_pipeline.py
src/llm_eval/rag/strategies/naive.py, hyde.py, reranking.py
scripts/evaluate_rag.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT_DIR = Path(__file__).parent.parent
SRC_LLMEVAL = ROOT_DIR / "src" / "llm_eval"
TESTS_DIR = Path(__file__).parent

for path in [
    str(SRC_LLMEVAL / "shared"),        # types.py, config.py
    str(SRC_LLMEVAL / "benchmark"),     # evaluator.py
    str(SRC_LLMEVAL / "clients"),       # ollama_cloud_cliente.py
    str(SRC_LLMEVAL / "rag"),           # rag_pipeline.py
    str(SRC_LLMEVAL / "rag" / "strategies"),  # naive.py, hyde.py, reranking.py
    str(ROOT_DIR / "scripts"),          # evaluate_rag.py
]:
    if path not in sys.path:
        sys.path.insert(0, path)

# ---------------------------------------------------------------------------
from llm_eval.shared.types import JudgeScore, LLMResponse, RAGResult 
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def set_ollama_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Injects environment variables and clears module cache for all tests."""
    monkeypatch.setenv("OLLAMA_API_KEY", "test-fake-key-for-tests")
    monkeypatch.setenv("OLLAMA_CLOUD_MODEL", "llama3.3:70b")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")

    sys.modules.pop("rag_pipeline", None)

# ---------------------------------------------------------------------------
# Fixtures LLM
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_llm_response() -> LLMResponse:
    """LLMResponse dataclass with fields minimum expected by the project."""
    return LLMResponse(
        model="llama3.3:70b",
        prompt="What is ML?",
        response="Machine Learning is a subset of AI.",
        latency_ms=120.0,
    )

@pytest.fixture
def mock_judge_score() -> JudgeScore:
    """JudgeScore with values fixed for tests of evaluator."""
    return JudgeScore(score=4, reason="The answer is correct and concise.", is_correct=True)

# ---------------------------------------------------------------------------
# Fixtures RAG
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_documents():
    """LangChain Document List for retrieval testing."""
    from langchain_core.documents import Document
    return [
        Document(page_content="RAG combines retrieval with generation.", metadata={"source": "rag_overview.txt"}),
        Document(page_content="Faithfulness measures if claims are grounded in context.", metadata={"source": "rag_overview.txt"}),
        Document(page_content="HyDE generates a hypothetical answer before retrieval.", metadata={"source": "rag_overview.txt"}),
    ]

@pytest.fixture
def mock_vectorstore(mock_documents):
    """Mocked ChromaDB vectorstore — never accesses disk."""
    vs = MagicMock()
    vs.similarity_search.return_value = mock_documents
    vs.similarity_search_with_relevance_scores.return_value = [
        (doc, 0.85 - i * 0.05) for i, doc in enumerate(mock_documents)
    ]
    return vs

@pytest.fixture
def mock_rag_result() -> RAGResult:
    """RAGResult with complete fields for evaluate rag tests."""
    return RAGResult(
        question="What is RAG?",
        answer="RAG combines retrieval with generation to reduce hallucinations.",
        contexts=["RAG combines retrieval with generation."],
        ground_truth="RAG is a method that retrieves relevant documents before generating an answer.",
        strategy="naive",
        latency_ms=350.0,
    )

# ---------------------------------------------------------------------------
# Fixtures Dataset
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_jsonl(tmp_path: Path) -> Path:
    """JSONL temporary with 3 items MMLU valid."""
    import json
    data = [
        {"question": "What is 2+2?", "choices": ["3", "4", "5", "6"], "answer": "B", "subject": "math"},
        {"question": "Capital of France?", "choices": ["London", "Paris", "Berlin", "Rome"], "answer": "B", "subject": "geography"},
        {"question": "What is H2O?", "choices": ["Hydrogen", "Helium", "Water", "Oxygen"], "answer": "C", "subject": "chemistry"},
    ]
    path = tmp_path / "test_dataset.jsonl"
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    return path

@pytest.fixture
def tmp_qa_jsonl(tmp_path: Path) -> Path:
    """JSONL temporary with 2 pair Q&A valid for tests RAG."""
    import json
    data = [
        {"question": "What is RAG?", "ground_truth": "RAG combines retrieval with generation."},
        {"question": "What is faithfulness?", "ground_truth": "Faithfulness measures if claims are grounded in context."},
    ]
    path = tmp_path / "test_qa.jsonl"
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    return path