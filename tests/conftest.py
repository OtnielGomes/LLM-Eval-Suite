"""
tests/conftest.py
=================
Shared pytest fixtures for the llm-eval-suite test suite.

No test in this suite makes a real API call — every LLM client, vector store,
and external service is mocked here. This file is the single place to update
when adding new fixtures or changing mock behaviour.

Project layout (src/ layout with editable install via uv sync):
    src/llm_eval/shared/         → types.py, config.py, rate_limiter.py
    src/llm_eval/benchmark/      → evaluator.py
    src/llm_eval/clients/        → gemini_cliente.py, ollama_cloud_cliente.py
    src/llm_eval/rag/            → rag_pipeline.py
    src/llm_eval/rag/strategies/ → naive.py, hyde.py, reranking.py
    scripts/                     → evaluate_rag.py

sys.path strategy:
    All sub-package directories are registered once here. Individual test
    files must NOT add their own ``sys.path.insert`` calls — centralising
    path registration here prevents ordering-dependent import bugs.

Environment variable strategy:
    The ``set_ollama_env`` autouse fixture injects all required environment
    variables before every test and clears the ``rag_pipeline`` module cache
    so that lazy ``os.getenv()`` calls inside ``RAGPipeline.__init__`` and
    ``_build_langchain_llm`` read the injected values rather than stale
    import-time constants.

Fixture inventory:

.. list-table::
   :header-rows: 1

   * - Fixture
     - Scope
     - Description
   * - ``set_ollama_env``
     - function (autouse)
     - Injects env vars and clears module cache for every test
   * - ``mock_gemini_response``
     - function
     - Simulates a ``GenerateContentResponse`` from the google-genai SDK
   * - ``mock_llm_response``
     - function
     - ``LLMResponse`` dataclass with minimal valid fields
   * - ``mock_judge_score``
     - function
     - ``JudgeScore`` with fixed values for ``Evaluator`` tests
   * - ``mock_documents``
     - function
     - List of 3 LangChain ``Document`` objects for retrieval tests
   * - ``mock_vectorstore``
     - function
     - ``MagicMock`` ChromaDB instance, never touches disk
   * - ``mock_rag_result``
     - function
     - Complete ``RAGResult`` for ``evaluate_rag.py`` tests
   * - ``tmp_jsonl``
     - function
     - Temporary JSONL file with 3 valid MMLU items
   * - ``tmp_qa_jsonl``
     - function
     - Temporary JSONL file with 2 valid RAG QA pairs
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# sys.path registration
# ---------------------------------------------------------------------------

ROOT_DIR    = Path(__file__).parent.parent
SRC_LLMEVAL = ROOT_DIR / "src" / "llm_eval"
TESTS_DIR   = Path(__file__).parent

# Register all sub-packages once. Tests must not add their own sys.path entries.
for _path in [
    str(SRC_LLMEVAL / "shared"),              # types.py, config.py, rate_limiter.py
    str(SRC_LLMEVAL / "benchmark"),           # evaluator.py
    str(SRC_LLMEVAL / "clients"),             # gemini_cliente.py, ollama_cloud_cliente.py
    str(SRC_LLMEVAL / "rag"),                 # rag_pipeline.py
    str(SRC_LLMEVAL / "rag" / "strategies"),  # naive.py, hyde.py, reranking.py
    str(ROOT_DIR / "scripts"),                # evaluate_rag.py
]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

# ---------------------------------------------------------------------------
# Central imports — use the installed package for canonical types
# ---------------------------------------------------------------------------

from llm_eval.shared.types import JudgeScore, LLMResponse, RAGResult  # noqa: E402


# ---------------------------------------------------------------------------
# Environment fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def set_ollama_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject required environment variables and reset the rag_pipeline module cache.

    Applied automatically to every test via ``autouse=True``. Prevents
    ``EnvironmentError`` and ``ValueError`` from ``OllamaCloudClient.__init__``
    and ``RAGPipeline.__init__``, which validate ``OLLAMA_API_KEY`` at
    instantiation time.

    The ``sys.modules.pop("rag_pipeline")`` call is required because
    ``rag_pipeline`` may have been cached from an earlier import with
    different environment values. Removing it from the cache forces Python
    to re-execute the module on the next import, ensuring that the lazy
    ``os.getenv()`` calls inside ``_build_langchain_llm`` and
    ``RAGPipeline.__init__`` read the values injected by this fixture rather
    than stale import-time constants.

    Environment variables injected:

    - ``OLLAMA_API_KEY``     : placeholder key that satisfies the non-empty
                               check in ``OllamaCloudClient`` and ``RAGPipeline``.
    - ``OLLAMA_CLOUD_MODEL`` : model name forwarded to ``ChatOpenAI``.
    - ``LANGSMITH_TRACING``  : set to ``"false"`` to disable tracing overhead
                               in tests.

    Args:
        monkeypatch: pytest built-in fixture; changes are automatically
                     reverted after each test.
    """
    monkeypatch.setenv("OLLAMA_API_KEY",     "test-fake-key-for-tests")
    monkeypatch.setenv("OLLAMA_CLOUD_MODEL", "llama3.3:70b")
    monkeypatch.setenv("LANGSMITH_TRACING",  "false")
    # Force re-import of rag_pipeline so lazy os.getenv() reads the patched values
    sys.modules.pop("rag_pipeline", None)


# ---------------------------------------------------------------------------
# LLM response fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_gemini_response() -> MagicMock:
    """Simulate a ``GenerateContentResponse`` from the google-genai SDK.

    Provides the minimum structure expected by ``GeminiClient.complete()``:
    a ``text`` attribute for the response content and a ``candidates`` list
    with a ``finish_reason`` field for stop-reason inspection.

    Returns:
        ``MagicMock`` with:

        - ``mock.text``                             : ``"The answer is A."``
        - ``mock.candidates[0].finish_reason``      : ``"STOP"``
    """
    mock = MagicMock()
    mock.text = "The answer is A."
    mock.candidates = [MagicMock()]
    mock.candidates[0].finish_reason = "STOP"
    return mock


@pytest.fixture()
def mock_llm_response() -> LLMResponse:
    """Return a minimal valid ``LLMResponse`` for client and evaluator tests.

    Uses realistic field values to avoid masking bugs that would appear with
    empty strings or zero latency.

    Returns:
        ``LLMResponse`` with model ``"llama3.3:70b"``, a short response, and
        ``latency_ms=120.0``.
    """
    return LLMResponse(
        model="llama3.3:70b",
        prompt="What is ML?",
        response="Machine Learning is a subset of AI.",
        latency_ms=120.0,
    )


@pytest.fixture()
def mock_judge_score() -> JudgeScore:
    """Return a fixed ``JudgeScore`` for ``Evaluator`` unit tests.

    Score ``4`` with ``is_correct=True`` represents a realistic "mostly
    correct" evaluation, avoiding edge-case values (0, 1, 5) that could
    mask aggregation bugs in ``Evaluator.summarize()``.

    Returns:
        ``JudgeScore(score=4, reason="The answer is correct and concise.", is_correct=True)``
    """
    return JudgeScore(
        score=4,
        reason="The answer is correct and concise.",
        is_correct=True,
    )


# ---------------------------------------------------------------------------
# RAG fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_documents():
    """Return a list of 3 LangChain ``Document`` objects for retrieval tests.

    Documents cover three distinct RAG concepts (RAG overview, faithfulness,
    HyDE) to allow tests to assert on retrieved content without ambiguity.
    All share the same ``source`` metadata to simplify metadata assertions.

    Returns:
        ``list[Document]`` with 3 items, each with ``page_content`` and
        ``metadata={"source": "rag_overview.txt"}``.
    """
    from langchain_core.documents import Document
    return [
        Document(
            page_content="RAG combines retrieval with generation.",
            metadata={"source": "rag_overview.txt"},
        ),
        Document(
            page_content="Faithfulness measures if claims are grounded in context.",
            metadata={"source": "rag_overview.txt"},
        ),
        Document(
            page_content="HyDE generates a hypothetical answer before retrieval.",
            metadata={"source": "rag_overview.txt"},
        ),
    ]


@pytest.fixture()
def mock_vectorstore(mock_documents):
    """Return a ``MagicMock`` ChromaDB vector store that never touches disk.

    Configures both retrieval methods with deterministic return values:

    - ``similarity_search`` returns the full ``mock_documents`` list.
    - ``similarity_search_with_relevance_scores`` returns ``(doc, score)``
      tuples with decreasing scores (``0.85``, ``0.80``, ``0.75``) to
      simulate a realistic ranked result.

    Args:
        mock_documents: Injected via fixture dependency.

    Returns:
        ``MagicMock`` with ``similarity_search`` and
        ``similarity_search_with_relevance_scores`` pre-configured.
    """
    vs = MagicMock()
    vs.similarity_search.return_value = mock_documents
    vs.similarity_search_with_relevance_scores.return_value = [
        (doc, 0.85 - i * 0.05) for i, doc in enumerate(mock_documents)
    ]
    return vs


@pytest.fixture()
def mock_rag_result() -> RAGResult:
    """Return a complete ``RAGResult`` for ``evaluate_rag.py`` tests.

    Uses realistic content for all fields so that RAGAS metric computations
    in tests produce meaningful (non-zero, non-trivial) scores.

    Returns:
        ``RAGResult`` with strategy ``"naive"``, one context string, and
        a ground truth that partially overlaps with the generated answer.
    """
    return RAGResult(
        question="What is RAG?",
        answer="RAG combines retrieval with generation to reduce hallucinations.",
        contexts=["RAG combines retrieval with generation."],
        ground_truth="RAG is a method that retrieves relevant documents before generating an answer.",
        strategy="naive",
        latency_ms=350.0,
    )


# ---------------------------------------------------------------------------
# Dataset file fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_jsonl(tmp_path: Path) -> Path:
    """Create a temporary JSONL file with 3 valid MMLU-format items.

    Each item has the full schema expected by ``load_dataset()`` in
    ``run_benchmark.py``: ``question``, ``choices`` (4 options), ``answer``
    (single letter), and ``subject``.

    Args:
        tmp_path: pytest built-in temporary directory fixture (function-scoped,
                  cleaned up automatically after the test).

    Returns:
        ``Path`` to the created ``test_dataset.jsonl`` file.
    """
    import json
    data = [
        {
            "question": "What is 2+2?",
            "choices":  ["3", "4", "5", "6"],
            "answer":   "B",
            "subject":  "math",
        },
        {
            "question": "Capital of France?",
            "choices":  ["London", "Paris", "Berlin", "Rome"],
            "answer":   "B",
            "subject":  "geography",
        },
        {
            "question": "What is H2O?",
            "choices":  ["Hydrogen", "Helium", "Water", "Oxygen"],
            "answer":   "C",
            "subject":  "chemistry",
        },
    ]
    path = tmp_path / "test_dataset.jsonl"
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    return path


@pytest.fixture()
def tmp_qa_jsonl(tmp_path: Path) -> Path:
    """Create a temporary JSONL file with 2 valid RAG QA pairs.

    Each item has ``question`` and ``ground_truth`` keys, matching the
    schema expected by ``_load_qa_pairs()`` in ``evaluate_rag.py``. Note
    that the key is ``"ground_truth"`` here, not ``"answer"`` — different
    from the MMLU schema used in ``tmp_jsonl``.

    Args:
        tmp_path: pytest built-in temporary directory fixture.

    Returns:
        ``Path`` to the created ``test_qa.jsonl`` file.
    """
    import json
    data = [
        {
            "question":     "What is RAG?",
            "ground_truth": "RAG combines retrieval with generation.",
        },
        {
            "question":     "What is faithfulness?",
            "ground_truth": "Faithfulness measures if claims are grounded in context.",
        },
    ]
    path = tmp_path / "test_qa.jsonl"
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    return path