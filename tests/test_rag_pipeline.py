"""
tests/test_rag_pipeline.py
===========================
Unit tests for RAG retrieval strategies and the ``RAGPipeline`` orchestrator.

Public APIs under test (from production files):

.. list-table::
   :header-rows: 1

   * - Class
     - Constructor
     - Methods tested
   * - ``NaiveRetriever``
     - ``(vectorstore, k)``
     - ``.retrieve()``, ``.retrieve_with_scores()``, ``.name``
   * - ``HyDERetriever``
     - ``(vectorstore, llm, k)``
     - ``.retrieve()`` with fallback, ``.retrieve_with_scores()`` with fallback, ``.name``
   * - ``RerankingRetriever``
     - ``(vectorstore, llm, k)``
     - ``.retrieve()`` with ``k×2`` pool, fallback on bad indices, ``.name``
   * - ``RAGPipeline``
     - ``(vectorstore, strategy, k)``
     - ``.query(question, ground_truth)`` → ``RAGResult``

Isolation strategy:
    ChromaDB, all LLM clients, and the LCEL chain are fully mocked.
    No disk I/O, no network calls, and no real model inference occur.

Patching strategy for ``TestRAGPipeline``:
    Three patches are applied together on every ``RAGPipeline`` test:

    1. ``rag_pipeline.ChatOpenAI``      — prevents real Ollama Cloud connections.
    2. ``rag_pipeline.OllamaCloudClient`` — prevents instantiation of the real
       client used by HyDE and Reranking strategies.
    3. ``rag_pipeline.NaiveRetriever``  — injects a ``MagicMock`` retriever so
       that ``.retrieve()`` returns a controlled ``Document`` list.

    After pipeline construction, ``pipeline._chain`` is replaced with a
    ``MagicMock`` whose ``.invoke()`` returns a deterministic answer string,
    bypassing the LCEL chain entirely.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from llm_eval.shared.types import RAGResult  # installed package

# sys.path registered by conftest.py
from naive     import NaiveRetriever       # type: ignore[import]
from hyde      import HyDERetriever        # type: ignore[import]
from reranking import RerankingRetriever   # type: ignore[import]


# ---------------------------------------------------------------------------
# Test helper
# ---------------------------------------------------------------------------


def _mock_llm_client(response_text: str = "A hypothetical answer.") -> MagicMock:
    """Build a mock LLM client whose ``complete()`` returns a response object.

    Mimics the ``_LLMClient`` protocol used by ``HyDERetriever`` and
    ``RerankingRetriever``: ``client.complete(...)`` returns an object with
    a ``response`` attribute, matching the ``getattr(resp, "response", resp)``
    normalisation in both retrievers.

    Args:
        response_text: Text assigned to ``mock_resp.response``.
                       Defaults to ``"A hypothetical answer."``.

    Returns:
        ``MagicMock`` with ``client.complete.return_value.response`` set to
        ``response_text``.
    """
    mock_resp          = MagicMock()
    mock_resp.response = response_text
    client             = MagicMock()
    client.complete.return_value = mock_resp
    return client


# ---------------------------------------------------------------------------
# NaiveRetriever tests
# ---------------------------------------------------------------------------


class TestNaiveRetriever:
    """Tests for :class:`NaiveRetriever`."""

    def test_retrieve_retorna_lista_de_documents(
        self, mock_vectorstore, mock_documents
    ):
        """``.retrieve()`` should return a list of ``Document`` objects.

        Verifies type correctness for all elements and that the count matches
        the mock vector store's configured return value.
        """
        retriever = NaiveRetriever(vectorstore=mock_vectorstore, k=3)
        results   = retriever.retrieve("What is RAG?")

        assert isinstance(results, list)
        assert len(results) == len(mock_documents)
        assert all(isinstance(d, Document) for d in results)

    def test_retrieve_chama_similarity_search_com_k(self, mock_vectorstore):
        """``.retrieve()`` should call ``vectorstore.similarity_search`` with the correct ``k``.

        Verifies that ``k`` is forwarded to the vector store rather than
        being silently overridden or ignored.
        """
        retriever = NaiveRetriever(vectorstore=mock_vectorstore, k=3)
        retriever.retrieve("Test query")

        mock_vectorstore.similarity_search.assert_called_once_with("Test query", k=3)

    def test_retrieve_with_scores_retorna_tuplas(self, mock_vectorstore):
        """``.retrieve_with_scores()`` should return a list of ``(Document, float)`` tuples."""
        retriever = NaiveRetriever(vectorstore=mock_vectorstore, k=3)
        results   = retriever.retrieve_with_scores("Test query")

        assert all(
            isinstance(doc, Document) and isinstance(score, float)
            for doc, score in results
        )

    def test_name_property(self, mock_vectorstore):
        """``.name`` should return ``"naive"``."""
        retriever = NaiveRetriever(vectorstore=mock_vectorstore, k=3)
        assert retriever.name == "naive"

    def test_k_personalizado(self, mock_vectorstore):
        """``NaiveRetriever`` should use the ``k`` value passed to the constructor."""
        retriever = NaiveRetriever(vectorstore=mock_vectorstore, k=7)
        retriever.retrieve("query")

        mock_vectorstore.similarity_search.assert_called_once_with("query", k=7)


# ---------------------------------------------------------------------------
# HyDERetriever tests
# ---------------------------------------------------------------------------


class TestHyDERetriever:
    """Tests for :class:`HyDERetriever`."""

    def test_retrieve_usa_resposta_hipotetica(self, mock_vectorstore):
        """HyDE should generate a hypothetical doc and use it as the vector store query.

        Verifies the core HyDE behaviour: the LLM is called once, and the
        text from ``mock_resp.response`` is passed to ``similarity_search``
        instead of the original question.
        """
        hyp_text  = "RAG is a method combining retrieval and generation."
        llm       = _mock_llm_client(hyp_text)
        retriever = HyDERetriever(vectorstore=mock_vectorstore, llm=llm, k=3)

        retriever.retrieve("What is RAG?")

        llm.complete.assert_called_once()
        called_query = mock_vectorstore.similarity_search.call_args[0][0]
        assert called_query == hyp_text

    def test_fallback_quando_llm_falha(self, mock_vectorstore):
        """When the LLM raises, ``.retrieve()`` should fall back to the raw query.

        Simulates a network or quota error from the LLM client and verifies
        that the retriever degrades gracefully to naive-retrieval quality
        rather than propagating the exception.
        """
        llm             = MagicMock()
        llm.complete.side_effect = RuntimeError("API error")
        retriever       = HyDERetriever(vectorstore=mock_vectorstore, llm=llm, k=3)

        results      = retriever.retrieve("What is faithfulness?")
        called_query = mock_vectorstore.similarity_search.call_args[0][0]

        assert isinstance(results, list)
        assert called_query == "What is faithfulness?"

    def test_retrieve_with_scores_fallback(self, mock_vectorstore):
        """``.retrieve_with_scores()`` should also fall back on LLM failure.

        Mirrors ``test_fallback_quando_llm_falha`` for the scored variant to
        ensure both retrieval methods apply the same fallback logic.
        """
        llm             = MagicMock()
        llm.complete.side_effect = RuntimeError("API error")
        retriever       = HyDERetriever(vectorstore=mock_vectorstore, llm=llm, k=3)

        results = retriever.retrieve_with_scores("What is HyDE?")
        assert isinstance(results, list)

    def test_name_property(self, mock_vectorstore):
        """``.name`` should return ``"hyde"``."""
        retriever = HyDERetriever(
            vectorstore=mock_vectorstore, llm=_mock_llm_client(), k=3
        )
        assert retriever.name == "hyde"


# ---------------------------------------------------------------------------
# RerankingRetriever tests
# ---------------------------------------------------------------------------


class TestRerankingRetriever:
    """Tests for :class:`RerankingRetriever`."""

    def test_busca_pool_maior_que_k(self, mock_vectorstore):
        """The initial similarity search should request more than ``k`` candidates.

        Verifies the ``k×2`` pool expansion. Uses ``.call_args.kwargs`` with a
        positional-arg fallback for robustness across Python and mock versions.
        """
        llm       = _mock_llm_client("0")
        k         = 2
        retriever = RerankingRetriever(vectorstore=mock_vectorstore, llm=llm, k=k)
        retriever.retrieve("What is reranking?")

        called_k = mock_vectorstore.similarity_search.call_args.kwargs.get(
            "k",
            mock_vectorstore.similarity_search.call_args[1].get("k"),
        )
        assert called_k > k, (
            f"Expected pool > {k}, but similarity_search received k={called_k}"
        )

    def test_retorna_no_maximo_k_documentos(self, mock_vectorstore):
        """``.retrieve()`` should return at most ``k`` documents after reranking."""
        llm       = _mock_llm_client("0")
        retriever = RerankingRetriever(vectorstore=mock_vectorstore, llm=llm, k=2)
        results   = retriever.retrieve("Test query")

        assert len(results) <= 2

    def test_score_invalido_nao_lanca_excecao(self, mock_vectorstore):
        """A non-numeric reranker response should fall back to naive top-k silently.

        Verifies that the ``isdigit()`` filter in the index parser catches
        non-numeric tokens and triggers the fallback path without raising.
        """
        llm       = _mock_llm_client("not a number")
        retriever = RerankingRetriever(vectorstore=mock_vectorstore, llm=llm, k=2)
        results   = retriever.retrieve("query")

        assert isinstance(results, list)

    def test_score_fora_de_range_e_clampado(self, mock_vectorstore):
        """An out-of-range index should trigger the fallback path without raising.

        Verifies the ``0 <= i < len(candidates)`` safety filter that prevents
        ``IndexError`` when the LLM returns an index beyond the candidate pool.
        """
        llm       = _mock_llm_client("999")
        retriever = RerankingRetriever(vectorstore=mock_vectorstore, llm=llm, k=2)
        results   = retriever.retrieve("query")

        assert isinstance(results, list)

    def test_name_property(self, mock_vectorstore):
        """``.name`` should return ``"reranking"``."""
        retriever = RerankingRetriever(
            vectorstore=mock_vectorstore, llm=_mock_llm_client(), k=3
        )
        assert retriever.name == "reranking"


# ---------------------------------------------------------------------------
# RAGPipeline tests
# ---------------------------------------------------------------------------


class TestRAGPipeline:
    """Tests for :class:`RAGPipeline`.

    Each test applies the same three patches (``ChatOpenAI``,
    ``OllamaCloudClient``, ``NaiveRetriever``) and replaces
    ``pipeline._chain`` after construction to bypass the LCEL chain.
    See module docstring for the full patching rationale.
    """

    def test_query_retorna_rag_result(self, mock_vectorstore):
        """``.query()`` should return a ``RAGResult`` with all fields populated."""
        with patch("rag_pipeline.ChatOpenAI")        as mock_llm_cls,  \
             patch("rag_pipeline.OllamaCloudClient") as mock_client_cls, \
             patch("rag_pipeline.NaiveRetriever")    as mock_retriever_cls:

            mock_llm_cls.return_value    = MagicMock()
            mock_client_cls.return_value = MagicMock()

            mock_retriever = MagicMock()
            mock_retriever.retrieve.return_value = [
                Document(page_content="RAG combines retrieval with generation.")
            ]
            mock_retriever_cls.return_value = mock_retriever

            mock_chain        = MagicMock()
            mock_chain.invoke.return_value = "RAG reduces hallucinations."

            from rag_pipeline import RAGPipeline  # type: ignore[import]
            pipeline         = RAGPipeline(vectorstore=mock_vectorstore, strategy="naive")
            pipeline._chain  = mock_chain

            result = pipeline.query(
                question="What is RAG?",
                ground_truth="RAG combines retrieval and generation.",
            )

        assert isinstance(result, RAGResult)
        assert result.question     == "What is RAG?"
        assert result.answer       == "RAG reduces hallucinations."
        assert result.strategy     == "naive"
        assert result.ground_truth == "RAG combines retrieval and generation."
        assert result.latency_ms   >= 0

    def test_contexts_e_lista_de_strings(self, mock_vectorstore):
        """``RAGResult.contexts`` should be a list of strings from retrieved ``Document`` objects."""
        with patch("rag_pipeline.ChatOpenAI")        as mock_llm_cls,  \
             patch("rag_pipeline.OllamaCloudClient") as mock_client_cls, \
             patch("rag_pipeline.NaiveRetriever")    as mock_retriever_cls:

            mock_llm_cls.return_value    = MagicMock()
            mock_client_cls.return_value = MagicMock()

            mock_retriever = MagicMock()
            mock_retriever.retrieve.return_value = [
                Document(page_content="Context 1."),
                Document(page_content="Context 2."),
            ]
            mock_retriever_cls.return_value = mock_retriever

            mock_chain        = MagicMock()
            mock_chain.invoke.return_value = "Answer."

            from rag_pipeline import RAGPipeline  # type: ignore[import]
            pipeline        = RAGPipeline(vectorstore=mock_vectorstore, strategy="naive")
            pipeline._chain = mock_chain

            result = pipeline.query("Test?")

        assert all(isinstance(c, str) for c in result.contexts)
        assert "Context 1." in result.contexts
        assert "Context 2." in result.contexts

    def test_ground_truth_padrao_vazio(self, mock_vectorstore):
        """``ground_truth`` should default to an empty string when not provided."""
        with patch("rag_pipeline.ChatOpenAI")        as mock_llm_cls,  \
             patch("rag_pipeline.OllamaCloudClient") as mock_client_cls, \
             patch("rag_pipeline.NaiveRetriever")    as mock_retriever_cls:

            mock_llm_cls.return_value    = MagicMock()
            mock_client_cls.return_value = MagicMock()

            mock_retriever = MagicMock()
            mock_retriever.retrieve.return_value = []
            mock_retriever_cls.return_value      = mock_retriever

            mock_chain        = MagicMock()
            mock_chain.invoke.return_value = "Answer."

            from rag_pipeline import RAGPipeline  # type: ignore[import]
            pipeline        = RAGPipeline(vectorstore=mock_vectorstore, strategy="naive")
            pipeline._chain = mock_chain

            result = pipeline.query("Question without ground truth?")

        assert result.ground_truth == ""


# ---------------------------------------------------------------------------
# evaluate_rag helpers tests
# ---------------------------------------------------------------------------


class TestEvaluateRAGHelpers:
    """Tests for helper functions in ``scripts/evaluate_rag.py``."""

    def test_load_qa_dataset_schema_valido(self, tmp_qa_jsonl):
        """``load_qa_dataset()`` should load items with ``question`` and ``ground_truth`` keys.

        Uses the ``tmp_qa_jsonl`` fixture (2 valid items) from ``conftest.py``.
        """
        from evaluate_rag import load_qa_dataset  # type: ignore[import]
        items = load_qa_dataset(tmp_qa_jsonl)

        assert len(items) == 2
        for item in items:
            assert "question"     in item
            assert "ground_truth" in item

    def test_load_qa_dataset_arquivo_invalido(self, tmp_path):
        """``load_qa_dataset()`` should raise ``ValueError`` when no valid QA pairs exist.

        Writes a JSONL file where every item is missing both ``question`` and
        ``ground_truth``, triggering the validation guard in the loader.
        """
        from evaluate_rag import load_qa_dataset  # type: ignore[import]
        bad_file = tmp_path / "bad.jsonl"
        bad_file.write_text('{"no_question": "x"}\n{"also_bad": "y"}\n')

        with pytest.raises(ValueError, match="Nenhum par Q&A válido"):
            load_qa_dataset(bad_file)

    def test_load_qa_dataset_arquivo_inexistente(self):
        """``load_qa_dataset()`` should raise ``FileNotFoundError`` for a non-existent path."""
        from evaluate_rag import load_qa_dataset  # type: ignore[import]
        with pytest.raises(FileNotFoundError):
            load_qa_dataset("/nao/existe/qa.jsonl")