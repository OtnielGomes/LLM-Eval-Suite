"""
tests/test_rag_pipeline.py

Tests for RAG strategies and the full pipeline.
Real API mapped from production files:

NaiveRetriever(vectorstore, k)       → .retrieve(query), .retrieve_with_scores(query), .name
HyDERetriever(vectorstore, llm, k)   → .retrieve(query) with fallback, .name
RerankingRetriever(vectorstore, llm, k) → .retrieve(query) with POOL_MULTIPLIER=2, .name
RAGPipeline(vectorstore, strategy, k)   → .query(question, ground_truth) → RAGResult

ChromaDB, LLMs, and the LCEL chain are 100% mocked — no real I/O.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from llm_eval.shared.types import RAGResult

# Imports resolved by conftest.py (sys.path configured)
from naive import NaiveRetriever        # type: ignore[import]
from hyde import HyDERetriever          # type: ignore[import]
from reranking import RerankingRetriever  # type: ignore[import]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_llm_client(response_text: str = "A hypothetical answer.") -> MagicMock:
    """Mocked LLM client with .complete() interface → object with .response."""
    mock_resp = MagicMock()
    mock_resp.response = response_text
    client = MagicMock()
    client.complete.return_value = mock_resp
    return client

# ---------------------------------------------------------------------------
# NaiveRetriever
# ---------------------------------------------------------------------------

class TestNaiveRetriever:

    def test_retrieve_returns_document_list(self, mock_vectorstore, mock_documents):
        """.retrieve() should return a list of Documents with len <= k."""
        retriever = NaiveRetriever(vectorstore=mock_vectorstore, k=3)
        results = retriever.retrieve("What is RAG?")
        assert isinstance(results, list)
        assert len(results) == len(mock_documents)
        assert all(isinstance(d, Document) for d in results)

    def test_retrieve_calls_similarity_search_with_k(self, mock_vectorstore):
        """.retrieve() should call vectorstore.similarity_search with the correct k."""
        retriever = NaiveRetriever(vectorstore=mock_vectorstore, k=3)
        retriever.retrieve("Test query")
        mock_vectorstore.similarity_search.assert_called_once_with("Test query", k=3)

    def test_retrieve_with_scores_returns_tuples(self, mock_vectorstore):
        """.retrieve_with_scores() should return a list of (Document, float) tuples."""
        retriever = NaiveRetriever(vectorstore=mock_vectorstore, k=3)
        results = retriever.retrieve_with_scores("Test query")
        assert all(isinstance(doc, Document) and isinstance(score, float)
                   for doc, score in results)

    def test_name_property(self, mock_vectorstore):
        """.name should return 'naive'."""
        retriever = NaiveRetriever(vectorstore=mock_vectorstore, k=3)
        assert retriever.name == "naive"

    def test_custom_k_is_used(self, mock_vectorstore):
        """NaiveRetriever should use the k value passed to the constructor."""
        retriever = NaiveRetriever(vectorstore=mock_vectorstore, k=7)
        retriever.retrieve("query")
        mock_vectorstore.similarity_search.assert_called_once_with("query", k=7)

# ---------------------------------------------------------------------------
# HyDERetriever
# ---------------------------------------------------------------------------

class TestHyDERetriever:

    def test_retrieve_uses_hypothetical_response(self, mock_vectorstore):
        """HyDE should generate a hypothetical text and use it as the vectorstore query."""
        llm = _mock_llm_client("RAG is a method combining retrieval and generation.")
        retriever = HyDERetriever(vectorstore=mock_vectorstore, llm=llm, k=3)

        retriever.retrieve("What is RAG?")

        llm.complete.assert_called_once()
        called_query = mock_vectorstore.similarity_search.call_args[0][0]
        assert called_query == "RAG is a method combining retrieval and generation."

    def test_fallback_when_llm_fails(self, mock_vectorstore):
        """If the LLM raises an exception, HyDE should fall back to the original query."""
        llm = MagicMock()
        llm.complete.side_effect = RuntimeError("API error")
        retriever = HyDERetriever(vectorstore=mock_vectorstore, llm=llm, k=3)

        results = retriever.retrieve("What is faithfulness?")

        assert isinstance(results, list)
        called_query = mock_vectorstore.similarity_search.call_args[0][0]
        assert called_query == "What is faithfulness?"

    def test_retrieve_with_scores_fallback(self, mock_vectorstore):
        """retrieve_with_scores() should also fall back gracefully on LLM error."""
        llm = MagicMock()
        llm.complete.side_effect = RuntimeError("API error")
        retriever = HyDERetriever(vectorstore=mock_vectorstore, llm=llm, k=3)

        results = retriever.retrieve_with_scores("What is HyDE?")
        assert isinstance(results, list)

    def test_name_property(self, mock_vectorstore):
        """.name should return 'hyde'."""
        retriever = HyDERetriever(
            vectorstore=mock_vectorstore, llm=_mock_llm_client(), k=3
        )
        assert retriever.name == "hyde"

# ---------------------------------------------------------------------------
# RerankingRetriever
# ---------------------------------------------------------------------------

class TestRerankingRetriever:

    def test_fetches_pool_larger_than_k(self, mock_vectorstore):
        """Reranking should fetch more candidates than k (expanded pool)."""
        llm = _mock_llm_client("0")
        k = 2
        retriever = RerankingRetriever(vectorstore=mock_vectorstore, llm=llm, k=k)
        retriever.retrieve("What is reranking?")

        # Accesses k via keyword arg — more robust than positional [1]
        called_k = mock_vectorstore.similarity_search.call_args.kwargs.get(
            "k", mock_vectorstore.similarity_search.call_args[1].get("k")
        )
        assert called_k > k, f"Expected pool > {k}, but similarity_search received k={called_k}"

    def test_returns_at_most_k_documents(self, mock_vectorstore, mock_documents):
        """retrieve() should return at most k documents after reranking."""
        llm = _mock_llm_client("0")
        retriever = RerankingRetriever(vectorstore=mock_vectorstore, llm=llm, k=2)
        results = retriever.retrieve("Test query")
        assert len(results) <= 2

    def test_invalid_score_does_not_raise(self, mock_vectorstore):
        """RerankingRetriever should handle a non-numeric score without raising."""
        llm = _mock_llm_client("not a number")
        retriever = RerankingRetriever(vectorstore=mock_vectorstore, llm=llm, k=2)
        results = retriever.retrieve("query")
        assert isinstance(results, list)

    def test_out_of_range_score_falls_back(self, mock_vectorstore):
        """A score outside valid index range should trigger the fallback without error."""
        llm = _mock_llm_client("999")
        retriever = RerankingRetriever(vectorstore=mock_vectorstore, llm=llm, k=2)
        results = retriever.retrieve("query")
        assert isinstance(results, list)

    def test_name_property(self, mock_vectorstore):
        """.name should return 'reranking'."""
        retriever = RerankingRetriever(
            vectorstore=mock_vectorstore, llm=_mock_llm_client(), k=3
        )
        assert retriever.name == "reranking"

# ---------------------------------------------------------------------------
# RAGPipeline
# ---------------------------------------------------------------------------

class TestRAGPipeline:

    def test_query_returns_rag_result(self, mock_vectorstore):
        """query() should return a RAGResult with all fields populated."""
        with patch("rag_pipeline.ChatOpenAI") as mock_llm_cls, \
             patch("rag_pipeline.OllamaCloudClient") as mock_client_cls, \
             patch("rag_pipeline.NaiveRetriever") as mock_retriever_cls:

            mock_llm_cls.return_value = MagicMock()
            mock_client_cls.return_value = MagicMock()

            mock_retriever = MagicMock()
            mock_retriever.retrieve.return_value = [
                Document(page_content="RAG combines retrieval with generation.")
            ]
            mock_retriever_cls.return_value = mock_retriever

            mock_chain = MagicMock()
            mock_chain.invoke.return_value = "RAG reduces hallucinations."

            from rag_pipeline import RAGPipeline  # type: ignore[import]
            pipeline = RAGPipeline(vectorstore=mock_vectorstore, strategy="naive")
            pipeline._chain = mock_chain

            result = pipeline.query(
                question="What is RAG?",
                ground_truth="RAG combines retrieval and generation."
            )

            assert isinstance(result, RAGResult)
            assert result.question == "What is RAG?"
            assert result.answer == "RAG reduces hallucinations."
            assert result.strategy == "naive"
            assert result.ground_truth == "RAG combines retrieval and generation."
            assert result.latency_ms >= 0

    def test_contexts_is_list_of_strings(self, mock_vectorstore):
        """RAGResult.contexts should be a list of strings extracted from Documents."""
        with patch("rag_pipeline.ChatOpenAI") as mock_llm_cls, \
             patch("rag_pipeline.OllamaCloudClient") as mock_client_cls, \
             patch("rag_pipeline.NaiveRetriever") as mock_retriever_cls:

            mock_llm_cls.return_value = MagicMock()
            mock_client_cls.return_value = MagicMock()

            mock_retriever = MagicMock()
            mock_retriever.retrieve.return_value = [
                Document(page_content="Context 1."),
                Document(page_content="Context 2."),
            ]
            mock_retriever_cls.return_value = mock_retriever

            mock_chain = MagicMock()
            mock_chain.invoke.return_value = "Answer."

            from rag_pipeline import RAGPipeline  # type: ignore[import]
            pipeline = RAGPipeline(vectorstore=mock_vectorstore, strategy="naive")
            pipeline._chain = mock_chain

            result = pipeline.query("Test?")

            assert all(isinstance(c, str) for c in result.contexts)
            assert "Context 1." in result.contexts
            assert "Context 2." in result.contexts

    def test_ground_truth_defaults_to_empty_string(self, mock_vectorstore):
        """ground_truth should default to an empty string when not provided."""
        with patch("rag_pipeline.ChatOpenAI") as mock_llm_cls, \
             patch("rag_pipeline.OllamaCloudClient") as mock_client_cls, \
             patch("rag_pipeline.NaiveRetriever") as mock_retriever_cls:

            mock_llm_cls.return_value = MagicMock()
            mock_client_cls.return_value = MagicMock()

            mock_retriever = MagicMock()
            mock_retriever.retrieve.return_value = []
            mock_retriever_cls.return_value = mock_retriever

            mock_chain = MagicMock()
            mock_chain.invoke.return_value = "Answer."

            from rag_pipeline import RAGPipeline  # type: ignore[import]
            pipeline = RAGPipeline(vectorstore=mock_vectorstore, strategy="naive")
            pipeline._chain = mock_chain

            result = pipeline.query("Question without ground truth?")

            assert result.ground_truth == ""

# ---------------------------------------------------------------------------
# evaluate_rag helpers
# ---------------------------------------------------------------------------

class TestEvaluateRAGHelpers:

    def test_load_qa_dataset_valid_schema(self, tmp_qa_jsonl):
        """load_qa_dataset() should load Q&A pairs with question and ground_truth fields."""
        from evaluate_rag import load_qa_dataset  # type: ignore[import]
        items = load_qa_dataset(tmp_qa_jsonl)

        assert len(items) == 2
        for item in items:
            assert "question" in item
            assert "ground_truth" in item

    def test_load_qa_dataset_invalid_file(self, tmp_path):
        """load_qa_dataset() should raise ValueError for a file with no valid items."""
        from evaluate_rag import load_qa_dataset  # type: ignore[import]
        bad_file = tmp_path / "bad.jsonl"
        bad_file.write_text('{"no_question": "x"}\n{"also_bad": "y"}\n')

        with pytest.raises(ValueError, match="No valid Q&A pair found"):
            load_qa_dataset(bad_file)

    def test_load_qa_dataset_file_not_found(self):
        """load_qa_dataset() should raise FileNotFoundError for a non-existent file."""
        from evaluate_rag import load_qa_dataset  # type: ignore[import]
        with pytest.raises(FileNotFoundError):
            load_qa_dataset("/does/not/exist/qa.jsonl")