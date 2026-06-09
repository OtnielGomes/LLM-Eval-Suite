"""
src/llm_eval/rag/strategies/naive.py
=====================================
Naive dense retrieval strategy — the RAG baseline.

Embeds the raw query using the vector store's configured embedding model and
performs cosine similarity search directly against the indexed document chunks.
No preprocessing, query rewriting, or reranking is applied.

Pipeline:
    query → embed → cosine similarity search → top-k documents

Serves as the performance baseline against which ``HyDERetriever`` and
``RerankingRetriever`` are compared in ``evaluate_rag.py``. Any accuracy
improvement from the more complex strategies is measured relative to this
implementation.

Trade-offs:

.. list-table::
   :header-rows: 1

   * - Aspect
     - NaiveRetriever
   * - Latency
     - Lowest (no extra LLM call)
   * - Cost
     - Lowest (embed only)
   * - Recall on paraphrased queries
     - Lower than HyDE
   * - Context precision
     - Lower than reranking
   * - Failure modes
     - None beyond vector store availability

Configuration:
    ``TOP_K_RETRIEVAL`` from ``shared/config.py`` controls the default number
    of documents returned per query.
"""
from __future__ import annotations

from langchain_core.documents import Document
from langchain_chroma import Chroma

from llm_eval.shared.config import TOP_K_RETRIEVAL


class NaiveRetriever:
    """Baseline retriever using direct cosine similarity search.

    Embeds the query with the same model used during ingestion and returns
    the ``k`` most similar document chunks from the vector store. No query
    transformation or result reranking is performed.

    Because it requires no additional LLM calls, ``NaiveRetriever`` has the
    lowest latency and cost of the three strategies, making it the natural
    starting point before investing in more complex approaches.

    Attributes:
        _vs (Chroma): Vector store used for similarity search.
        _k (int):     Number of documents to retrieve per query.

    Example:
        >>> retriever = NaiveRetriever(vectorstore=vs)
        >>> docs = retriever.retrieve("What is retrieval-augmented generation?")
        >>> len(docs) <= 5  # bounded by TOP_K_RETRIEVAL
        True
    """

    def __init__(self, vectorstore: Chroma, k: int = TOP_K_RETRIEVAL):
        """Initialise the naive retriever.

        Args:
            vectorstore: Chroma vector store to search against. Must be
                         populated via ``ingest.py`` before retrieval.
            k:           Number of documents to return per query. Defaults to
                         ``TOP_K_RETRIEVAL`` from ``config.py``.
        """
        self._vs = vectorstore
        self._k  = k

    def retrieve(self, query: str) -> list[Document]:
        """Retrieve the top-k most similar documents for the given query.

        Embeds ``query`` using the vector store's configured embedding model
        and returns the ``k`` nearest neighbours by cosine similarity.

        Args:
            query: User question or search string.

        Returns:
            List of up to ``k`` ``Document`` objects ranked by descending
            cosine similarity to the query embedding.
        """
        return self._vs.similarity_search(query, k=self._k)

    def retrieve_with_scores(self, query: str) -> list[tuple[Document, float]]:
        """Retrieve the top-k documents with their cosine similarity scores.

        Identical to :meth:`retrieve` but includes the relevance score for
        each document. Useful for score-based context filtering, reranking
        pipelines, and RAGAS ``context_precision`` / ``context_recall``
        evaluation.

        Args:
            query: User question or search string.

        Returns:
            List of up to ``k`` ``(Document, float)`` tuples ranked by
            descending similarity score. Scores are in ``[0.0, 1.0]`` for
            normalised cosine similarity (exact range depends on the
            ChromaDB distance metric configured at collection creation).
        """
        return self._vs.similarity_search_with_relevance_scores(query, k=self._k)

    @property
    def name(self) -> str:
        """Strategy identifier used in result filenames and RAGAS reports.

        Returns:
            ``"naive"``
        """
        return "naive"