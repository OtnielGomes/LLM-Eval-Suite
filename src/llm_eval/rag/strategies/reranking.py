"""
src/llm_eval/rag/strategies/reranking.py
=========================================
LLM-based reranking retrieval strategy.

Implements a retrieve-then-rerank pipeline: fetches ``k * 2`` candidate
documents via cosine similarity, then asks an LLM to select the ``k`` most
relevant passages from the enlarged candidate set. The extra LLM call acts
as a cross-attention reranker, filtering out topically adjacent but
semantically irrelevant chunks that dense retrieval alone cannot distinguish.

Pipeline:
    query → embed → similarity_search(k×2) → LLM reranker → top-k documents

Compared to naive dense retrieval:

.. list-table::
   :header-rows: 1

   * - Aspect
     - NaiveRetriever
     - RerankingRetriever
   * - Candidate pool
     - k
     - k × 2
   * - Selection method
     - Cosine similarity only
     - LLM relevance judgement
   * - Extra LLM call
     - No
     - Yes (+latency, +cost)
   * - Context precision
     - Baseline
     - Higher
   * - Risk
     - None
     - LLM may return malformed indices → fallback to naive top-k

Failure mode and mitigation:
    If the LLM call fails or returns unparseable output, ``retrieve`` silently
    falls back to the top-``k`` cosine-ranked candidates, preserving
    naive-retrieval quality rather than raising.

Limitation — ``retrieve_with_scores``:
    After reranking, the original cosine similarity scores no longer reflect
    the final document ordering. ``retrieve_with_scores`` therefore returns
    a placeholder score of ``1.0`` for all reranked documents. RAGAS metrics
    that use scores (e.g. ``context_precision``) will treat all retrieved
    contexts as equally relevant.
"""
from __future__ import annotations

from typing import Protocol

from langchain_core.documents import Document
from langchain_chroma import Chroma

from llm_eval.shared.config import TOP_K_RETRIEVAL


# ---------------------------------------------------------------------------
# LLM client protocol
# ---------------------------------------------------------------------------


class _LLMClient(Protocol):
    """Minimal structural interface for the LLM used as the reranker.

    Any client with a ``complete()`` method satisfies this protocol. All
    three clients in the suite (``OllamaClient``, ``OllamaCloudClient``,
    ``GeminiClient``) conform to it, keeping ``RerankingRetriever`` decoupled
    from any specific implementation.

    The leading underscore marks this as module-private — not exported via
    ``__init__.py`` and not part of the public API.
    """

    def complete(self, prompt: str, system: str = "") -> object:
        """Generate a text response for the given prompt.

        Args:
            prompt: Input text sent to the model.
            system: Optional system prompt.

        Returns:
            Any object with a ``response`` attribute (``LLMResponse``) or a
            raw string. ``retrieve`` handles both via ``getattr``.
        """
        ...


# ---------------------------------------------------------------------------
# Reranking retriever
# ---------------------------------------------------------------------------


class RerankingRetriever:
    """Retrieval strategy using an LLM reranker to refine a larger candidate set.

    Fetches ``k * 2`` candidates via cosine similarity to widen the initial
    recall window, then prompts the LLM to select the ``k`` most relevant
    passages by index. This two-stage approach improves context precision by
    delegating the final selection to a model that can reason about semantic
    relevance rather than vector distance alone.

    Attributes:
        SYSTEM_PROMPT (str): Instruction that constrains the LLM to return
                             only a comma-separated list of passage indices.
        _vs (Chroma):        Vector store used for the initial candidate retrieval.
        _llm (_LLMClient):   LLM client used as the relevance reranker.
        _k (int):            Number of documents to return after reranking.

    Example:
        >>> from llm_eval.clients.ollama_cloud_cliente import OllamaCloudClient
        >>> retriever = RerankingRetriever(vectorstore=vs, llm=OllamaCloudClient())
        >>> docs = retriever.retrieve("What is context precision in RAG?")
        >>> len(docs) <= 5  # bounded by TOP_K_RETRIEVAL
        True
    """

    # "Return ONLY a comma-separated list of indices" is the critical constraint:
    # any preamble or explanation in the response breaks the index parser and
    # triggers the fallback to naive top-k candidates.
    SYSTEM_PROMPT = (
        "You are a relevance judge. Given a question and a list of passages, "
        "select the indices of the most relevant passages. "
        "Return ONLY a comma-separated list of indices (e.g. 0,2,4)."
    )

    def __init__(self, vectorstore: Chroma, llm: _LLMClient, k: int = TOP_K_RETRIEVAL):
        """Initialise the reranking retriever.

        Args:
            vectorstore: Chroma vector store for the initial similarity search.
                         Must be populated via ``ingest.py`` before retrieval.
            llm:         LLM client used to rerank the candidate passages.
                         Any object satisfying ``_LLMClient`` works.
            k:           Number of documents to return after reranking.
                         The initial candidate pool is ``k * 2``. Defaults
                         to ``TOP_K_RETRIEVAL`` from ``config.py``.
        """
        self._vs  = vectorstore
        self._llm = llm
        self._k   = k

    def retrieve(self, query: str) -> list[Document]:
        """Retrieve and rerank the top-k most relevant documents for the query.

        Two-stage process:

        1. **Retrieve** ``k * 2`` candidates from the vector store via cosine
           similarity to widen the recall window beyond what naive retrieval
           would return.
        2. **Rerank** by prompting the LLM with the question and truncated
           passage text (first 300 characters each). The LLM returns a
           comma-separated list of indices identifying the most relevant
           passages, which are mapped back to the original candidate list.

        Index parsing applies two safety filters:
        - Non-digit tokens in the LLM response are silently discarded.
        - Out-of-range indices (``< 0`` or ``>= len(candidates)``) are dropped
          to prevent ``IndexError``.

        Args:
            query: User question string.

        Returns:
            List of up to ``k`` reranked ``Document`` objects. Falls back to
            the naive cosine-ranked top-``k`` candidates if:

            - The vector store returns no results.
            - The LLM call raises an exception.
            - The LLM response contains no valid parseable indices.

        Note:
            Passage text is truncated to 300 characters in the reranking prompt
            to keep the LLM context window manageable when ``k * 2`` is large.
            Full document content is returned in the final result list.
        """
        candidates = self._vs.similarity_search(query, k=self._k * 2)
        if not candidates:
            return []
        try:
            # Truncate each passage to 300 chars to keep the reranking prompt
            # within a manageable context window for smaller local models
            passages = "\n".join(
                f"[{i}] {doc.page_content[:300]}" for i, doc in enumerate(candidates)
            )
            prompt = f"Question: {query}\n\nPassages:\n{passages}"
            resp   = self._llm.complete(prompt=prompt, system=self.SYSTEM_PROMPT)
            # Normalise LLMResponse dataclass and raw-string returns to str
            raw     = str(getattr(resp, "response", resp))
            indices = [int(x.strip()) for x in raw.split(",") if x.strip().isdigit()]
            # Drop out-of-range indices to prevent IndexError on malformed output
            indices = [i for i in indices if 0 <= i < len(candidates)]
            if indices:
                return [candidates[i] for i in indices[: self._k]]
        except Exception:
            # Silent fallback: preserve naive top-k quality rather than
            # aborting the RAG pipeline on a reranker failure
            pass
        return candidates[: self._k]

    def retrieve_with_scores(self, query: str) -> list[tuple[Document, float]]:
        """Retrieve reranked documents with placeholder relevance scores.

        Delegates to :meth:`retrieve` for the actual retrieval and reranking,
        then attaches a uniform score of ``1.0`` to each returned document.

        The placeholder score reflects the fact that cosine similarity scores
        from the initial retrieval stage no longer correspond to the final
        reranked ordering — assigning them would be misleading. RAGAS metrics
        that consume scores (e.g. ``context_precision``) will treat all
        reranked contexts as equally relevant.

        Args:
            query: User question string.

        Returns:
            List of up to ``k`` ``(Document, 1.0)`` tuples in reranked order.
        """
        docs = self.retrieve(query)
        # Placeholder score 1.0: cosine scores from the candidate stage do not
        # reflect the reranked ordering and would distort score-based metrics
        return [(doc, 1.0) for doc in docs]

    @property
    def name(self) -> str:
        """Strategy identifier used in result filenames and RAGAS reports.

        Returns:
            ``"reranking"``
        """
        return "reranking"