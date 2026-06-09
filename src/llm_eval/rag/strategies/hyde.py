"""
src/llm_eval/rag/strategies/hyde.py
====================================
HyDE (Hypothetical Document Embeddings) retrieval strategy.

Instead of embedding the raw query and searching directly, HyDE first prompts
an LLM to generate a *hypothetical* answer document, then embeds that
synthetic document and uses it as the search pivot. The intuition is that
a generated answer is linguistically closer to the indexed documents than
the original question, improving recall on queries phrased differently from
the corpus.

Pipeline:
    query → LLM → hypothetical_doc → embed → similarity_search → documents

Compared to naive dense retrieval:

.. list-table::
   :header-rows: 1

   * - Aspect
     - NaiveRetriever
     - HyDERetriever
   * - Search pivot
     - Raw query embedding
     - Hypothetical answer embedding
   * - Extra LLM call
     - No
     - Yes (+latency, +cost)
   * - Recall on paraphrased queries
     - Lower
     - Higher
   * - Risk
     - None
     - Hallucinated pivot may retrieve irrelevant docs

Failure mode and mitigation:
    If the LLM call to generate the hypothetical document fails (network
    error, quota exhaustion, ``None`` response), :meth:`retrieve` and
    :meth:`retrieve_with_scores` silently fall back to the raw query as the
    search pivot — preserving naive-retrieval quality rather than raising.

Reference:
    Gao et al., 2022 — "Precise Zero-Shot Dense Retrieval without Relevance Labels"
    https://arxiv.org/abs/2212.10496
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
    """Minimal structural interface for the LLM used in hypothetical doc generation.

    Any client with a ``complete()`` method satisfies this protocol.
    All three clients in the suite (``OllamaClient``, ``OllamaCloudClient``,
    ``GeminiClient``) conform to it, keeping ``HyDERetriever`` decoupled
    from any specific client implementation.

    The leading underscore marks this as a module-private protocol — it is
    not exported via ``__init__.py`` and is not part of the public API.
    """

    def complete(self, prompt: str, system: str = "") -> object:
        """Generate a text response for the given prompt.

        Args:
            prompt: Input text sent to the model.
            system: Optional system prompt.

        Returns:
            Any object with a ``response`` attribute (``LLMResponse``) or a
            string. ``_generate_hypothetical_doc`` handles both via ``getattr``.
        """
        ...


# ---------------------------------------------------------------------------
# HyDE retriever
# ---------------------------------------------------------------------------


class HyDERetriever:
    """Retrieval strategy using a hypothetical document as the semantic pivot.

    Generates a synthetic answer to the query via an LLM call, then embeds
    that answer and uses it for similarity search instead of the raw query.
    This bridges the lexical gap between question-style queries and
    answer-style indexed documents.

    Attributes:
        SYSTEM_PROMPT (str): System instruction that constrains the LLM to
                             produce a concise, factual answer paragraph
                             without preamble or hedging.
        _vs (Chroma):        Vector store used for similarity search.
        _llm (_LLMClient):   LLM client used to generate the hypothetical document.
        _k (int):            Number of documents to retrieve per query.

    Example:
        >>> from llm_eval.clients.ollama_cloud_cliente import OllamaCloudClient
        >>> retriever = HyDERetriever(vectorstore=vs, llm=OllamaCloudClient())
        >>> docs = retriever.retrieve("What is retrieval-augmented generation?")
        >>> len(docs) <= 5  # bounded by TOP_K_RETRIEVAL
        True
    """

    # System instruction for the hypothetical document generation step.
    # "Write only the answer — no preamble" is critical: preamble text
    # (e.g. "Sure! Here is an answer...") degrades embedding quality by
    # shifting the vector away from the target semantic space.
    SYSTEM_PROMPT = (
        "You are a knowledgeable assistant. "
        "Given a question, write a concise, factual paragraph that would "
        "directly answer it. Write only the answer — no preamble."
    )

    def __init__(self, vectorstore: Chroma, llm: _LLMClient, k: int = TOP_K_RETRIEVAL):
        """Initialise the HyDE retriever.

        Args:
            vectorstore: Chroma vector store to search against. Must be
                         populated via ``ingest.py`` before retrieval.
            llm:         LLM client used to generate the hypothetical answer
                         document. Any object satisfying ``_LLMClient`` works.
            k:           Number of documents to return per query. Defaults to
                         ``TOP_K_RETRIEVAL`` from ``config.py``.
        """
        self._vs  = vectorstore
        self._llm = llm
        self._k   = k

    # ---- Hypothetical document generation ------------------------------- #

    def _generate_hypothetical_doc(self, query: str) -> str:
        """Generate a hypothetical answer document for the given query.

        Calls the LLM with ``SYSTEM_PROMPT`` to produce a short factual
        paragraph that would plausibly answer the query. The response object
        is normalised to a string via ``getattr(resp, "response", resp)`` to
        handle both ``LLMResponse`` dataclass instances and raw strings.

        Args:
            query: The user question to generate a hypothetical answer for.

        Returns:
            Hypothetical answer string to use as the embedding search pivot.

        Note:
            This method does not handle exceptions — callers (``retrieve`` and
            ``retrieve_with_scores``) are responsible for catching errors and
            falling back to the raw query.
        """
        resp = self._llm.complete(
            prompt=f"Question: {query}",
            system=self.SYSTEM_PROMPT,
        )
        # Normalise LLMResponse dataclass and raw-string returns to str
        return str(getattr(resp, "response", resp))

    # ---- Retrieval ------------------------------------------------------ #

    def retrieve(self, query: str) -> list[Document]:
        """Retrieve the top-k most relevant documents using HyDE.

        Generates a hypothetical document, embeds it, and performs cosine
        similarity search against the vector store. Falls back to embedding
        the raw query if hypothetical document generation fails.

        Args:
            query: User question string.

        Returns:
            List of up to ``k`` ``Document`` objects ranked by similarity to
            the hypothetical document (or raw query on fallback).
        """
        try:
            hyp_doc = self._generate_hypothetical_doc(query)
        except Exception:
            # Fallback to raw query preserves naive-retrieval quality
            # rather than raising and aborting the RAG pipeline
            hyp_doc = query
        return self._vs.similarity_search(hyp_doc, k=self._k)

    def retrieve_with_scores(self, query: str) -> list[tuple[Document, float]]:
        """Retrieve the top-k documents with their relevance scores using HyDE.

        Identical to :meth:`retrieve` but returns cosine similarity scores
        alongside each document. Useful for reranking, score-based filtering,
        and RAGAS ``context_precision`` / ``context_recall`` evaluation.

        Args:
            query: User question string.

        Returns:
            List of up to ``k`` ``(Document, float)`` tuples ranked by
            similarity score (higher is more relevant). Falls back to raw
            query embedding on hypothetical document generation failure.
        """
        try:
            hyp_doc = self._generate_hypothetical_doc(query)
        except Exception:
            hyp_doc = query
        return self._vs.similarity_search_with_relevance_scores(hyp_doc, k=self._k)

    # ---- Metadata ------------------------------------------------------- #

    @property
    def name(self) -> str:
        """Strategy identifier used in result filenames and RAGAS reports.

        Returns:
            ``"hyde"``
        """
        return "hyde"