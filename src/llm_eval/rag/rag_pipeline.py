"""
src/llm_eval/rag/rag_pipeline.py
=================================
RAG pipeline orchestrator with integrated LangSmith tracing.

Combines retrieval, prompt construction, and LLM generation into a single
callable using LangChain LCEL (LangChain Expression Language). Supports three
interchangeable retrieval strategies and emits LangSmith traces automatically
when the environment is configured.

Usage:
    from llm_eval.rag.rag_pipeline import RAGPipeline
    from scripts.ingest import get_vectorstore

    vs       = get_vectorstore()
    pipeline = RAGPipeline(vectorstore=vs, strategy="naive")
    result   = pipeline.query("What is RAG?", ground_truth="...")
    print(result.answer)

Available strategies:
    - ``"naive"``     — direct cosine similarity search (baseline)
    - ``"hyde"``      — hypothetical document embedding pivot
    - ``"reranking"`` — retrieve k×2 candidates, then LLM-rerank to k

LangSmith tracing:
    Set ``LANGSMITH_TRACING=true`` and ``LANGSMITH_API_KEY`` in ``.env``.
    No code changes required — ``@traceable`` activates automatically.

Environment variables:
    OLLAMA_API_KEY      : Ollama Cloud API key (required).
                          Generate at https://ollama.com/settings/keys
    OLLAMA_CLOUD_MODEL  : Cloud model override (optional, e.g. ``"llama3.3:70b"``).

Architecture — LCEL chain:
    question
      → retriever.retrieve(question) → _format_docs → {context}
      → RAG_PROMPT (context + question)
      → ChatOpenAI (Ollama Cloud)
      → StrOutputParser
      → str answer

Note on lazy API key reads:
    Both ``_build_langchain_llm`` and ``RAGPipeline.__init__`` read
    ``OLLAMA_API_KEY`` via ``os.getenv()`` at call/instantiation time, not at
    module import time. This ensures ``pytest`` fixtures using
    ``monkeypatch.setenv`` or ``patch.dict("os.environ")`` inject the key
    before it is consumed, without requiring module reloads.

Dependencies:
    langchain-core, langchain-openai, langchain-chroma, langsmith,
    python-dotenv, pydantic
"""
from __future__ import annotations

import os
import time
from typing import Literal

from dotenv import load_dotenv

load_dotenv()

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langsmith import traceable  # type: ignore[import-untyped]
from pydantic import SecretStr

from llm_eval.shared.types import RAGResult
from llm_eval.shared.config import (
    TOP_K_RETRIEVAL,
    OLLAMA_CLOUD_BASE_URL,
    OLLAMA_CLOUD_MODEL,
)
from llm_eval.rag.strategies.naive import NaiveRetriever
from llm_eval.rag.strategies.hyde import HyDERetriever
from llm_eval.rag.strategies.reranking import RerankingRetriever
from llm_eval.clients.ollama_cloud_cliente import OllamaCloudClient


# ---------------------------------------------------------------------------
# RAG prompt template
# ---------------------------------------------------------------------------

# Strict grounding instruction ("ONLY the provided context") reduces
# hallucination by preventing the model from supplementing retrieved
# passages with parametric knowledge.
RAG_PROMPT = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer the question using ONLY the provided context.
If the context doesn't contain enough information, say "I don't have enough context to answer."

Context:
{context}

Question: {question}

Answer:""")


def _format_docs(docs: list[Document]) -> str:
    """Concatenate document page contents into a single context string.

    Joins each document's ``page_content`` with a ``"\\n\\n---\\n\\n"``
    separator to visually delimit chunk boundaries in the prompt. This helps
    the LLM distinguish between separate source passages rather than treating
    the entire context as a single continuous text.

    Args:
        docs: List of ``Document`` objects returned by the retriever.

    Returns:
        Single string with all page contents separated by ``"---"`` dividers.
        Returns an empty string when ``docs`` is empty.
    """
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


# ---------------------------------------------------------------------------
# LangChain LLM factory
# ---------------------------------------------------------------------------


def _build_langchain_llm(temperature: float = 0.1) -> ChatOpenAI:
    """Instantiate a ``ChatOpenAI`` client pointed at Ollama Cloud.

    Used exclusively inside the LCEL generation chain. Reads ``OLLAMA_API_KEY``
    lazily via ``os.getenv()`` at call time — not at module import time — so
    that ``pytest`` fixtures can inject the key via ``monkeypatch.setenv``
    before the client is created.

    Args:
        temperature: Sampling temperature forwarded to the Ollama Cloud model.
                     Defaults to ``0.1`` for near-deterministic RAG answers.

    Returns:
        ``ChatOpenAI`` instance configured for Ollama Cloud with the model
        resolved from ``OLLAMA_CLOUD_MODEL`` env var or ``config.py`` fallback.

    Note:
        ``api_key`` is wrapped in ``SecretStr`` to prevent accidental logging
        of the key in LangChain traces and stack traces.
    """
    return ChatOpenAI(
        base_url=OLLAMA_CLOUD_BASE_URL,
        # SecretStr prevents the key from appearing in LangChain trace logs
        api_key=SecretStr(os.getenv("OLLAMA_API_KEY") or ""),
        model=os.getenv("OLLAMA_CLOUD_MODEL", OLLAMA_CLOUD_MODEL),
        temperature=temperature,
    )


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

#: Valid strategy names accepted by ``RAGPipeline``. Used as a ``Literal``
#: type to enable static exhaustiveness checking in ``_build_retriever``.
StrategyName = Literal["naive", "hyde", "reranking"]


# ---------------------------------------------------------------------------
# RAG pipeline
# ---------------------------------------------------------------------------


class RAGPipeline:
    """RAG pipeline with automatic LangSmith tracing.

    Orchestrates the full retrieve → prompt → generate flow for a single
    question. The retrieval strategy is selected at construction time and
    held for the lifetime of the pipeline instance.

    LangSmith tracing is activated automatically when
    ``LANGSMITH_TRACING=true`` is set in ``.env`` — no code changes are
    required. Each :meth:`query` call is recorded as a ``"rag_query"`` run
    with inputs, outputs, and latency.

    Attributes:
        _llm (ChatOpenAI):                            LangChain LLM used in the LCEL chain.
        _strategy_name (StrategyName):                Active strategy identifier.
        _retriever (NaiveRetriever
                    | HyDERetriever
                    | RerankingRetriever):             Retriever instance for the active strategy.
        _chain:                                       Compiled LCEL chain
                                                      (retriever → prompt → LLM → parser).

    Example:
        >>> from scripts.ingest import get_vectorstore
        >>> vs       = get_vectorstore()
        >>> pipeline = RAGPipeline(vectorstore=vs, strategy="hyde")
        >>> result   = pipeline.query("What is context recall?")
        >>> isinstance(result.answer, str)
        True
    """

    def __init__(
        self,
        vectorstore: Chroma,
        strategy: StrategyName = "naive",
        k: int = TOP_K_RETRIEVAL,
    ):
        """Initialise the RAG pipeline and validate the environment.

        Reads ``OLLAMA_API_KEY`` at instantiation time (lazy evaluation) so
        that test fixtures can inject the key before the pipeline is created.
        Raises immediately if the key is absent, providing an actionable error
        message rather than failing silently on the first query.

        Args:
            vectorstore: Populated Chroma instance from ``ingest.get_vectorstore()``.
            strategy:    Retrieval strategy to use. One of ``"naive"``,
                         ``"hyde"``, or ``"reranking"``. Defaults to ``"naive"``.
            k:           Number of document chunks to retrieve per query.
                         Defaults to ``TOP_K_RETRIEVAL`` from ``config.py``.

        Raises:
            EnvironmentError: If ``OLLAMA_API_KEY`` is not set in the environment.
        """
        # Lazy read — must occur after load_dotenv() and after pytest fixtures
        # have had the opportunity to inject OLLAMA_API_KEY via monkeypatch
        api_key = os.getenv("OLLAMA_API_KEY", "")
        if not api_key:
            raise EnvironmentError(
                "OLLAMA_API_KEY not found in .env.\n"
                "Create one at https://ollama.com/settings/keys"
            )

        self._llm            = _build_langchain_llm(temperature=0.1)
        self._strategy_name  = strategy
        self._retriever      = self._build_retriever(vectorstore, strategy, k)
        self._chain          = self._build_chain()

    def _build_retriever(
        self,
        vectorstore: Chroma,
        strategy: StrategyName,
        k: int,
    ) -> NaiveRetriever | HyDERetriever | RerankingRetriever:
        """Instantiate the retriever for the selected strategy.

        ``HyDERetriever`` and ``RerankingRetriever`` both require an LLM client
        for their respective generation steps. Both use ``OllamaCloudClient``
        directly — this eliminates the need for a redundant minimal client
        wrapper that was previously duplicated in this file.

        Args:
            vectorstore: Chroma vector store passed through to the retriever.
            strategy:    Strategy key selecting which retriever to instantiate.
            k:           Number of documents to retrieve.

        Returns:
            Configured retriever instance for the selected strategy.
        """
        if strategy == "naive":
            return NaiveRetriever(vectorstore, k=k)

        # HyDE and Reranking share the same LLM client — instantiated once
        # here and passed to whichever strategy is selected
        client = OllamaCloudClient()

        if strategy == "hyde":
            return HyDERetriever(vectorstore, llm=client, k=k)
        return RerankingRetriever(vectorstore, llm=client, k=k)

    def _build_chain(self):  # type: ignore[return]
        """Compile the LCEL generation chain.

        Assembles the retrieve → format → prompt → LLM → parse pipeline
        using LangChain Expression Language. The chain is built once at
        construction time and reused across all :meth:`query` calls.

        LCEL chain structure::

            {
                "context":  lambda q → retriever.retrieve(q) → _format_docs,
                "question": RunnablePassthrough(),
            }
            | RAG_PROMPT
            | ChatOpenAI (Ollama Cloud)
            | StrOutputParser

        Returns:
            Compiled LCEL runnable that accepts a question string and returns
            the generated answer string.

        Note:
            ``retriever.retrieve`` is captured by reference so that swapping
            ``self._retriever`` after construction (e.g. in tests) does not
            affect the compiled chain. The chain always calls the retriever
            that was active when ``_build_chain`` was called.
        """
        retriever = self._retriever.retrieve

        return (
            {
                "context":  lambda q: _format_docs(retriever(q)),
                "question": RunnablePassthrough(),
            }
            | RAG_PROMPT
            | self._llm
            | StrOutputParser()
        )

    @traceable(name="rag_query")  # type: ignore[misc]
    def query(
        self,
        question: str,
        ground_truth: str = "",
    ) -> RAGResult:
        """Execute a question through the RAG pipeline and return a structured result.

        Calls the retriever twice: once to collect context strings for the
        ``RAGResult`` (used by RAGAS evaluation) and once implicitly inside
        the LCEL chain for answer generation. Both calls hit the same
        retriever, so the contexts in the result always correspond to the
        contexts used during generation.

        The ``@traceable(name="rag_query")`` decorator records this call as a
        LangSmith run when ``LANGSMITH_TRACING=true``, capturing inputs,
        the generated answer, and end-to-end latency automatically.

        Args:
            question:     The user's question to answer.
            ground_truth: Expected answer for RAGAS evaluation. Pass an empty
                          string when ground truth is unavailable (e.g. in
                          interactive use). Defaults to ``""``.

        Returns:
            ``RAGResult`` with:

            - ``question``    : the original question.
            - ``answer``      : the generated answer string.
            - ``contexts``    : list of retrieved ``page_content`` strings.
            - ``ground_truth``: the provided expected answer.
            - ``strategy``    : active strategy name for downstream grouping.
            - ``latency_ms``  : end-to-end pipeline latency in milliseconds.

        Note:
            The retriever is called once to populate ``contexts`` and once
            inside ``self._chain.invoke(question)``. This double retrieval is
            intentional — it ensures that ``RAGResult.contexts`` reflects the
            exact passages the chain used, rather than a separate retrieval call
            that could return different results if the vector store is updated
            between calls.
        """
        start = time.monotonic()

        contexts = [
            doc.page_content
            for doc in self._retriever.retrieve(question)
        ]

        answer: str = self._chain.invoke(question)

        return RAGResult(
            question=question,
            answer=answer,
            contexts=contexts,
            ground_truth=ground_truth,
            strategy=self._strategy_name,
            latency_ms=(time.monotonic() - start) * 1000,
        )