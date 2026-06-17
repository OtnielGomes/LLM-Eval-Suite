"""
src/llm_eval/rag/strategies/hyde.py
"""
from __future__ import annotations
from typing import Protocol
from langchain_core.documents import Document
from langchain_chroma import Chroma
from llm_eval.shared.config import TOP_K_RETRIEVAL



class _LLMClient(Protocol):
    def complete(self, prompt: str, system: str = "") -> object: ...



class HyDERetriever:
    """Retrieval com documento hipotético como pivot semântico."""


    SYSTEM_PROMPT = (
        "You are a knowledgeable assistant. "
        "Given a question, write a concise, factual paragraph that would "
        "directly answer it. Write only the answer — no preamble."
    )


    def __init__(self, vectorstore: Chroma, llm: _LLMClient, k: int = TOP_K_RETRIEVAL):
        self._vs  = vectorstore
        self._llm = llm
        self._k   = k


    def _generate_hypothetical_doc(self, query: str) -> str:
        resp = self._llm.complete(prompt=f"Question: {query}", system=self.SYSTEM_PROMPT)
        return str(getattr(resp, "response", resp))


    def retrieve(self, query: str) -> list[Document]:
        try:
            hyp_doc = self._generate_hypothetical_doc(query)
        except Exception:
            hyp_doc = query
        return self._vs.similarity_search(hyp_doc, k=self._k)


    def retrieve_with_scores(self, query: str) -> list[tuple[Document, float]]:
        try:
            hyp_doc = self._generate_hypothetical_doc(query)
        except Exception:
            hyp_doc = query
        return self._vs.similarity_search_with_relevance_scores(hyp_doc, k=self._k)


    @property
    def name(self) -> str:
        return "hyde"