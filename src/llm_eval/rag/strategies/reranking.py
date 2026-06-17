"""
src/llm_eval/rag/strategies/reranking.py
"""
from __future__ import annotations
from typing import Protocol
from langchain_core.documents import Document
from langchain_chroma import Chroma
from llm_eval.shared.config import TOP_K_RETRIEVAL



class _LLMClient(Protocol):
    def complete(self, prompt: str, system: str = "") -> object: ...



class RerankingRetriever:
    """
    Retrieval com reranking via LLM.
    Recupera k*2 candidatos, depois pede ao LLM para selecionar os k mais relevantes.
    """


    SYSTEM_PROMPT = (
        "You are a relevance judge. Given a question and a list of passages, "
        "select the indices of the most relevant passages. "
        "Return ONLY a comma-separated list of indices (e.g. 0,2,4)."
    )


    def __init__(self, vectorstore: Chroma, llm: _LLMClient, k: int = TOP_K_RETRIEVAL):
        self._vs  = vectorstore
        self._llm = llm
        self._k   = k


    def retrieve(self, query: str) -> list[Document]:
        candidates = self._vs.similarity_search(query, k=self._k * 2)
        if not candidates:
            return []
        try:
            passages = "\n".join(
                f"[{i}] {doc.page_content[:300]}" for i, doc in enumerate(candidates)
            )
            prompt = f"Question: {query}\n\nPassages:\n{passages}"
            resp   = self._llm.complete(prompt=prompt, system=self.SYSTEM_PROMPT)
            raw    = str(getattr(resp, "response", resp))
            indices = [int(x.strip()) for x in raw.split(",") if x.strip().isdigit()]
            indices = [i for i in indices if 0 <= i < len(candidates)]
            if indices:
                return [candidates[i] for i in indices[: self._k]]
        except Exception:
            pass
        return candidates[: self._k]


    def retrieve_with_scores(self, query: str) -> list[tuple[Document, float]]:
        docs = self.retrieve(query)
        return [(doc, 1.0) for doc in docs]


    @property
    def name(self) -> str:
        return "reranking"