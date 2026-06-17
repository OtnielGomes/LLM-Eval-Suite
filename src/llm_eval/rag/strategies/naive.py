"""
src/llm_eval/rag/strategies/naive.py
"""
from __future__ import annotations
from langchain_core.documents import Document
from langchain_chroma import Chroma
from llm_eval.shared.config import TOP_K_RETRIEVAL



class NaiveRetriever:
    """Busca por similaridade direta (cosine similarity)."""


    def __init__(self, vectorstore: Chroma, k: int = TOP_K_RETRIEVAL):
        self._vs = vectorstore
        self._k  = k


    def retrieve(self, query: str) -> list[Document]:
        return self._vs.similarity_search(query, k=self._k)


    def retrieve_with_scores(self, query: str) -> list[tuple[Document, float]]:
        return self._vs.similarity_search_with_relevance_scores(query, k=self._k)


    @property
    def name(self) -> str:
        return "naive"