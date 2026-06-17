"""
src/llm_eval/rag/rag_pipeline.py

Orchestra retrieval + prompt + generation using LangChain LCEL.
Embedded LangSmith tracing — enabled via environment variable.

Usage:

from rag_pipeline import RAGPipeline

from ingest import get_vectorstore

vs = get_vectorstore()
pipeline = RAGPipeline(vectorstore=vs, strategy="naive")
result = pipeline.query("What is RAG?", ground_truth="...")

print(result.answer)

Available strategies: "naive" | "hyde" | "reranking" LangSmith: set LANGSMITH_TRACING=true and LANGSMITH_API_KEY in .env

LLM Backend: Ollama Cloud via OllamaCloudClient (OpenAI-compatible).
OLLAMA_API_KEY → https://ollama.com/settings/keys
OLLAMA_CLOUD_MODEL → ex: llama3.3:70b
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
# Prompt RAG
# ---------------------------------------------------------------------------


RAG_PROMPT = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer the question using ONLY the provided context.
If the context doesn't contain enough information, say "I don't have enough context to answer."


Context:
{context}


Question: {question}


Answer:""")



def _format_docs(docs: list[Document]) -> str:
    return "\n\n---\n\n".join(doc.page_content for doc in docs)






def _build_langchain_llm(temperature: float = 0.1) -> ChatOpenAI:
    return ChatOpenAI(
        base_url=OLLAMA_CLOUD_BASE_URL,
        api_key=SecretStr(os.getenv("OLLAMA_API_KEY") or ""),  
        model=os.getenv("OLLAMA_CLOUD_MODEL", OLLAMA_CLOUD_MODEL),
        temperature=temperature,
    )



# ---------------------------------------------------------------------------
# RAGPipeline
# ---------------------------------------------------------------------------


StrategyName = Literal["naive", "hyde", "reranking"]



class RAGPipeline:
    

    def __init__(
        self,
        vectorstore: Chroma,
        strategy: StrategyName = "naive",
        k: int = TOP_K_RETRIEVAL,
    ):
        
        api_key = os.getenv("OLLAMA_API_KEY", "")
        if not api_key:
            raise EnvironmentError(
                "OLLAMA_API_KEY notfound in .env.\n"
                "Create one at https://ollama.com/settings/keys"
            )


        self._llm = _build_langchain_llm(temperature=0.1)
        self._strategy_name = strategy
        self._retriever = self._build_retriever(vectorstore, strategy, k)
        self._chain = self._build_chain()


    def _build_retriever(
        self,
        vectorstore: Chroma,
        strategy: StrategyName,
        k: int,
    ) -> NaiveRetriever | HyDERetriever | RerankingRetriever:
        
        if strategy == "naive":
            return NaiveRetriever(vectorstore, k=k)


        client = OllamaCloudClient()


        if strategy == "hyde":
            return HyDERetriever(vectorstore, llm=client, k=k)
        return RerankingRetriever(vectorstore, llm=client, k=k)


    def _build_chain(self):  
        retriever = self._retriever.retrieve


        return (
            {
                "context": lambda q: _format_docs(retriever(q)),
                "question": RunnablePassthrough(),
            }
            | RAG_PROMPT
            | self._llm
            | StrOutputParser()
        )


    @traceable(name="rag_query")  
    def query(
        self,
        question: str,
        ground_truth: str = "",
    ) -> RAGResult:
        
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