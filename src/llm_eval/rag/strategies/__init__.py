"""
src/llm_eval/rag/strategies/__init__.py
=======================================
Public API for the RAG retrieval strategy sub-package.

Exposes all three retrieval strategy implementations so that consumers can
import directly from the sub-package root without knowing the internal module
structure::

    # Preferred — import from sub-package root
    from llm_eval.rag.strategies import NaiveRetriever, HyDERetriever

    # Instead of the verbose internal path
    from llm_eval.rag.strategies.naive import NaiveRetriever

Available strategies:

.. list-table::
   :header-rows: 1

   * - Class
     - Strategy
     - Description
   * - ``NaiveRetriever``
     - Baseline dense retrieval
     - Embeds the question directly and performs cosine similarity search.
   * - ``HyDERetriever``
     - Hypothetical Document Embeddings
     - Generates a synthetic answer first, embeds it, then retrieves. Improves
       recall on queries phrased differently from indexed documents.
   * - ``RerankingRetriever``
     - Retrieve-then-rerank
     - Fetches a larger candidate set and applies a cross-encoder reranker to
       select the most relevant contexts before generation.

Adding a new strategy:
    1. Create ``src/llm_eval/rag/strategies/<name>.py`` and implement the
       retrieval logic.
    2. Add an import and an ``__all__`` entry in this file.
    3. Register the strategy key in ``RAGPipeline``'s strategy registry
       (``rag_pipeline.py``).
"""
from llm_eval.rag.strategies.naive import NaiveRetriever
from llm_eval.rag.strategies.hyde import HyDERetriever
from llm_eval.rag.strategies.reranking import RerankingRetriever

__all__ = ["NaiveRetriever", "HyDERetriever", "RerankingRetriever"]