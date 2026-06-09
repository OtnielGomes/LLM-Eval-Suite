"""
scripts/evaluate_rag.py
=======================
CLI entry point for evaluating RAG pipeline strategies using the RAGAS framework.

Runs one or more retrieval strategies (naive, hyde, reranking) against a curated
QA dataset, computes the four core RAGAS metrics, and writes per-strategy JSON
reports to ``results/``.

Entry point:
    uv run evaluate-rag

Usage examples:
    uv run evaluate-rag
    uv run evaluate-rag --limit 5
    uv run evaluate-rag --strategy naive --limit 3

Architecture:
    - LLM evaluator  : Ollama Cloud via ``langchain-openai`` (no Google API / ADC required)
    - Embeddings     : Ollama local ``nomic-embed-text``     (no OPENAI_API_KEY required)
    - Metrics        : context_precision, context_recall, faithfulness, answer_relevancy

Environment variables (set in ``.env``):
    OLLAMA_API_KEY        : API key for Ollama Cloud (required).
    OLLAMA_CLOUD_MODEL    : Model name override (optional, falls back to config default).

Output schema (``results/ragas_<strategy>.json``):
    {
        "strategy": str,         # "naive" | "hyde" | "reranking"
        "n":        int,         # number of QA pairs evaluated
        "scores":   list[dict],  # per-question RAGAS metric scores
    }

Dependencies:
    ragas, langchain-ollama, langchain-openai, datasets, python-dotenv
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import cast

from dotenv import load_dotenv
load_dotenv()

from datasets import Dataset
from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from scripts.ingest import get_vectorstore
from src.llm_eval.rag.rag_pipeline import RAGPipeline, StrategyName
from src.llm_eval.shared.config import (
    OLLAMA_BASE_URL,
    OLLAMA_CLOUD_BASE_URL,
    OLLAMA_CLOUD_MODEL,
    OLLAMA_EMBED_MODEL,
)
from src.llm_eval.shared.types import RAGResult


# Available RAG strategies passed to RAGPipeline

STRATEGIES = ["naive", "hyde", "reranking"]


# ---------------------------------------------------------------------------
# RAGAS LLM
# ---------------------------------------------------------------------------

def _make_ragas_llm() -> LangchainLLMWrapper:
    
    """Instantiate the LLM used by RAGAS to compute evaluation metrics.

    Uses ``ChatOpenAI`` pointed at Ollama Cloud instead of
    ``ChatGoogleGenerativeAI``, eliminating the need for a ``GOOGLE_API_KEY``
    or Application Default Credentials (ADC).

    The model and base URL are resolved in order:
    1. ``OLLAMA_CLOUD_MODEL`` environment variable.
    2. ``OLLAMA_CLOUD_MODEL`` constant from ``shared.config``.

    Returns:
        A ``LangchainLLMWrapper`` around a zero-temperature ``ChatOpenAI``
        instance configured for Ollama Cloud.

    Raises:
        ValueError: If ``OLLAMA_API_KEY`` is not set in the environment.
    """

    api_key = os.getenv("OLLAMA_API_KEY")
    if not api_key:
        raise ValueError(
            "OLLAMA_API_KEY not found. Configure in .env:\n"
            "  OLLAMA_API_KEY=your-key (https://ollama.com/settings/keys)"
        )
    llm = ChatOpenAI(
        model=os.getenv("OLLAMA_CLOUD_MODEL", OLLAMA_CLOUD_MODEL),
        base_url=OLLAMA_CLOUD_BASE_URL,
        api_key=SecretStr(api_key),
        temperature=0,
    )
    return LangchainLLMWrapper(llm)


# ── Embeddings avaliador (Ollama local) ────────────────────────────────────────

def _make_ragas_embeddings() -> LangchainEmbeddingsWrapper:
    """
    FIX: RAGAS tenta criar OpenAIEmbeddings por padrão quando embeddings não
    é passado explicitamente — falha sem OPENAI_API_KEY.
    Solução: passar OllamaEmbeddings local, consistente com o ingest.py.
    """
    return LangchainEmbeddingsWrapper(
        OllamaEmbeddings(
            model=OLLAMA_EMBED_MODEL,   # "nomic-embed-text"
            base_url=OLLAMA_BASE_URL,   # "http://localhost:11434"
        )
    )


# ── Carregamento do dataset ────────────────────────────────────────────────────

def _load_qa_pairs(limit: int = 0) -> list[dict]:
    dataset_path = Path("src/llm_eval/datasets/curated_qa.jsonl")
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}\n"
            "Run 'uv run build-datasets' first."
        )
    pairs = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
    if limit and limit > 0:
        pairs = pairs[:limit]
    return pairs


# ---------------------------------------------------------------------------
# Strategy runner
# ---------------------------------------------------------------------------

def _run_strategy(strategy: str, qa_pairs: list[dict]) -> dict:
    """Run a single RAG strategy against all QA pairs and compute RAGAS metrics.

    Workflow:
        1. Loads the vector store via :func:`scripts.ingest.get_vectorstore`.
        2. Instantiates :class:`RAGPipeline` with the given strategy.
        3. Queries the pipeline for every QA pair, collecting answers and
           retrieved contexts.
        4. Wraps results in a Hugging Face ``Dataset`` and calls
           :func:`ragas.evaluate` with the four core metrics.
        5. Serialises the per-question scores to
           ``results/ragas_<strategy>.json``.

    RAGAS metrics computed:
        - ``context_precision``  : fraction of retrieved contexts that are relevant.
        - ``context_recall``     : fraction of ground-truth information covered by contexts.
        - ``faithfulness``       : degree to which the answer is grounded in contexts.
        - ``answer_relevancy``   : semantic relevance of the answer to the question.

    Args:
        strategy:  One of ``"naive"``, ``"hyde"``, or ``"reranking"``.
        qa_pairs:  List of QA dicts, each with ``"question"`` and optional ``"answer"`` keys.

    Returns:
        Dict with keys ``"strategy"`` (str) and ``"scores"`` (list[dict] of
        per-question metric values).

    Note:
        ``scores.to_pandas()`` is used when available for serialisation. If
        the RAGAS version does not expose ``to_pandas``, an empty dict is saved
        instead.
    """

    vs = get_vectorstore()
    pipeline = RAGPipeline(
        vectorstore=vs,
        strategy=cast(StrategyName, strategy),  # FIX: str → StrategyName
    )

    results: list[dict] = []
    start = time.monotonic()

    print(f"\n{'=' * 60}")
    print(f" RAG Eval: {strategy} | {len(qa_pairs)} questions")
    print(f"{'=' * 60}")

    for i, qa in enumerate(qa_pairs, start=1):
        t0 = time.monotonic()
        out: RAGResult = pipeline.query(
            question=qa["question"],
            ground_truth=qa.get("answer", ""),
        )
        latency_ms = int((time.monotonic() - t0) * 1000)
        print(f" [{i:>2}/{len(qa_pairs)}] latency: {latency_ms}ms | contexts: {len(out.contexts)}")
        results.append({
            "question": out.question,
            "answer": out.answer,
            "contexts": out.contexts,
            "ground_truth": out.ground_truth,
        })

    elapsed = time.monotonic() - start
    print(f"\n ✓ {len(results)} answers generated in {elapsed:.1f}s")
    print("Computing RAGAS metrics ...")

    ragas_dataset = Dataset.from_list(results)  # FIX: list[dict] → Dataset
    scores = evaluate(
        ragas_dataset,
        metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
        llm=_make_ragas_llm(),
        embeddings=_make_ragas_embeddings(),    # FIX: vita OpenAIEmbeddings default
    )

    # Save results
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"ragas_{strategy}.json"
    scores_dict = (
        scores.to_pandas().to_dict(orient="records")
        if hasattr(scores, "to_pandas") else {}
    )
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(
            {"strategy": strategy, "n": len(qa_pairs), "scores": scores_dict},
            f, indent=2, ensure_ascii=False,
        )
    print(f" ✓ Results save to '{out_file}'")

    return {"strategy": strategy, "scores": scores_dict}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:

    """Parse command-line arguments for the RAG evaluation runner.

    Returns:
        argparse.Namespace with the following attributes:

        - ``limit`` (int):    Maximum number of QA pairs to evaluate.
                              ``0`` means evaluate the full dataset.
        - ``strategy`` (str): RAG strategy to run. One of ``"naive"``,
                              ``"hyde"``, ``"reranking"``, or ``"all"``
                              (default) to run every strategy sequentially.
    """

    parser = argparse.ArgumentParser(description="Evaluate RAG strategies with RAGAS.")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limits the number of questions (0 = all)",
    )
    parser.add_argument(
        "--strategy",
        choices=STRATEGIES + ["all"],
        default="all",
        help="RAG strategy to evaluate (default: all)",
    )
    return parser.parse_args()


def main() -> None:

    """Orchestrate the RAG evaluation pipeline.

    Loads QA pairs, resolves the target strategy list, and calls
    :func:`_run_strategy` for each. Prints a completion summary when all
    strategies finish.

    This function is registered as the ``evaluate-rag`` entry point in
    ``pyproject.toml`` and is invoked via ``uv run evaluate-rag``.
    """

    args = parse_args()
    qa_pairs = _load_qa_pairs(limit=args.limit)
    strategies = STRATEGIES if args.strategy == "all" else [args.strategy]

    for strategy in strategies:
        _run_strategy(strategy, qa_pairs)

    print("\n✅ Evaluation completed. Results in 'results/'")


if __name__ == "__main__":
    main()