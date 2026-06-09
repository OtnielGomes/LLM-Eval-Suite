"""
scripts/run_benchmark.py
========================
CLI entry point for running LLM benchmark evaluations across prompt strategies and backends.

Loads a JSONL dataset, applies a prompt strategy (zero-shot, few-shot, or
chain-of-thought), queries the target LLM backend, evaluates each response
with BLEU, ROUGE, and LLM-as-judge, and writes per-strategy results to
``results/`` as both JSON and CSV.

Entry point:
    uv run run-benchmark

Usage examples:
    uv run run-benchmark --strategy zero_shot --limit 5
    uv run run-benchmark --strategy zero_shot --local
    uv run run-benchmark --strategy zero_shot --gemini
    uv run run-benchmark --all-strategies --limit 50

Output files:
    results/<strategy>_<model>.json   — full per-item EvalResult records
    results/<strategy>_<model>.csv    — same data in tabular format

Backend selection (mutually exclusive flags):
    --local   → OllamaClient       (localhost, free, requires local GPU/CPU)
    default   → OllamaCloudClient  (Ollama Cloud, zero cost)

Observability:
    All LLM calls are traced via LangSmith when ``LANGSMITH_API_KEY`` is set.
    Run summaries are logged after each strategy completes.

Dependencies:
    langchain, ragas, nltk, rouge-score, python-dotenv
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from src.llm_eval.shared.langsmith_tracer import trace_llm_call, log_run_summary, is_enabled

from src.llm_eval.shared.types import EvalResult
from src.llm_eval.shared.config import  OLLAMA_LLM_MODEL, OLLAMA_CLOUD_MODEL
from src.llm_eval.benchmark.evaluator import Evaluator
from src.llm_eval.benchmark.prompt_strategies.zero_shot import ZeroShotStrategy
from src.llm_eval.benchmark.prompt_strategies.few_shot import FewShotStrategy
from src.llm_eval.benchmark.prompt_strategies.chain_of_thought import ChainOfThoughtStrategy


STRATEGIES = {
    "zero_shot": ZeroShotStrategy(),
    "few_shot": FewShotStrategy(),
    "chain_of_thought": ChainOfThoughtStrategy(),
}

RESULTS_DIR = Path(__file__).parent / "results"


def load_dataset(path: str | Path) -> list[dict]:

    """Load a JSONL benchmark dataset from disk.

    Reads the file line by line, skipping blank lines, and deserialises each
    line as a JSON object. Each item is expected to contain at minimum the
    keys ``"question"`` and ``"answer"``. An optional ``"choices"`` key is
    used by multiple-choice strategies.

    Args:
        path: Path to the ``.jsonl`` dataset file.

    Returns:
        List of dicts, one per dataset item.

    Raises:
        ValueError: If the file is empty or contains no non-blank lines.
        FileNotFoundError: If ``path`` does not exist (raised by ``open``).
    """

    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    if not items:
        raise ValueError(f"Empty or missing dataset: {path}")
    return items



def save_results(results: list[EvalResult], strategy: str, model: str) -> Path:

    """Serialise benchmark results to JSON and CSV files under ``RESULTS_DIR``.

    The output filename stem is ``<strategy>_<model>`` where ``model`` has
    ``/`` and ``:`` replaced with ``-`` to produce a valid filename on all
    operating systems (e.g. ``zero_shot_llama3.3-70b``).

    Both files share the same stem; only the extension differs:
    - ``.json``: list of ``EvalResult.to_dict()`` records, pretty-printed.
    - ``.csv``: same records in tabular format with a header row.

    Args:
        results:  List of evaluated ``EvalResult`` objects to persist.
        strategy: Strategy name used as the first component of the filename
                  (e.g. ``"zero_shot"``).
        model:    Model label used as the second component of the filename
                  (e.g. ``"llama3.3:70b"`` → sanitised to ``"llama3.3-70b"``).

    Returns:
        Path to the generated ``.json`` file. The corresponding ``.csv`` file
        is at the same path with ``.csv`` extension (``path.with_suffix('.csv')``).

    Note:
        ``RESULTS_DIR`` is created automatically if it does not exist.
        If ``results`` is empty, the CSV is written with a header only.
    """

    RESULTS_DIR.mkdir(exist_ok=True)
    stem = f"{strategy}_{model.replace('/', '-').replace(':', '-')}"

    json_path = RESULTS_DIR / f"{stem}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([r.to_dict() for r in results], f, indent=2, ensure_ascii=False)

    csv_path = RESULTS_DIR / f"{stem}.csv"
    if results:
        fieldnames = list(results[0].to_dict().keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(r.to_dict() for r in results)

    return json_path


def _build_client(use_local: bool, use_gemini: bool):
    
    """Instantiate the appropriate LLM client based on CLI flags.

    Resolution priority:
        1. ``--local``  → ``OllamaClient`` (localhost Ollama daemon)
        32. default      → ``OllamaCloudClient`` (Ollama Cloud, zero cost)

    The two flags are declared as a mutually exclusive group in ``parse_args``,
    so only one can be active at a time.

    Args:
        use_local:  If ``True``, use the local Ollama backend.

    Returns:
        A 3-tuple of ``(client, model_label, backend_display_name)`` where:

        - ``client``       is an instance of ``OllamaClient``,
          ``GeminiClient``, or ``OllamaCloudClient``.
        - ``model_label``  is the model identifier string used for filenames
          and LangSmith traces.
        - ``backend_display_name`` is a human-readable label printed in the
          benchmark header.

    Note:
        Clients are imported lazily inside each branch to avoid importing
    """

    if use_local:
        from src.llm_eval.clients.ollama_cliente import OllamaClient
        return OllamaClient(), OLLAMA_LLM_MODEL, "Ollama Local"

    from src.llm_eval.clients.ollama_cloud_cliente import OllamaCloudClient
    return OllamaCloudClient(), OLLAMA_CLOUD_MODEL, f"Ollama Cloud ({OLLAMA_CLOUD_MODEL})"

# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run(
    dataset_path: str,
    strategy_name: str,
    use_local: bool = False,
    use_gemini: bool = False,
    limit: int | None = None,
) -> list[EvalResult]:
    
    """Execute a full benchmark run for a single strategy and backend.

    Workflow:
        1. Resolves the LLM client via :func:`_build_client`.
        2. Loads and optionally truncates the dataset via :func:`load_dataset`.
        3. For each item: builds the prompt, calls the LLM, evaluates the
           response (BLEU + ROUGE + LLM-as-judge), and records the result.
        4. All LLM calls are wrapped in :func:`trace_llm_call` for LangSmith
           tracing; metadata (latency, judge score, metrics) is attached to
           each trace span.
        5. Prints a progress line every 10 items and a full summary at the end.
        6. Persists results via :func:`save_results` and logs the run summary
           to LangSmith via :func:`log_run_summary`.

    Args:
        dataset_path:  Path to the JSONL dataset file to evaluate against.
        strategy_name: Key into the ``STRATEGIES`` registry. One of
                       ``"zero_shot"``, ``"few_shot"``, or ``"chain_of_thought"``.
        use_local:     If ``True``, route requests to the local Ollama daemon.
        use_gemini:    If ``True``, route requests to the Gemini API.
        limit:         If set, evaluate only the first ``limit`` items.
                       Useful for smoke-testing the pipeline before a full run.

    Returns:
        List of ``EvalResult`` objects, one per evaluated dataset item.

    Raises:
        ValueError: If ``strategy_name`` is not a key in ``STRATEGIES``.
    """

    if strategy_name not in STRATEGIES:
        raise ValueError(
            f"Strategy '{strategy_name}' invalid. "
            f"Options: {list(STRATEGIES.keys())}"
        )

    client, model_label, backend = _build_client(use_local, use_gemini)
    strategy  = STRATEGIES[strategy_name]
    evaluator = Evaluator(llm=client)
    dataset   = load_dataset(dataset_path)

    if limit:
        dataset = dataset[:limit]

    n = len(dataset)
    print(f"\n{'='*60}")
    print(f" Benchmark: {strategy_name} | {backend} | {n} items")
    print(f"{'='*60}\n")

    results: list[EvalResult] = []
    start_total = time.monotonic()

    for i, item in enumerate(dataset, 1):
        question = item["question"]
        choices  = item.get("choices", [])
        expected = item["answer"]

        prompt = strategy.build_prompt(question, choices)
        config = strategy.get_config()
        system = getattr(config, "system_instruction", "") or ""

        with trace_llm_call(
            name=f"{strategy_name}/complete",
            model=model_label,
            strategy=strategy_name,
            prompt=prompt,
            system=system,
        ) as _meta:
            resp   = client.complete(prompt, system=system)
            result = evaluator.evaluate_all(
                question=question,
                reference=expected,
                hypothesis=resp.response,
                strategy=strategy_name,
                model=model_label,
                latency_ms=resp.latency_ms,
            )
            _meta["output"]      = resp.response
            _meta["latency_ms"]  = resp.latency_ms
            _meta["judge_score"] = result.judge_score
            _meta["is_correct"]  = result.judge_correct
            _meta["bleu"]        = result.bleu
            _meta["rougeL"]      = result.rougeL

        results.append(result)

        if i % 10 == 0 or i == n:
            elapsed = time.monotonic() - start_total
            pct = i / n * 100
            print(
                f" [{i:>4}/{n}] {pct:5.1f}% | "
                f"judge: {result.judge_score}/5 | "
                f"ROUGE-L: {result.rougeL:.3f} | "
                f"elapsed: {elapsed:.0f}s"
            )

    summary    = Evaluator.summarize(results)
    total_time = time.monotonic() - start_total

    print(f"\n{'-'*60}")
    print(f" Results — {strategy_name} | {backend}")
    print(f"{'-'*60}")
    print(f" Items evaluated : {summary['n']}")
    print(f" BLEU mean      : {summary['bleu_mean']:.4f}")
    print(f" ROUGE-1 mean   : {summary['rouge1_mean']:.4f}")
    print(f" ROUGE-2 mean   : {summary['rouge2_mean']:.4f}")
    print(f" ROUGE-L mean   : {summary['rougeL_mean']:.4f}")
    print(f" Judge score     : {summary['judge_score_mean']:.2f}/5.0")
    print(f" Judge accuracy  : {summary['judge_accuracy']*100:.1f}%")
    print(f" Latencia mean  : {summary['latency_ms_mean']:.0f}ms")
    print(f" Time total     : {total_time:.0f}s")

    
    out = save_results(results, strategy_name, model_label)
    print(f"\n ✓ Save in: {out}")
    print(f" ✓ CSV in  : {out.with_suffix('.csv')}\n")

    # ► LangSmith: log apos salvar, dentro de run() onde strategy_name e model_label existem
    log_run_summary(strategy_name, model_label, summary)
    if is_enabled():
        print(" ✓ LangSmith: https://smith.langchain.com/projects")

    return results

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLM benchmark with BLEU, ROUGE, and LLM-as-judge.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        default="src/llm_eval/datasets/mmlu_sample.jsonl",
    )
    parser.add_argument(
        "--strategy",
        choices=list(STRATEGIES.keys()),
        default="zero_shot",
    )
    parser.add_argument("--all-strategies", action="store_true")

    backend = parser.add_mutually_exclusive_group()
    backend.add_argument("--local",  action="store_true")

    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:

    """Orchestrate one or more benchmark runs based on CLI arguments.

    Resolves the target strategy list (single strategy or all registered
    strategies) and calls :func:`run` for each. Results are written to
    ``results/`` automatically inside :func:`run`.

    This function is registered as the ``run-benchmark`` entry point in
    ``pyproject.toml`` and is invoked via ``uv run run-benchmark``.
    """
    
    args       = parse_args()
    strategies = list(STRATEGIES.keys()) if args.all_strategies else [args.strategy]

    for strategy in strategies:
        run(
            dataset_path=args.dataset,
            strategy_name=strategy,
            use_local=args.local,
            use_gemini=args.gemini,
            limit=args.limit,
        )


if __name__ == "__main__":
    main()