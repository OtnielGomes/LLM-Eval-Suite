"""
scripts/run_benchmark.py
Entry point CLI: uv run run-benchmark

Use:
uv run run-benchmark --strategy zero_shot --limit 5
uv run run-benchmark --strategy zero_shot --local
uv run run-benchmark --all-strategies --limit 50

Exit:
results/benchmark/zero_shot_llama3.3-70b.json
results/benchmark/zero_shot_llama3.3-70b.csv
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
from src.llm_eval.shared.config import OLLAMA_LLM_MODEL, OLLAMA_CLOUD_MODEL
from src.llm_eval.benchmark.evaluator import Evaluator
from src.llm_eval.benchmark.prompt_strategies.zero_shot import ZeroShotStrategy
from src.llm_eval.benchmark.prompt_strategies.few_shot import FewShotStrategy
from src.llm_eval.benchmark.prompt_strategies.chain_of_thought import ChainOfThoughtStrategy

STRATEGIES = {
    "zero_shot": ZeroShotStrategy(),
    "few_shot": FewShotStrategy(),
    "chain_of_thought": ChainOfThoughtStrategy(),
}

# FIX: resultados na raiz do projeto, fora de scripts/
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "benchmark"

# ── Dataset ───────────────────────────────────────────────────────────────────

DEFAULT_DATASET = PROJECT_ROOT / "src" / "llm_eval" / "datasets" / "mmlu_sample.jsonl"


def load_dataset(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Run 'uv run build-datasets' first."
        )
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    item = json.loads(line)
                    # FIX: valida schema esperado pelo benchmark (múltipla escolha)
                    if "question" not in item or "answer" not in item:
                        raise ValueError(
                            f"Item sem campos obrigatórios 'question'/'answer': {item}"
                        )
                    items.append(item)
                except json.JSONDecodeError as e:
                    raise ValueError(f"JSON inválido em {path}: {e}") from e
    if not items:
        raise ValueError(f"Dataset vazio ou não encontrado: {path}")
    return items

# ── Persistência ──────────────────────────────────────────────────────────────

def save_results(results: list[EvalResult], strategy: str, model: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
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

# ── Client ────────────────────────────────────────────────────────────────────

def _build_client(use_local: bool) -> tuple:
    """
    Retorna (client, model_label, backend_name).
    --local → Ollama local (llama3.1:8b por padrão)
    default → Ollama Cloud (llama3.3:70b por padrão)
    """
    if use_local:
        from src.llm_eval.clients.ollama_cliente import OllamaClient
        return OllamaClient(), OLLAMA_LLM_MODEL, "Ollama Local"

    from src.llm_eval.clients.ollama_cloud_cliente import OllamaCloudClient
    return OllamaCloudClient(), OLLAMA_CLOUD_MODEL, f"Ollama Cloud ({OLLAMA_CLOUD_MODEL})"

# ── Runner ────────────────────────────────────────────────────────────────────

def run(
    dataset_path: str | Path,
    strategy_name: str,
    use_local: bool = False,
    limit: int | None = None,
) -> list[EvalResult]:
    if strategy_name not in STRATEGIES:
        raise ValueError(
            f"Strategy '{strategy_name}' invalid. "
            f"Options: {list(STRATEGIES.keys())}"
        )

    client, model_label, backend = _build_client(use_local)
    strategy = STRATEGIES[strategy_name]
    evaluator = Evaluator(llm=client)
    dataset = load_dataset(dataset_path)

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
        choices = item.get("choices", [])
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
            resp = client.complete(prompt, system=system)
            result = evaluator.evaluate_all(
                question=question,
                reference=expected,
                hypothesis=resp.response,
                strategy=strategy_name,
                model=model_label,
                latency_ms=resp.latency_ms,
            )
            _meta["output"] = resp.response
            _meta["latency_ms"] = resp.latency_ms
            _meta["judge_score"] = result.judge_score
            _meta["is_correct"] = result.judge_correct
            _meta["bleu"] = result.bleu
            _meta["rougeL"] = result.rougeL

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

    summary = Evaluator.summarize(results)
    total_time = time.monotonic() - start_total

    print(f"\n{'-'*60}")
    print(f" Resultados — {strategy_name} | {backend}")
    print(f"{'-'*60}")
    print(f" Items evaluated : {summary['n']}")
    print(f" BLEU mean       : {summary['bleu_mean']:.4f}")
    print(f" ROUGE-1 mean    : {summary['rouge1_mean']:.4f}")
    print(f" ROUGE-2 mean    : {summary['rouge2_mean']:.4f}")
    print(f" ROUGE-L mean    : {summary['rougeL_mean']:.4f}")
    print(f" Judge score     : {summary['judge_score_mean']:.2f}/5.0")
    print(f" Judge accuracy  : {summary['judge_accuracy']*100:.1f}%")
    print(f" Latency mean    : {summary['latency_ms_mean']:.0f}ms")
    print(f" Total time      : {total_time:.0f}s")

    out = save_results(results, strategy_name, model_label)
    print(f"\n ✅ JSON: {out}")
    print(f" ✅ CSV : {out.with_suffix('.csv')}\n")

    log_run_summary(strategy_name, model_label, summary)
    if is_enabled():
        print(" ✅ LangSmith: https://smith.langchain.com/projects")

    return results

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark de LLMs com BLEU, ROUGE e LLM-as-judge.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET),
        help="Path to the JSONL dataset (default: mmlu_sample.jsonl)",
    )
    parser.add_argument(
        "--strategy",
        choices=list(STRATEGIES.keys()),
        default="zero_shot",
        help="Prompt strategy (default: zero_shot)",
    )
    parser.add_argument(
        "--all-strategies",
        action="store_true",
        help="Run zero_shot, few_shot, and chain_of_thought in sequence.",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use Ollama local instead of Ollama Cloud.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limits the number of items evaluated.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    strategies = list(STRATEGIES.keys()) if args.all_strategies else [args.strategy]

    for strategy in strategies:
        run(
            dataset_path=args.dataset,
            strategy_name=strategy,
            use_local=args.local,
            limit=args.limit,
        )


if __name__ == "__main__":
    main()