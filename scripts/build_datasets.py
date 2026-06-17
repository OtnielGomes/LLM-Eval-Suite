"""
scripts/build_datasets.py
Entry point CLI: uv run build-datasets

Usage:
uv run build-datasets
uv run build-datasets --mmlu-n 500 --humaneval-n 100
uv run build-datasets --skip-humaneval
uv run build-datasets --skip-mmlu
uv run build-datasets --rag-qa-n 3 --rag-qa-model qwen3-coder:480b-cloud

Datasets generated:
  mmlu_sample.jsonl      → benchmark MCQ  (run_benchmark.py)
  humaneval_sample.jsonl → benchmark MCQ  (run_benchmark.py)
  curated_qa.jsonl       → RAG Q&A discursive answers (evaluate_rag.py)

WARNING: curated_qa.jsonl is generated from the RAG corpus .txt files via LLM,
         NOT as a merge of MMLU data. The two datasets have incompatible schemas
         and serve different scripts.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any, cast

from tqdm import tqdm

DATASETS_DIR    = Path(__file__).parent.parent / "src" / "llm_eval" / "datasets"
MMLU_PATH       = DATASETS_DIR / "mmlu_sample.jsonl"
HUMANEVAL_PATH  = DATASETS_DIR / "humaneval_sample.jsonl"
CURATED_QA_PATH = DATASETS_DIR / "curated_qa.jsonl"
RAW_DIR         = DATASETS_DIR / "raw"

MMLU_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "college_biology",
    "college_chemistry", "college_computer_science", "college_mathematics",
    "college_physics", "conceptual_physics", "elementary_mathematics",
    "formal_logic", "global_facts", "high_school_biology",
    "high_school_chemistry", "high_school_computer_science",
    "high_school_mathematics", "high_school_physics", "high_school_psychology",
    "high_school_world_history", "logical_fallacies", "machine_learning",
    "moral_scenarios", "philosophy", "professional_accounting",
    "professional_law", "professional_medicine", "professional_psychology",
    "world_religions",
]

# ---------------------------------------------------------------------------
# MMLU — schema: {question, choices, answer:"A"|"B"|"C"|"D", subject}
# Used by: run_benchmark.py
# ---------------------------------------------------------------------------

def build_mmlu(n: int = 200, seed: int = 42) -> None:
    try:
        from datasets import load_dataset
    except ImportError:
        print(" ✗ Package 'datasets' not found. Run: uv sync")
        raise SystemExit(1)

    print(f"\n[MMLU] Downloading {n} items from {len(MMLU_SUBJECTS)} subjects...")
    items_per_subject = max(1, n // len(MMLU_SUBJECTS))
    collected: list[dict[str, Any]] = []
    letter_map = {0: "A", 1: "B", 2: "C", 3: "D"}

    for subject in tqdm(MMLU_SUBJECTS, desc=" Subjects"):
        try:
            ds = load_dataset("cais/mmlu", subject, split="test", trust_remote_code=True)
            rows: list[dict[str, Any]] = cast(list[dict[str, Any]], list(ds))
            random.seed(seed)
            sample = random.sample(rows, min(items_per_subject, len(rows)))
            for row in sample:
                collected.append({
                    "question": row["question"].strip(),
                    "choices":  [c.strip() for c in row["choices"]],
                    "answer":   letter_map[int(row["answer"])],
                    "subject":  subject,
                })
        except Exception as e:
            tqdm.write(f" ⚠ Subject '{subject}' failed: {e}")

    random.seed(seed)
    random.shuffle(collected)
    collected = collected[:n]

    _save_jsonl(collected, MMLU_PATH)
    print(f" ✓ {len(collected)} items saved to {MMLU_PATH}")
    _print_subject_distribution(collected)


def _print_subject_distribution(items: list[dict]) -> None:
    from collections import Counter
    dist = Counter(item["subject"] for item in items)
    print("\n Distribution by subject (top 10):")
    for subject, count in dist.most_common(10):
        print(f"   {subject:<40} {count:>3} {'█' * count}")

# ---------------------------------------------------------------------------
# HumanEval — schema: {question, choices, answer:"A"|"B"|"C"|"D", task_id}
# Used by: run_benchmark.py
# ---------------------------------------------------------------------------

def build_humaneval(n: int = 50, seed: int = 42) -> None:
    try:
        from datasets import load_dataset
    except ImportError:
        print(" ✗ Package 'datasets' not found.")
        raise SystemExit(1)

    print(f"\n[HumanEval] Downloading {n} items...")
    ds = load_dataset("openai/openai_humaneval", split="test", trust_remote_code=True)
    rows: list[dict[str, Any]] = cast(list[dict[str, Any]], list(ds))
    random.seed(seed)
    sample = random.sample(rows, min(n, len(rows)))
    collected: list[dict[str, Any]] = []

    for row in tqdm(sample, desc=" Converting to MCQ"):
        prompt    = row["prompt"].strip()
        canonical = row["canonical_solution"].strip()
        fn_match  = re.search(r"def (\w+)\(", prompt)
        fn_name   = fn_match.group(1) if fn_match else "function"
        question  = (
            f"Given the following Python function signature and docstring:\n\n"
            f"```python\n{prompt[:400]}\n```\n\n"
            f"Which of the following best describes the correct implementation of `{fn_name}`?"
        )
        choices, answer = _generate_code_choices(canonical)
        collected.append({
            "question": question,
            "choices":  choices,
            "answer":   answer,
            "task_id":  row["task_id"],
        })

    _save_jsonl(collected, HUMANEVAL_PATH)
    print(f" ✓ {len(collected)} items saved to {HUMANEVAL_PATH}")


def _generate_code_choices(canonical: str) -> tuple[list[str], str]:
    def mutate_operators(c: str) -> str:
        c = re.sub(r">=", ">", c)
        return re.sub(r"<=", "<", c)

    def mutate_init(c: str) -> str:
        c = re.sub(r"\b0\b", "1", c, count=2)
        return re.sub(r"\[\]", "{}", c, count=1)

    def mutate_condition(c: str) -> str:
        return re.sub(r"\bif\b", "if not", c, count=1)

    distractors = [mutate_operators(canonical), mutate_init(canonical), mutate_condition(canonical)]
    distractors = [d for d in distractors if d != canonical]
    while len(distractors) < 3:
        distractors.append("return None  # incomplete implementation")

    options     = [canonical] + distractors[:3]
    random.shuffle(options)
    correct_idx = options.index(canonical)
    choices     = [f"{opt[:120]}..." if len(opt) > 120 else opt for opt in options]
    return choices, chr(65 + correct_idx)

# ---------------------------------------------------------------------------
# RAG QA — schema: {question, ground_truth, source_file, subject}
# Used by: evaluate_rag.py  ← the ONLY valid consumer of this dataset
#
# Generation: the LLM reads each .txt from the corpus and produces
# one discursive Q&A pair per document. ground_truth is a prose answer
# extracted from the text — never an MCQ letter.
# ---------------------------------------------------------------------------

_QA_SYSTEM = (
    "You are a dataset curator. Given a text passage, generate exactly ONE "
    "question-answer pair that tests factual understanding of the passage. "
    "The answer must be a complete sentence (15-80 words) derived strictly "
    "from the passage — do not add external knowledge. "
    "Respond ONLY with valid JSON, no markdown fences:\n"
    '{"question": "...", "ground_truth": "..."}'
)


def _generate_qa_for_doc(
    text: str,
    source_file: str,
    subject: str,
    client: Any,
    model: str,
    max_chars: int = 3000,
) -> dict | None:
    snippet = text[:max_chars].strip()
    prompt  = f"Passage:\n\n{snippet}\n\nGenerate the JSON now:"
    try:
        resp = client.complete(prompt, system=_QA_SYSTEM)
        raw  = resp.response.strip()
        # Strip markdown fences if the model ignores the instruction
        raw  = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw  = re.sub(r"\s*```$", "", raw)
        item = json.loads(raw)
        if "question" not in item or "ground_truth" not in item:
            raise ValueError("Missing required keys")
        if len(item["ground_truth"].split()) < 5:
            raise ValueError(f"ground_truth too short: {item['ground_truth']!r}")
        item["source_file"] = source_file
        item["subject"]     = subject
        return item
    except Exception as e:
        tqdm.write(f"   ⚠ Skipped {source_file}: {e}")
        return None


def build_rag_qa(
    n_per_doc: int = 1,
    model: str | None = None,
    seed: int = 42,
) -> None:
    """
    Generate curated_qa.jsonl from the RAG corpus .txt files.

    For each .txt file under datasets/raw/:
      - Sends the text (truncated to 3k chars) to the LLM
      - Requests one discursive {question, ground_truth} pair
      - Appends the result to curated_qa.jsonl

    Requires Ollama Cloud (OLLAMA_API_KEY must be set in .env).
    """
    from dotenv import load_dotenv
    load_dotenv()

    from src.llm_eval.clients.ollama_cloud_cliente import OllamaCloudClient
    from src.llm_eval.shared.config import OLLAMA_CLOUD_MODEL

    gen_model = model or os.getenv("OLLAMA_CLOUD_MODEL") or OLLAMA_CLOUD_MODEL
    client    = OllamaCloudClient()

    txt_files = sorted(RAW_DIR.rglob("*.txt"))
    if not txt_files:
        print(f" ⚠ No .txt files found in {RAW_DIR}. Run: uv run populate-raw-docs")
        return

    print(f"\n[RAG QA] Generating Q&A pairs from {len(txt_files)} docs | model: {gen_model}")
    print(f"         Output → {CURATED_QA_PATH}\n")

    random.seed(seed)
    collected: list[dict] = []

    for txt_path in tqdm(txt_files, desc=" Docs"):
        subject = txt_path.parent.name
        text    = txt_path.read_text(encoding="utf-8")
        source  = str(txt_path.relative_to(RAW_DIR))

        item = _generate_qa_for_doc(
            text=text,
            source_file=source,
            subject=subject,
            client=client,
            model=gen_model,
        )
        if item:
            collected.append(item)
        time.sleep(0.2)  # avoid rate-limiting

    if not collected:
        print(" ✗ No Q&A pairs generated. Check your OLLAMA_API_KEY and model.")
        return

    _save_jsonl(collected, CURATED_QA_PATH)
    print(f"\n ✓ {len(collected)} Q&A pairs saved to {CURATED_QA_PATH}")
    _validate_rag_qa(CURATED_QA_PATH)


def _validate_rag_qa(path: Path) -> None:
    print(f"\n Validating {path.name}...")
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            item    = json.loads(line)
            missing = [k for k in ("question", "ground_truth") if k not in item]
            if missing:
                print(f" ⚠ Row {i+1}: missing fields: {missing}")
            else:
                gt_words = len(item["ground_truth"].split())
                print(
                    f" ✓ Row {i+1} ({item['subject']}): "
                    f"Q={item['question'][:55]}... | "
                    f"GT={gt_words} words"
                )

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_jsonl(items: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _validate(path: Path, expected_keys: list[str]) -> None:
    print(f"\n Validating {path.name}...")
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            item    = json.loads(line)
            missing = [k for k in expected_keys if k not in item]
            if missing:
                print(f" ⚠ Row {i+1}: missing fields: {missing}")
            else:
                print(f" ✓ Row {i+1}: OK — {item['question'][:60]}...")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build MMLU, HumanEval and RAG QA datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Datasets generated:\n"
            "  mmlu_sample.jsonl      → run_benchmark.py  (MCQ, answer='A'|'B'|'C'|'D')\n"
            "  humaneval_sample.jsonl → run_benchmark.py  (MCQ, answer='A'|'B'|'C'|'D')\n"
            "  curated_qa.jsonl       → evaluate_rag.py   (discursive Q&A, ground_truth=prose)\n"
        ),
    )
    parser.add_argument("--mmlu-n",      type=int, default=200)
    parser.add_argument("--humaneval-n", type=int, default=50)
    parser.add_argument(
        "--rag-qa-n", type=int, default=1,
        help="Q&A pairs per document for curated_qa.jsonl (default: 1)",
    )
    parser.add_argument(
        "--rag-qa-model", type=str, default=None,
        help="Model for RAG QA generation (default: OLLAMA_CLOUD_MODEL from .env)",
    )
    parser.add_argument("--seed",           type=int, default=42)
    parser.add_argument("--skip-mmlu",      action="store_true")
    parser.add_argument("--skip-humaneval", action="store_true")
    parser.add_argument(
        "--skip-rag-qa", action="store_true",
        help="Skip curated_qa.jsonl generation (requires an LLM call)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("=" * 60)
    print(" LLM-Eval-Suite — Dataset Builder")
    print("=" * 60)

    if not args.skip_mmlu:
        build_mmlu(n=args.mmlu_n, seed=args.seed)
        _validate(MMLU_PATH, ["question", "choices", "answer", "subject"])

    if not args.skip_humaneval:
        build_humaneval(n=args.humaneval_n, seed=args.seed)
        _validate(HUMANEVAL_PATH, ["question", "choices", "answer", "task_id"])

    if not args.skip_rag_qa:
        build_rag_qa(n_per_doc=args.rag_qa_n, model=args.rag_qa_model, seed=args.seed)
    else:
        print("\n[RAG QA] Skipped (--skip-rag-qa)")

    print("\n ✅ Datasets ready:")
    print("    run_benchmark → mmlu_sample.jsonl + humaneval_sample.jsonl")
    print("    evaluate_rag  → curated_qa.jsonl\n")


if __name__ == "__main__":
    main()
