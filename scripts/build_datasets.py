"""
scripts/build_datasets.py
=========================
CLI entry point for downloading and curating benchmark datasets.

Downloads samples from MMLU and HumanEval via Hugging Face ``datasets``,
converts them to a unified JSONL schema, and saves them to
``src/llm_eval/datasets/``.

Entry point:
    uv run build-datasets

Usage examples:
    uv run build-datasets
    uv run build-datasets --mmlu-n 500 --humaneval-n 100
    uv run build-datasets --skip-humaneval
    uv run build-datasets --skip-mmlu

Output schema (both datasets):
    {
        "question": str,       # question text or code prompt
        "choices":  list[str], # 4 answer options (A-D)
        "answer":   str,       # correct letter ("A", "B", "C" or "D")
        "subject":  str,       # MMLU only — academic subject
        "task_id":  str,       # HumanEval only — e.g. "HumanEval/42"
    }

Dependencies:
    datasets >= 2.0
    tqdm >= 4.0
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, cast

from tqdm import tqdm

# Datasets directory resolved relative to this script.
# No sys.path manipulation needed: llm_eval is installed via uv sync.
DATASETS_DIR   = Path(__file__).parent.parent / "src" / "llm_eval" / "datasets"
MMLU_PATH      = DATASETS_DIR / "mmlu_sample.jsonl"
HUMANEVAL_PATH = DATASETS_DIR / "humaneval_sample.jsonl"

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
# MMLU
# ---------------------------------------------------------------------------

def build_mmlu(n: int = 200, seed: int = 42) -> None:

    """Download and sample items from the MMLU benchmark.

    Loads the ``cais/mmlu`` dataset from Hugging Face, draws an equal number
    of items from each subject (``n // len(MMLU_SUBJECTS)``), shuffles the
    result with a fixed seed, and writes up to ``n`` items to
    ``mmlu_sample.jsonl``.

    Each output item follows the schema::

        {
            "question": str,       # raw question text
            "choices":  list[str], # exactly 4 answer options
            "answer":   str,       # correct letter: "A", "B", "C" or "D"
            "subject":  str,       # originating MMLU subject
        }

    Args:
        n:    Total number of items to collect across all subjects.
              Distributed evenly; subjects with fewer rows are capped at
              their actual size.
        seed: Random seed used for both per-subject sampling and the final
              global shuffle, ensuring reproducibility.

    Raises:
        SystemExit: If the ``datasets`` package is not installed.

    Note:
        Subjects that fail to load (network error, missing split, etc.) are
        skipped with a warning rather than aborting the entire run.
    """

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
            ds   = load_dataset("cais/mmlu", subject, split="test", trust_remote_code=True)
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
            tqdm.write(f" ⚠ Subject '{subject}' fail: {e}")

    random.seed(seed)
    random.shuffle(collected)
    collected = collected[:n]

    _save_jsonl(collected, MMLU_PATH)
    print(f" ✓ {len(collected)} Save items in {MMLU_PATH}")
    _print_subject_distribution(collected)


def _print_subject_distribution(items: list[dict]) -> None:

    """Print a bar chart of item counts grouped by MMLU subject.

    Uses ASCII block characters (█) as bars. Only the top 10 subjects
    by frequency are shown to keep the output readable.

    Args:
        items: List of MMLU items, each expected to have a ``"subject"`` key.
    """
    
    from collections import Counter
    dist = Counter(item["subject"] for item in items)
    print("\n Distribution by subject (top 10):")
    for subject, count in dist.most_common(10):
        print(f"   {subject:<40} {count:>3} {'█' * count}")


# ---------------------------------------------------------------------------
# HumanEval
# ---------------------------------------------------------------------------

def build_humaneval(n: int = 50, seed: int = 42) -> None:

    """Download and convert HumanEval problems to multiple-choice format.

    Loads the ``openai/openai_humaneval`` dataset from Hugging Face and
    converts each coding problem into a 4-option MCQ. The canonical solution
    is used as the correct answer (A-D, randomised), and three syntactically
    mutated distractors are generated via :func:`_generate_code_choices`.

    Each output item follows the schema::

        {
            "question": str,       # function signature + docstring as MCQ stem
            "choices":  list[str], # 4 code snippets (truncated to 120 chars)
            "answer":   str,       # correct letter: "A", "B", "C" or "D"
            "task_id":  str,       # original HumanEval ID, e.g. "HumanEval/0"
        }

    Args:
        n:    Number of problems to sample. Capped at the dataset size (164).
        seed: Random seed for reproducible sampling and choice shuffling.

    Raises:
        SystemExit: If the ``datasets`` package is not installed.
    """

    try:
        from datasets import load_dataset
    except ImportError:
        print(" ✗ Package 'datasets' not found.")
        raise SystemExit(1)

    print(f"\n[HumanEval] Downloading {n} items...")
    ds   = load_dataset("openai/openai_humaneval", split="test", trust_remote_code=True)
    rows: list[dict[str, Any]] = cast(list[dict[str, Any]], list(ds))
    random.seed(seed)
    sample    = random.sample(rows, min(n, len(rows)))
    collected: list[dict[str, Any]] = []

    for row in tqdm(sample, desc=" Converting for MCQ"):
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
    print(f" ✓ {len(collected)} Save items in {HUMANEVAL_PATH}")


def _generate_code_choices(canonical: str) -> tuple[list[str], str]:
    """Generate one correct answer and three distractor options for a code MCQ.

    Applies three deterministic mutation strategies to produce syntactically
    plausible but semantically incorrect variants of the canonical solution:

    - **Operator mutation**: replaces ``>=`` with ``>`` and ``<=`` with ``<``.
    - **Init mutation**: replaces the first two ``0`` literals with ``1`` and
      the first ``[]`` with ``{}``.
    - **Condition mutation**: inserts ``not`` into the first ``if`` condition.

    Mutations that produce output identical to ``canonical`` are discarded.
    If fewer than 3 distinct distractors remain after filtering, the list is
    padded with ``"return None  # incomplete implementation"``.

    Args:
        canonical: The ground-truth solution string from HumanEval.

    Returns:
        A tuple of ``(choices, answer)`` where:
        - ``choices`` is a list of 4 strings (each truncated to 120 chars).
        - ``answer`` is the letter (``"A"``–``"D"``) of the correct option.
    """
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
# Utilities
# ---------------------------------------------------------------------------

def _save_jsonl(items: list[dict], path: Path) -> None:

    """Serialise a list of dicts to a JSONL file, one JSON object per line.

    Creates any missing parent directories before writing.
    Uses ``ensure_ascii=False`` to preserve Unicode characters in questions.

    Args:
        items: List of dicts to serialise.
        path:  Destination file path. Parent directories are created if absent.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _validate(path: Path, expected_keys: list[str]) -> None:

    """Spot-check the first 3 lines of a JSONL file for required keys.

    Prints a ``✓`` confirmation for each valid line and a ``⚠`` warning
    listing any missing keys. Intended as a lightweight post-build sanity
    check rather than a full schema validation.

    Args:
        path:          Path to the JSONL file to validate.
        expected_keys: List of keys that must be present in every item.
    """

    print(f"\n Validatiing {path.name}...")
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            item    = json.loads(line)
            missing = [k for k in expected_keys if k not in item]
            if missing:
                print(f" ⚠ Line {i+1}: missing fields: {missing}")
            else:
                print(f" ✓ Line {i+1}: OK — {item['question'][:60]}...")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:

    """Parse command-line arguments for the dataset builder.

    Returns:
        argparse.Namespace with the following attributes:

        - ``mmlu_n`` (int):          Number of MMLU items to collect (default: 200).
        - ``humaneval_n`` (int):     Number of HumanEval items to collect (default: 50).
        - ``seed`` (int):            Random seed for reproducibility (default: 42).
        - ``skip_mmlu`` (bool):      If True, skips the MMLU download step.
        - ``skip_humaneval`` (bool): If True, skips the HumanEval download step.
    """

    parser = argparse.ArgumentParser(description="Download and curate MMLU and HumanEval.")
    parser.add_argument("--mmlu-n",       type=int, default=200)
    parser.add_argument("--humaneval-n",  type=int, default=50)
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--skip-mmlu",       action="store_true")
    parser.add_argument("--skip-humaneval",  action="store_true")
    return parser.parse_args()


def main() -> None:

    """Orchestrate the dataset build pipeline.

    Parses CLI arguments and conditionally runs :func:`build_mmlu` and/or
    :func:`build_humaneval`, followed by a :func:`_validate` spot-check on
    each generated file.

    This function is registered as the ``build-datasets`` entry point in
    ``pyproject.toml`` and is invoked via ``uv run build-datasets``.
    """
    
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

    print("\n ✓ Datasets ready for use with run-benchmark\n")


if __name__ == "__main__":
    main()