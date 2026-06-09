"""
src/llm_eval/shared/langsmith_tracer.py
========================================
Optional LangSmith integration for tracing LLM calls and benchmark runs.

Provides a zero-overhead tracing layer that activates automatically when
``LANGSMITH_API_KEY`` is present in the environment. When the key is absent,
all functions in this module are no-ops — the benchmark pipeline runs
identically with zero additional latency or error surface.

How it works:
    1. At module import time, ``_ENABLED`` is set based on the presence of
       ``LANGSMITH_API_KEY`` in the environment.
    2. If enabled, a ``langsmith.Client`` is instantiated once and reused
       across all trace calls.
    3. Each LLM call in ``run_benchmark.py`` is wrapped in
       :func:`trace_llm_call`, which yields a ``meta`` dict for the caller
       to populate with outputs (response, scores, latency).
    4. After each full strategy run, :func:`log_run_summary` sends an
       aggregated summary trace.

Setup (``.env``):
    .. code-block:: ini

        LANGSMITH_API_KEY=lsv2_pt_...     # required to enable tracing
        LANGSMITH_PROJECT=llm-eval-suite  # optional, defaults to "llm-eval-suite"
        LANGSMITH_TRACING_V2=true         # optional, enables v2 trace format

Trace structure in LangSmith UI:
    - **Run type ``"llm"``** : one trace per LLM call, with prompt, system,
      model, strategy, response, judge score, BLEU, ROUGE-L, and latency.
    - **Run type ``"chain"``**: one trace per completed strategy run, with
      aggregated metric means (from ``Evaluator.summarize``).

Design principle — never break the benchmark:
    All LangSmith API calls are wrapped in ``try/except Exception: pass``.
    A LangSmith outage, quota limit, or network error must never propagate
    into the benchmark pipeline and corrupt results.

Dependencies:
    langsmith (optional — gracefully disabled if not installed), python-dotenv
"""
from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import Any, Generator

from dotenv import load_dotenv

load_dotenv()

# Resolved once at import time. All functions check this flag before making
# any LangSmith API calls, keeping the no-op path branch-free and fast.
_ENABLED: bool = bool(os.getenv("LANGSMITH_API_KEY"))

if _ENABLED:
    try:
        from langsmith import Client  # type: ignore
        _client = Client()
    except ImportError:
        # langsmith package not installed — disable silently rather than
        # raising an ImportError that would break the benchmark on import
        _ENABLED = False
        _client  = None
else:
    _client = None


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def is_enabled() -> bool:
    """Return whether LangSmith tracing is active for this process.

    Reflects the value of ``_ENABLED`` as set at module import time.
    Used in ``run_benchmark.py`` to conditionally print the LangSmith
    dashboard URL after a run completes.

    Returns:
        ``True`` if ``LANGSMITH_API_KEY`` was present at import time and
        the ``langsmith`` package is installed. ``False`` otherwise.
    """
    return _ENABLED


@contextmanager
def trace_llm_call(
    name: str,
    model: str,
    strategy: str,
    prompt: str,
    system: str = "",
) -> Generator[dict[str, Any], None, None]:
    """Context manager that traces a single LLM call to LangSmith.

    Yields a mutable ``meta`` dict to the caller. The caller populates it
    with outputs (response text, scores, latency) after the LLM call
    completes. When the ``with`` block exits, the accumulated metadata is
    sent to LangSmith as a single ``"llm"`` run.

    If ``LANGSMITH_API_KEY`` is not configured, the context manager yields
    the empty dict and returns immediately — zero overhead, no side effects.

    Args:
        name:     LangSmith run name, used as the display label in the UI.
                  Convention: ``"<strategy>/complete"``
                  (e.g. ``"zero_shot/complete"``).
        model:    Model label string (e.g. ``"llama3.3:70b"``). Stored in
                  both ``inputs`` and ``extra`` for filtering in the UI.
        strategy: Prompt strategy name (e.g. ``"few_shot"``). Stored in
                  ``inputs`` and ``extra`` for cross-strategy comparison.
        prompt:   Full prompt text sent to the model. Stored in ``inputs``.
        system:   System prompt text, if any. Stored in ``inputs``.
                  Defaults to ``""``.

    Yields:
        dict[str, Any]: Mutable metadata dict. The caller should populate
        the following keys after the LLM call completes:

        - ``"output"``      (str):   raw model response text.
        - ``"latency_ms"``  (float): end-to-end call latency in ms.
        - ``"judge_score"`` (int):   LLM-as-judge score (1–5).
        - ``"is_correct"``  (bool):  judge correctness verdict.
        - ``"bleu"``        (float): BLEU score for this item.
        - ``"rougeL"``      (float): ROUGE-L score for this item.

    Example:
        >>> with trace_llm_call(
        ...     name="zero_shot/complete",
        ...     model="llama3.3:70b",
        ...     strategy="zero_shot",
        ...     prompt="Question: ...",
        ... ) as meta:
        ...     resp = client.complete(prompt)
        ...     meta["output"]     = resp.response
        ...     meta["latency_ms"] = resp.latency_ms

    Note:
        LangSmith API errors are silently suppressed. A tracing failure
        must never abort a benchmark run or corrupt its results.
    """
    meta: dict[str, Any] = {}
    start = time.monotonic()

    yield meta  # caller populates meta inside the with block

    if not _ENABLED or _client is None:
        return

    elapsed = (time.monotonic() - start) * 1000
    project = os.getenv("LANGSMITH_PROJECT", "llm-eval-suite")

    try:
        _client.create_run(
            project_name=project,
            name=name,
            run_type="llm",
            inputs={
                "prompt":   prompt,
                "system":   system,
                "model":    model,
                "strategy": strategy,
            },
            outputs={
                "response":    meta.get("output", ""),
                "judge_score": meta.get("judge_score"),
                "is_correct":  meta.get("is_correct"),
                "bleu":        meta.get("bleu"),
                "rougeL":      meta.get("rougeL"),
            },
            extra={
                # Use caller-measured latency when available; fall back to
                # the context manager's own wall-clock measurement
                "latency_ms": meta.get("latency_ms", elapsed),
                "model":      model,
                "strategy":   strategy,
            },
        )
    except Exception:
        pass  # LangSmith must never break the benchmark


def log_run_summary(
    strategy: str,
    model: str,
    summary: dict[str, Any],
) -> None:
    """Send an aggregated benchmark run summary to LangSmith.

    Creates a single ``"chain"`` run in LangSmith representing the completed
    strategy run, with mean metric scores from ``Evaluator.summarize`` as
    outputs. Useful for comparing strategy performance across runs directly
    in the LangSmith UI without exporting CSV files.

    If LangSmith is not enabled, this function returns immediately.

    Args:
        strategy: Prompt strategy name (e.g. ``"chain_of_thought"``).
                  Used as part of the run name
                  (``"benchmark_summary/<strategy>"``).
        model:    Model label string stored in run inputs for filtering.
        summary:  Aggregated metrics dict from ``Evaluator.summarize()``,
                  containing keys such as ``"bleu_mean"``, ``"rougeL_mean"``,
                  ``"judge_score_mean"``, ``"judge_accuracy"``, etc.

    Note:
        LangSmith API errors are silently suppressed. This function is
        called after results have already been saved to disk, so a tracing
        failure has no impact on the benchmark output files.
    """
    if not _ENABLED or _client is None:
        return

    project = os.getenv("LANGSMITH_PROJECT", "llm-eval-suite")

    try:
        _client.create_run(
            project_name=project,
            name=f"benchmark_summary/{strategy}",
            run_type="chain",
            inputs={
                "strategy": strategy,
                "model":    model,
                "n":        summary.get("n"),
            },
            outputs=summary,
        )
    except Exception:
        pass  # LangSmith must never break the benchmark