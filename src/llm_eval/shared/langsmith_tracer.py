"""
Optional integration with LangSmith for tracing each LLM call.

How it works:

- If LANGSMITH_API_KEY is in the .env file -> traces automatically
- If not -> silent, zero impact on the benchmark

Configure in the .env file:
LANGSMITH_API_KEY=lsv2_pt_...
LANGSMITH_PROJECT=llm-eval-suite
LANGSMITH_TRACING_V2=true
"""
from __future__ import annotations


import os
import time
from contextlib import contextmanager
from typing import Any, Generator


from dotenv import load_dotenv


load_dotenv()


_ENABLED = bool(os.getenv("LANGSMITH_API_KEY"))


if _ENABLED:
    try:
        from langsmith import Client 
        _client = Client()
    except ImportError:
        _ENABLED = False
        _client = None
else:
    _client = None



def is_enabled() -> bool:
    return _ENABLED



@contextmanager
def trace_llm_call(
    name: str,
    model: str,
    strategy: str,
    prompt: str,
    system: str = "",
) -> Generator[dict[str, Any], None, None]:
    
    meta: dict[str, Any] = {}
    start = time.monotonic()
    yield meta


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
                "prompt": prompt,
                "system": system,
                "model": model,
                "strategy": strategy,
            },
            outputs={
                "response": meta.get("output", ""),
                "judge_score": meta.get("judge_score"),
                "is_correct": meta.get("is_correct"),
                "bleu": meta.get("bleu"),
                "rougeL": meta.get("rougeL"),
            },
            extra={
                "latency_ms": meta.get("latency_ms", elapsed),
                "model": model,
                "strategy": strategy,
            },
        )
    except Exception:
        pass  



def log_run_summary(
    strategy: str,
    model: str,
    summary: dict[str, Any],
) -> None:
    
    if not _ENABLED or _client is None:
        return


    project = os.getenv("LANGSMITH_PROJECT", "llm-eval-suite")


    try:
        _client.create_run(
            project_name=project,
            name=f"benchmark_summary/{strategy}",
            run_type="chain",
            inputs={"strategy": strategy, "model": model, "n": summary.get("n")},
            outputs=summary,
        )
    except Exception:
        pass