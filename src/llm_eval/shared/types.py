"""
src/llm_eval/shared/types.py
=============================
Canonical data types shared across all modules in llm-eval-suite.

Defining all shared types in a single module eliminates the duplication that
would otherwise occur between ``gemini_cliente``, ``ollama_cliente``, and
``ollama_cloud_cliente``, and provides a single source of truth for the
data contracts used throughout the pipeline.

Type inventory:

.. list-table::
   :header-rows: 1

   * - Class
     - Kind
     - Used by
   * - ``LLMResponse``
     - ``@dataclass``
     - All three LLM clients — return type of ``client.complete()``
   * - ``JudgeScore``
     - Pydantic ``BaseModel``
     - All three LLM clients — return type of ``client.judge()``; also used
       as ``response_schema`` in the Gemini SDK for native structured output
   * - ``EvalResult``
     - ``@dataclass``
     - ``Evaluator.evaluate_all()``, ``run_benchmark.py``
   * - ``RAGResult``
     - ``@dataclass``
     - ``RAGPipeline.query()``, ``evaluate_rag.py``

Design note — dataclass vs Pydantic:
    ``LLMResponse``, ``EvalResult``, and ``RAGResult`` use ``@dataclass``
    because they are plain data containers that require no validation.
    ``JudgeScore`` uses Pydantic ``BaseModel`` because it is also passed
    directly as ``response_schema=`` to the Gemini SDK, which requires a
    Pydantic model for native structured output. The Ollama clients parse
    JSON manually and then instantiate ``JudgeScore(**data)`` to reuse the
    same type.
"""
from dataclasses import dataclass, field
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# LLM response
# ---------------------------------------------------------------------------


@dataclass
class LLMResponse:
    """Generic response container returned by every LLM client's ``complete()``.

    Provides a unified return type so that ``run_benchmark.py`` and the
    ``Evaluator`` can consume responses from ``OllamaClient``,
    ``OllamaCloudClient``, and ``GeminiClient`` interchangeably.

    Attributes:
        model:         Identifier of the model that generated the response
                       (e.g. ``"llama3.1:8b"`` or ``"gemini-2.0-flash"``).
        prompt:        The user prompt sent to the model. Stored for
                       traceability in LangSmith and result files.
        response:      The generated text returned by the model.
        latency_ms:    End-to-end call latency in milliseconds, measured
                       by the client from request start to response receipt.
        input_tokens:  Number of input tokens consumed, if reported by the
                       API. Defaults to ``0`` for clients that do not expose
                       token counts (Ollama local and cloud).
        output_tokens: Number of output tokens generated, if reported by the
                       API. Defaults to ``0`` for the same reason.
    """
    model: str
    prompt: str
    response: str
    latency_ms: float
    input_tokens: int  = 0
    output_tokens: int = 0


# ---------------------------------------------------------------------------
# Judge score
# ---------------------------------------------------------------------------


class JudgeScore(BaseModel):
    """Structured output schema for LLM-as-judge evaluation responses.

    Pydantic ``BaseModel`` is used instead of ``@dataclass`` for two reasons:

    1. **Gemini structured output**: passed directly as ``response_schema=``
       to the Gemini SDK, which requires a Pydantic model to enforce the
       response format at the API level.
    2. **Manual JSON parsing**: Ollama clients parse the JSON response
       manually and instantiate this model via ``JudgeScore(**data)``,
       benefiting from Pydantic's field-level type coercion and validation.

    Attributes:
        score:      Numeric quality score on a 1–5 rubric, where:
                    5 = fully correct and complete,
                    4 = mostly correct with minor omissions,
                    3 = partially correct with key gaps,
                    2 = mostly wrong with some relevant content,
                    1 = completely wrong or off-topic.
        reason:     Textual justification for the assigned score. Used for
                    qualitative analysis and stored in ``EvalResult.judge_reason``.
        is_correct: Boolean correctness verdict. Aggregated into
                    ``judge_accuracy`` by ``Evaluator.summarize()``.
    """
    score: int        # 1–5 quality rubric (see JUDGE_PROMPT_TEMPLATE in config.py)
    reason: str
    is_correct: bool


# ---------------------------------------------------------------------------
# Benchmark evaluation result
# ---------------------------------------------------------------------------


@dataclass
class EvalResult:
    """Complete evaluation record for a single benchmark item.

    Produced by ``Evaluator.evaluate_all()`` and collected into a list by
    ``run_benchmark.py``. Serialised to JSON and CSV via ``save_results()``.

    The dataclass combines three categories of fields:
    - **Identity**: question, expected answer, predicted answer, strategy, model.
    - **Lexical metrics**: BLEU, ROUGE-1, ROUGE-2, ROUGE-L (computed locally).
    - **Semantic metrics**: LLM-as-judge score, reason, and correctness verdict.

    Attributes:
        question:      The original benchmark question.
        expected:      Ground-truth answer from the dataset.
        predicted:     Model-generated answer being evaluated.
        strategy:      Prompt strategy used (e.g. ``"zero_shot"``).
        model:         Model label string (e.g. ``"llama3.3:70b"``).
        latency_ms:    End-to-end LLM call latency in milliseconds.
        bleu:          Sentence BLEU score in ``[0.0, 1.0]``. Defaults to ``0.0``.
        rouge1:        ROUGE-1 F1 score in ``[0.0, 1.0]``. Defaults to ``0.0``.
        rouge2:        ROUGE-2 F1 score in ``[0.0, 1.0]``. Defaults to ``0.0``.
        rougeL:        ROUGE-L F1 score in ``[0.0, 1.0]``. Defaults to ``0.0``.
        judge_score:   LLM-as-judge score (1–5). Defaults to ``0`` (unevaluated).
        judge_reason:  Textual justification from the judge. Defaults to ``""``.
        judge_correct: Boolean correctness verdict from the judge.
                       Defaults to ``False``.
    """
    question:      str
    expected:      str
    predicted:     str
    strategy:      str
    model:         str
    latency_ms:    float
    bleu:          float = 0.0
    rouge1:        float = 0.0
    rouge2:        float = 0.0
    rougeL:        float = 0.0
    judge_score:   int   = 0
    judge_reason:  str   = ""
    judge_correct: bool  = False

    def to_dict(self) -> dict:
        """Serialise the result to a plain dict for JSON and CSV output.

        Returns a shallow copy of ``__dict__`` so that the original instance
        is not mutated by downstream code that modifies the returned dict.

        Returns:
            Dict mapping field names to their current values. Field order
            follows dataclass definition order, which determines the column
            order in CSV output.
        """
        return self.__dict__.copy()


# ---------------------------------------------------------------------------
# RAG query result
# ---------------------------------------------------------------------------


@dataclass
class RAGResult:
    """Result of a single RAG pipeline query.

    Produced by ``RAGPipeline.query()`` and consumed by ``evaluate_rag.py``,
    which assembles a list of these objects into a Hugging Face ``Dataset``
    for RAGAS evaluation.

    The four RAGAS-required fields are ``question``, ``answer``,
    ``contexts``, and ``ground_truth``. The additional ``strategy`` and
    ``latency_ms`` fields provide metadata for result grouping and
    performance analysis.

    Attributes:
        question:     The user question passed to the pipeline.
        answer:       The generated answer string from the LLM.
        contexts:     List of retrieved document chunk strings used to
                      generate the answer. Corresponds to the ``contexts``
                      field expected by RAGAS metrics. Defaults to ``[]``.
        ground_truth: Expected answer for RAGAS evaluation. Empty string
                      when ground truth is unavailable (e.g. interactive
                      use). Defaults to ``""``.
        strategy:     Active retrieval strategy name (``"naive"``,
                      ``"hyde"``, or ``"reranking"``). Used for downstream
                      grouping in evaluation reports. Defaults to
                      ``"naive"``.
        latency_ms:   End-to-end pipeline latency in milliseconds,
                      from question input to answer output. Defaults to
                      ``0.0``.

    Note:
        ``contexts`` uses ``field(default_factory=list)`` rather than
        ``contexts: list[str] = []`` to avoid the mutable default argument
        pitfall — all instances would otherwise share the same list object.
    """
    question:     str
    answer:       str
    contexts:     list[str] = field(default_factory=list)
    ground_truth: str       = ""
    strategy:     str       = "naive"
    latency_ms:   float     = 0.0