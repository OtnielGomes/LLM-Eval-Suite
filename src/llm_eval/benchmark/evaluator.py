"""
src/llm_eval/benchmark/evaluator.py
=====================================
Core evaluation engine for the LLM benchmark suite.

Computes three complementary evaluation signals for each (reference, hypothesis)
pair and aggregates them into an ``EvalResult``:

- **BLEU** (Bilingual Evaluation Understudy): n-gram precision with brevity
  penalty. Fast and reference-based, but sensitive to exact wording.
- **ROUGE** (Recall-Oriented Understudy for Gisting Evaluation): unigram,
  bigram, and longest common subsequence F1 scores. Better suited for
  recall-heavy tasks such as summarisation.
- **LLM-as-judge**: delegates qualitative scoring (1-5) to the configured
  LLM backend via the ``LLMJudge`` protocol. Captures semantic correctness
  that n-gram metrics miss.

Backend-agnostic design:
    ``Evaluator`` depends only on the ``LLMJudge`` protocol — any client that
    implements ``judge()`` works without modification::

        evaluator = Evaluator(llm=OllamaClient())    # local dev
        evaluator = Evaluator(llm=GeminiClient())    # production
        evaluator = Evaluator(llm=OllamaCloudClient()) # zero-cost cloud

Metric trade-offs:

.. list-table::
   :header-rows: 1

   * - Metric
     - Measures
     - Strength
     - Weakness
   * - BLEU
     - n-gram precision
     - Fast, deterministic
     - Penalises valid paraphrases
   * - ROUGE-L
     - Longest common subsequence
     - Recall-oriented
     - Ignores semantics
   * - LLM-as-judge
     - Semantic correctness
     - Captures meaning
     - Slower, non-deterministic

Dependencies:
    nltk >= 3.8, rouge-score >= 0.1, llm_eval.shared.types
"""
from __future__ import annotations

import nltk
nltk.download("punkt_tab", quiet=True)

from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from typing import Protocol, runtime_checkable

from llm_eval.shared.types import EvalResult, JudgeScore


# ---------------------------------------------------------------------------
# LLM judge protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class LLMJudge(Protocol):
    """Structural interface for any LLM client used as an evaluation judge.

    Any class that implements :meth:`judge` satisfies this protocol —
    inheritance is not required. All three clients in the suite
    (``OllamaClient``, ``OllamaCloudClient``, ``GeminiClient``) conform to
    it, making them interchangeable as the ``llm`` argument of ``Evaluator``.

    The ``@runtime_checkable`` decorator allows ``isinstance(obj, LLMJudge)``
    checks in tests and the ``Evaluator.__init__`` type guard.
    """

    def judge(self, question: str, reference: str, hypothesis: str) -> JudgeScore:
        """Evaluate a hypothesis against a reference and return a structured score.

        Args:
            question:   The original question posed to the model under evaluation.
            reference:  The ground-truth / expected answer.
            hypothesis: The model-generated answer to be evaluated.

        Returns:
            ``JudgeScore`` with a numeric score (1-5), textual justification,
            and a boolean correctness verdict.
        """
        ...


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class Evaluator:
    """Computes BLEU, ROUGE, and LLM-as-judge scores for response pairs.

    Combines three evaluation signals into a single ``EvalResult`` per item:
    two fast lexical metrics (BLEU, ROUGE) and one semantic judge call.

    The LLM backend is injected at construction time and accessed exclusively
    through the ``LLMJudge`` protocol, keeping the evaluator decoupled from
    any specific client implementation.

    Attributes:
        _llm:      LLM client used for judge scoring.
        _rouge:    ``RougeScorer`` instance configured for rouge1, rouge2, rougeL.
        _smoother: NLTK smoothing function applied to BLEU to handle zero n-gram
                   counts in short hypotheses.

    Example:
        >>> from llm_eval.clients.ollama_cliente import OllamaClient
        >>> evaluator = Evaluator(llm=OllamaClient())
        >>> result = evaluator.evaluate_all(
        ...     question="What is RAG?",
        ...     reference="RAG combines retrieval with generation.",
        ...     hypothesis="RAG uses vector search to improve LLM responses.",
        ...     strategy="zero_shot",
        ...     model="llama3.1:8b",
        ...     latency_ms=312.5,
        ... )
        >>> result.judge_score  # 1-5
        4
    """

    def __init__(self, llm: LLMJudge):
        """Initialise the evaluator with an LLM judge backend.

        Args:
            llm: Any object satisfying the ``LLMJudge`` protocol. Typically
                 one of ``OllamaClient``, ``OllamaCloudClient``, or
                 ``GeminiClient``.
        """
        self._llm = llm
        # Compute rouge1, rouge2, and rougeL F1 scores with Porter stemming
        # to reduce sensitivity to morphological variation (e.g. "run"/"running")
        self._rouge = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )
        # SmoothingFunction.method1 adds epsilon counts to zero n-gram matches,
        # preventing BLEU from collapsing to 0.0 on very short hypotheses
        self._smoother = SmoothingFunction().method1

    # ---- Lexical metrics ------------------------------------------------ #

    def bleu(self, reference: str, hypothesis: str) -> float:
        """Compute the sentence-level BLEU score for a single hypothesis.

        Uses NLTK's ``sentence_bleu`` with ``SmoothingFunction.method1`` to
        handle zero n-gram counts that arise with short or mismatched responses.
        Both strings are lowercased and whitespace-tokenised before scoring.

        Args:
            reference:  Ground-truth answer string.
            hypothesis: Model-generated answer string.

        Returns:
            BLEU score in ``[0.0, 1.0]``. Returns ``0.0`` immediately for
            empty or whitespace-only hypotheses, bypassing NLTK to avoid
            a ``ZeroDivisionError``.

        Note:
            BLEU is a precision-oriented metric — it measures the fraction of
            n-grams in the hypothesis that appear in the reference. A short
            but accurate hypothesis can score lower than a longer paraphrase
            due to the brevity penalty.
        """
        if not hypothesis.strip():
            return 0.0
        score: float = sentence_bleu(  # type: ignore[assignment]
            [reference.lower().split()],
            hypothesis.lower().split(),
            smoothing_function=self._smoother,
        )
        return score

    def rouge(self, reference: str, hypothesis: str) -> dict[str, float]:
        """Compute ROUGE-1, ROUGE-2, and ROUGE-L F1 scores.

        Uses the ``rouge_score`` library with Porter stemming enabled.
        All three variants are computed in a single pass for efficiency.

        Args:
            reference:  Ground-truth answer string.
            hypothesis: Model-generated answer string.

        Returns:
            Dict with keys ``"rouge1"``, ``"rouge2"``, ``"rougeL"``, each
            containing the F1 score rounded to 4 decimal places.
            Returns all zeros immediately for empty or whitespace-only
            hypotheses.

        Note:
            - ``rouge1``: unigram overlap — sensitive to vocabulary match.
            - ``rouge2``: bigram overlap — captures local word order.
            - ``rougeL``: longest common subsequence — captures sentence-level
              structure without requiring contiguous matches.
        """
        if not hypothesis.strip():
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        scores = self._rouge.score(reference, hypothesis)
        return {
            "rouge1": round(scores["rouge1"].fmeasure, 4),
            "rouge2": round(scores["rouge2"].fmeasure, 4),
            "rougeL": round(scores["rougeL"].fmeasure, 4),
        }

    # ---- LLM judge ------------------------------------------------------ #

    def llm_judge(self, question: str, reference: str, hypothesis: str) -> JudgeScore:
        """Delegate qualitative evaluation to the configured LLM backend.

        A thin pass-through to ``self._llm.judge()`` that keeps ``evaluate_all``
        readable and allows the judge call to be mocked independently in tests.

        Args:
            question:   The original question posed to the evaluated model.
            reference:  The ground-truth / expected answer.
            hypothesis: The model-generated answer to be evaluated.

        Returns:
            ``JudgeScore`` from the configured LLM backend.
        """
        return self._llm.judge(
            question=question,
            reference=reference,
            hypothesis=hypothesis,
        )

    # ---- Combined evaluation -------------------------------------------- #

    def evaluate_all(
        self,
        question: str,
        reference: str,
        hypothesis: str,
        strategy: str,
        model: str,
        latency_ms: float,
    ) -> EvalResult:
        """Run all three evaluation signals and return a combined ``EvalResult``.

        Computes ROUGE and BLEU scores locally (no network call), then calls
        the LLM judge (one network call). The results are assembled into a
        single ``EvalResult`` that captures the full evaluation context
        including metadata (strategy, model, latency).

        Args:
            question:   The original benchmark question.
            reference:  The ground-truth answer from the dataset.
            hypothesis: The model-generated answer to evaluate.
            strategy:   Prompt strategy name (e.g. ``"zero_shot"``).
                        Stored in the result for downstream grouping.
            model:      Model label string (e.g. ``"llama3.1:8b"``).
                        Stored in the result for downstream grouping.
            latency_ms: End-to-end generation latency in milliseconds,
                        as measured by the calling client.

        Returns:
            ``EvalResult`` with all metric fields populated and metadata
            attached. Numeric fields are rounded for consistent serialisation:
            ``bleu`` and ROUGE scores to 4 decimal places, ``latency_ms``
            to 1 decimal place.
        """
        rouge_scores = self.rouge(reference, hypothesis)
        judge: JudgeScore = self.llm_judge(question, reference, hypothesis)
        return EvalResult(
            question=question,
            expected=reference,
            predicted=hypothesis,
            strategy=strategy,
            model=model,
            latency_ms=round(latency_ms, 1),
            bleu=round(self.bleu(reference, hypothesis), 4),
            rouge1=rouge_scores["rouge1"],
            rouge2=rouge_scores["rouge2"],
            rougeL=rouge_scores["rougeL"],
            judge_score=judge.score,
            judge_reason=judge.reason,
            judge_correct=judge.is_correct,
        )

    # ---- Aggregation ---------------------------------------------------- #

    @staticmethod
    def summarize(results: list[EvalResult]) -> dict[str, float]:
        """Aggregate a list of ``EvalResult`` objects into mean metric scores.

        Computes arithmetic means for all numeric fields. Intended to be called
        once after a full benchmark run to produce the summary printed to
        stdout and logged to LangSmith in ``run_benchmark.py``.

        Args:
            results: List of ``EvalResult`` objects from a completed benchmark
                     run. May contain results from a single strategy or
                     mixed strategies.

        Returns:
            Dict with the following keys:

            - ``"n"``                 (int):   number of evaluated items.
            - ``"bleu_mean"``         (float): mean BLEU score, 4 d.p.
            - ``"rouge1_mean"``       (float): mean ROUGE-1 F1, 4 d.p.
            - ``"rouge2_mean"``       (float): mean ROUGE-2 F1, 4 d.p.
            - ``"rougeL_mean"``       (float): mean ROUGE-L F1, 4 d.p.
            - ``"judge_score_mean"``  (float): mean judge score (1–5), 2 d.p.
            - ``"judge_accuracy"``    (float): fraction of ``is_correct=True``, 4 d.p.
            - ``"latency_ms_mean"``   (float): mean latency in ms, 1 d.p.

            Returns an empty dict ``{}`` when ``results`` is empty, allowing
            callers to handle the no-data case without a ``ZeroDivisionError``.
        """
        n = len(results)
        if n == 0:
            return {}
        return {
            "n":                 n,
            "bleu_mean":         round(sum(r.bleu          for r in results) / n, 4),
            "rouge1_mean":       round(sum(r.rouge1        for r in results) / n, 4),
            "rouge2_mean":       round(sum(r.rouge2        for r in results) / n, 4),
            "rougeL_mean":       round(sum(r.rougeL        for r in results) / n, 4),
            "judge_score_mean":  round(sum(r.judge_score   for r in results) / n, 2),
            "judge_accuracy":    round(sum(r.judge_correct for r in results) / n, 4),
            "latency_ms_mean":   round(sum(r.latency_ms    for r in results) / n, 1),
        }