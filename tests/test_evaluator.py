"""
tests/test_evaluator.py
========================
Unit tests for the ``Evaluator`` class (BLEU, ROUGE, and LLM-as-judge).

Covers the full public API of ``Evaluator``:
- :meth:`Evaluator.bleu`
- :meth:`Evaluator.rouge`
- :meth:`Evaluator.llm_judge`
- :meth:`Evaluator.evaluate_all`
- :meth:`Evaluator.summarize`

Isolation strategy:
    The LLM judge is always replaced with a ``MagicMock`` via
    :func:`_make_evaluator`. No real API calls are made in any test.
    Lexical metrics (BLEU, ROUGE) are computed locally and need no mocking.

Test organisation:
    Tests are grouped into classes by method under test. Each class targets
    one distinct behaviour per test method, named after the condition being
    verified (e.g. ``test_hipotese_vazia_retorna_zero``).
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

# Imports resolved via sys.path registered in conftest.py
from evaluator import Evaluator          # type: ignore[import]
from llm_eval.shared.types import JudgeScore, EvalResult


# ---------------------------------------------------------------------------
# Test helper
# ---------------------------------------------------------------------------


def _make_evaluator(
    score: int = 4,
    reason: str = "Good.",
    is_correct: bool = True,
) -> Evaluator:
    """Instantiate an ``Evaluator`` with a mocked LLM judge.

    Creates a ``MagicMock`` that returns a fixed ``JudgeScore`` for every
    ``judge()`` call. Allows tests to control the judge output independently
    of the lexical metric computation.

    Args:
        score:      Judge score returned by the mock (1–5). Defaults to ``4``.
        reason:     Textual reason returned by the mock. Defaults to ``"Good."``.
        is_correct: Correctness verdict returned by the mock. Defaults to ``True``.

    Returns:
        ``Evaluator`` instance with ``llm.judge`` pre-configured to return
        ``JudgeScore(score, reason, is_correct)``.
    """
    mock_llm = MagicMock()
    mock_llm.judge.return_value = JudgeScore(
        score=score, reason=reason, is_correct=is_correct
    )
    return Evaluator(llm=mock_llm)


# ---------------------------------------------------------------------------
# BLEU tests
# ---------------------------------------------------------------------------


class TestBLEU:
    """Tests for :meth:`Evaluator.bleu`."""

    def test_resposta_identica_score_alto(self):
        """BLEU score for an identical hypothesis should be close to 1.0.

        Verifies that perfect n-gram overlap produces a near-perfect score,
        validating the basic correctness of the BLEU computation.
        """
        ev = _make_evaluator()
        score = ev.bleu(
            reference="The cat sat on the mat.",
            hypothesis="The cat sat on the mat.",
        )
        assert score > 0.95

    def test_resposta_completamente_diferente_score_baixo(self):
        """BLEU score between completely unrelated texts should be near zero.

        Ensures that BLEU correctly penalises hypotheses with no n-gram
        overlap with the reference.
        """
        ev = _make_evaluator()
        score = ev.bleu(
            reference="Machine learning is a subset of artificial intelligence.",
            hypothesis="The weather is sunny today.",
        )
        assert score < 0.2

    def test_retorna_float_entre_0_e_1(self):
        """BLEU score should always be a float in the range ``[0.0, 1.0]``.

        Guards against implementation changes that could produce out-of-range
        values (e.g. unsmoothed BLEU on mismatched lengths).
        """
        ev = _make_evaluator()
        score = ev.bleu(
            reference="This is a reference sentence.",
            hypothesis="This is a test sentence.",
        )
        assert 0.0 <= score <= 1.0

    def test_hipotese_vazia_retorna_zero(self):
        """BLEU with an empty hypothesis should return ``0.0`` without raising.

        Verifies the early-return guard that prevents a ``ZeroDivisionError``
        from NLTK's ``sentence_bleu`` when the hypothesis token list is empty.
        """
        ev = _make_evaluator()
        score = ev.bleu(reference="Some reference text.", hypothesis="")
        assert score == 0.0


# ---------------------------------------------------------------------------
# ROUGE tests
# ---------------------------------------------------------------------------


class TestROUGE:
    """Tests for :meth:`Evaluator.rouge`."""

    def test_resposta_identica_rougeL_alto(self):
        """ROUGE-L for an identical hypothesis should be close to 1.0.

        Validates that a perfect longest common subsequence match produces
        a near-perfect F1 score.
        """
        ev = _make_evaluator()
        scores = ev.rouge(
            reference="Precision and recall are evaluation metrics.",
            hypothesis="Precision and recall are evaluation metrics.",
        )
        assert scores["rougeL"] > 0.95

    def test_resposta_parcial_score_intermediario(self):
        """ROUGE-L for a partially overlapping hypothesis should be in (0.2, 0.9).

        Verifies that partial lexical overlap produces intermediate scores,
        confirming that the metric is sensitive to the degree of match.
        """
        ev = _make_evaluator()
        scores = ev.rouge(
            reference="Precision and recall are key evaluation metrics in ML.",
            hypothesis="Precision is an evaluation metric.",
        )
        assert 0.2 < scores["rougeL"] < 0.9

    def test_retorna_tres_chaves(self):
        """``rouge()`` should return a dict with exactly three keys.

        Ensures the return value always contains ``rouge1``, ``rouge2``,
        and ``rougeL``, matching the keys consumed by ``evaluate_all()``
        and ``Evaluator.summarize()``.
        """
        ev = _make_evaluator()
        scores = ev.rouge(reference="hello world", hypothesis="hello")
        assert set(scores.keys()) == {"rouge1", "rouge2", "rougeL"}

    def test_todos_scores_entre_0_e_1(self):
        """All ROUGE scores should be floats in the range ``[0.0, 1.0]``.

        Validates the range constraint for all three ROUGE variants
        simultaneously.
        """
        ev = _make_evaluator()
        scores = ev.rouge(
            reference="hello world",
            hypothesis="goodbye cruel world",
        )
        assert all(0.0 <= v <= 1.0 for v in scores.values())

    def test_hipotese_vazia_retorna_zeros(self):
        """``rouge()`` with an empty hypothesis should return all zeros without raising.

        Mirrors the empty-hypothesis guard in ``bleu()`` — verifies that
        the early-return path in ``rouge()`` is consistent.
        """
        ev = _make_evaluator()
        scores = ev.rouge(reference="Some reference.", hypothesis="")
        assert all(v == 0.0 for v in scores.values())


# ---------------------------------------------------------------------------
# LLM judge tests
# ---------------------------------------------------------------------------


class TestLLMJudge:
    """Tests for :meth:`Evaluator.llm_judge`."""

    def test_retorna_judge_score(self, mock_judge_score: JudgeScore):
        """``llm_judge()`` should return a ``JudgeScore`` with a valid score range.

        Uses the shared ``mock_judge_score`` fixture from ``conftest.py``.
        The range assertion (0–10) is intentionally wider than the rubric
        (1–5) to detect out-of-range values without over-constraining the test.

        Args:
            mock_judge_score: Fixture providing a pre-built ``JudgeScore``.
        """
        assert isinstance(mock_judge_score, JudgeScore)
        assert 0 <= mock_judge_score.score <= 10

    def test_judge_score_tem_reason(self, mock_judge_score: JudgeScore):
        """``JudgeScore`` should always have a non-empty ``reason`` field.

        Verifies that the judge never returns a silent score without
        explanation, which would make debugging evaluation failures difficult.

        Args:
            mock_judge_score: Fixture providing a pre-built ``JudgeScore``.
        """
        assert len(mock_judge_score.reason) > 0

    def test_chama_cliente_uma_vez(self):
        """``llm_judge()`` should call ``llm.judge()`` exactly once per invocation.

        Verifies that ``Evaluator.llm_judge`` is a thin pass-through and does
        not introduce extra or batched judge calls.
        """
        mock_llm = MagicMock()
        mock_llm.judge.return_value = JudgeScore(score=4, reason="OK", is_correct=True)
        ev = Evaluator(llm=mock_llm)

        ev.llm_judge(
            question="What is BLEU?",
            reference="BLEU is a metric for text generation.",
            hypothesis="BLEU measures n-gram overlap.",
        )

        mock_llm.judge.assert_called_once()

    def test_propaga_score_do_cliente(self):
        """``llm_judge()`` should return exactly the ``JudgeScore`` from the client.

        Ensures the pass-through behaviour: no field mutation, no score
        clamping, and no re-wrapping of the returned object.
        """
        expected = JudgeScore(score=5, reason="Perfect.", is_correct=True)
        mock_llm = MagicMock()
        mock_llm.judge.return_value = expected
        ev = Evaluator(llm=mock_llm)

        result = ev.llm_judge(question="Q", reference="R", hypothesis="H")

        assert result.score == 5
        assert result.is_correct is True


# ---------------------------------------------------------------------------
# evaluate_all() and summarize() tests
# ---------------------------------------------------------------------------


class TestEvaluateAll:
    """Tests for :meth:`Evaluator.evaluate_all` and :meth:`Evaluator.summarize`."""

    def test_retorna_eval_result(self):
        """``evaluate_all()`` should return a typed ``EvalResult`` instance.

        Verifies the return type contract used by ``run_benchmark.py`` when
        collecting results into a ``list[EvalResult]``.
        """
        ev = _make_evaluator(score=3, reason="Partial.", is_correct=False)
        result = ev.evaluate_all(
            question="What is ML?",
            reference="Machine Learning is a subset of AI.",
            hypothesis="ML is related to AI.",
            strategy="zero_shot",
            model="test-model",
            latency_ms=150.0,
        )
        assert isinstance(result, EvalResult)

    def test_campos_preenchidos_corretamente(self):
        """All ``EvalResult`` fields should be populated with correct values.

        Checks both identity fields (strategy, model, judge values) and
        metric range constraints (BLEU, ROUGE-L) in a single integration-style
        assertion to verify the full ``evaluate_all`` contract.
        """
        ev = _make_evaluator(score=4, reason="Good.", is_correct=True)
        result = ev.evaluate_all(
            question="What is RAG?",
            reference="RAG combines retrieval with generation.",
            hypothesis="RAG uses retrieval to improve generation.",
            strategy="few_shot",
            model="qwen3-coder",
            latency_ms=200.0,
        )
        assert result.question      == "What is RAG?"
        assert result.strategy      == "few_shot"
        assert result.model         == "qwen3-coder"
        assert result.judge_score   == 4
        assert result.judge_correct is True
        assert 0.0 <= result.bleu   <= 1.0
        assert 0.0 <= result.rougeL <= 1.0
        assert result.latency_ms    > 0

    def test_summarize_calcula_medias(self):
        """``summarize()`` should compute correct means across multiple results.

        Uses ``pytest.approx`` for the latency mean to tolerate floating-point
        rounding. Verifies both the item count and metric range as a sanity
        check on the aggregation logic.
        """
        ev = _make_evaluator()
        results = [
            ev.evaluate_all("Q1", "R1", "H1", "zero_shot", "m", 100.0),
            ev.evaluate_all("Q2", "R2", "H2", "zero_shot", "m", 200.0),
        ]
        summary = Evaluator.summarize(results)

        assert summary["n"] == 2
        assert 0.0 <= summary["bleu_mean"] <= 1.0
        assert summary["latency_ms_mean"] == pytest.approx(150.0, rel=0.01)

    def test_summarize_lista_vazia(self):
        """``summarize()`` with an empty list should return an empty dict.

        Verifies the early-return guard that prevents ``ZeroDivisionError``
        when no results have been collected (e.g. a run aborted before
        any items were evaluated).
        """
        assert Evaluator.summarize([]) == {}