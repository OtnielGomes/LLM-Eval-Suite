"""
tests/test_evaluator.py

Tests for the Evaluator class (BLEU, ROUGE, LLM-as-judge).
Real API: Evaluator(llm=...) with methods .bleu(), .rouge(), .llm_judge(), .evaluate_all()
The LLM-judge is always mocked — no real API calls are made.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from evaluator import Evaluator          # type: ignore[import]
from llm_eval.shared.types import JudgeScore, EvalResult


# ---------------------------------------------------------------------------
# Helper — creates Evaluator with mocked judge
# ---------------------------------------------------------------------------

def _make_evaluator(score: int = 4, reason: str = "Good.", is_correct: bool = True) -> Evaluator:
    """Instantiates Evaluator with a mocked LLM client to avoid real API calls."""
    mock_llm = MagicMock()
    mock_llm.judge.return_value = JudgeScore(
        score=score, reason=reason, is_correct=is_correct
    )
    return Evaluator(llm=mock_llm)


# ---------------------------------------------------------------------------
# Tests — bleu()
# ---------------------------------------------------------------------------

class TestBLEU:

    def test_identical_response_high_score(self):
        """BLEU score for an identical hypothesis should be close to 1.0."""
        ev = _make_evaluator()
        score = ev.bleu(
            reference="The cat sat on the mat.",
            hypothesis="The cat sat on the mat."
        )
        assert score > 0.95

    def test_completely_different_response_low_score(self):
        """BLEU score between completely different texts should be close to 0."""
        ev = _make_evaluator()
        score = ev.bleu(
            reference="Machine learning is a subset of artificial intelligence.",
            hypothesis="The weather is sunny today."
        )
        assert score < 0.2

    def test_returns_float_between_0_and_1(self):
        """BLEU should always return a float in the range [0, 1]."""
        ev = _make_evaluator()
        score = ev.bleu(
            reference="This is a reference sentence.",
            hypothesis="This is a test sentence."
        )
        assert 0.0 <= score <= 1.0

    def test_empty_hypothesis_returns_zero(self):
        """BLEU with an empty hypothesis should return 0.0 without raising."""
        ev = _make_evaluator()
        score = ev.bleu(reference="Some reference text.", hypothesis="")
        assert score == 0.0


# ---------------------------------------------------------------------------
# Tests — rouge()
# ---------------------------------------------------------------------------

class TestROUGE:

    def test_identical_response_high_rougeL(self):
        """ROUGE-L for an identical hypothesis should be close to 1.0."""
        ev = _make_evaluator()
        scores = ev.rouge(
            reference="Precision and recall are evaluation metrics.",
            hypothesis="Precision and recall are evaluation metrics."
        )
        assert scores["rougeL"] > 0.95

    def test_partial_overlap_intermediate_score(self):
        """ROUGE-L with partial overlap should be between 0.2 and 0.9."""
        ev = _make_evaluator()
        scores = ev.rouge(
            reference="Precision and recall are key evaluation metrics in ML.",
            hypothesis="Precision is an evaluation metric."
        )
        assert 0.2 < scores["rougeL"] < 0.9

    def test_returns_three_keys(self):
        """rouge() should return a dict with rouge1, rouge2, and rougeL."""
        ev = _make_evaluator()
        scores = ev.rouge(reference="hello world", hypothesis="hello")
        assert set(scores.keys()) == {"rouge1", "rouge2", "rougeL"}

    def test_all_scores_between_0_and_1(self):
        """All ROUGE scores should be in the range [0, 1]."""
        ev = _make_evaluator()
        scores = ev.rouge(reference="hello world", hypothesis="goodbye cruel world")
        assert all(0.0 <= v <= 1.0 for v in scores.values())

    def test_empty_hypothesis_returns_zeros(self):
        """rouge() with an empty hypothesis should return zeros without raising."""
        ev = _make_evaluator()
        scores = ev.rouge(reference="Some reference.", hypothesis="")
        assert all(v == 0.0 for v in scores.values())


# ---------------------------------------------------------------------------
# Tests — llm_judge()
# ---------------------------------------------------------------------------

class TestLLMJudge:

    def test_returns_judge_score(self, mock_judge_score):
        """llm_judge() should return a JudgeScore with score between 0 and 10."""
        assert isinstance(mock_judge_score, JudgeScore)
        assert 0 <= mock_judge_score.score <= 10

    def test_judge_score_has_reason(self, mock_judge_score):
        """JudgeScore should contain a non-empty reason field."""
        assert len(mock_judge_score.reason) > 0

    def test_calls_client_once(self):
        """llm_judge() should call llm.judge() exactly once."""
        mock_llm = MagicMock()
        mock_llm.judge.return_value = JudgeScore(score=4, reason="OK", is_correct=True)
        ev = Evaluator(llm=mock_llm)

        ev.llm_judge(
            question="What is BLEU?",
            reference="BLEU is a metric for text generation.",
            hypothesis="BLEU measures n-gram overlap.",
        )

        mock_llm.judge.assert_called_once()

    def test_propagates_client_score(self):
        """llm_judge() should return exactly the JudgeScore provided by the client."""
        expected = JudgeScore(score=5, reason="Perfect.", is_correct=True)
        mock_llm = MagicMock()
        mock_llm.judge.return_value = expected
        ev = Evaluator(llm=mock_llm)

        result = ev.llm_judge(question="Q", reference="R", hypothesis="H")

        assert result.score == 5
        assert result.is_correct is True


# ---------------------------------------------------------------------------
# Tests — evaluate_all()
# ---------------------------------------------------------------------------

class TestEvaluateAll:

    def test_returns_eval_result(self):
        """evaluate_all() should return a typed EvalResult."""
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

    def test_all_fields_populated(self):
        """All EvalResult fields should be correctly populated."""
        ev = _make_evaluator(score=4, reason="Good.", is_correct=True)
        result = ev.evaluate_all(
            question="What is RAG?",
            reference="RAG combines retrieval with generation.",
            hypothesis="RAG uses retrieval to improve generation.",
            strategy="few_shot",
            model="qwen3-coder",
            latency_ms=200.0,
        )
        assert result.question == "What is RAG?"
        assert result.strategy == "few_shot"
        assert result.model == "qwen3-coder"
        assert result.judge_score == 4
        assert result.judge_correct is True
        assert 0.0 <= result.bleu <= 1.0
        assert 0.0 <= result.rougeL <= 1.0
        assert result.latency_ms > 0

    def test_summarize_computes_means(self):
        """summarize() should compute correct averages over a list of EvalResult."""
        ev = _make_evaluator()
        results = [
            ev.evaluate_all("Q1", "R1", "H1", "zero_shot", "m", 100.0),
            ev.evaluate_all("Q2", "R2", "H2", "zero_shot", "m", 200.0),
        ]
        summary = Evaluator.summarize(results)
        assert summary["n"] == 2
        assert 0.0 <= summary["bleu_mean"] <= 1.0
        assert summary["latency_ms_mean"] == pytest.approx(150.0, rel=0.01)

    def test_summarize_empty_list(self):
        """summarize() with an empty list should return an empty dict."""
        assert Evaluator.summarize([]) == {}