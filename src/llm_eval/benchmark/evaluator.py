"""
src/llm_eval/benchmark/evaluator.py
"""
from __future__ import annotations


import nltk
nltk.download("punkt_tab", quiet=True)


from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from typing import Protocol, runtime_checkable


from llm_eval.shared.types import EvalResult, JudgeScore



@runtime_checkable
class LLMJudge(Protocol):
    def judge(self, question: str, reference: str, hypothesis: str) -> JudgeScore: ...



class Evaluator:
   
   
    def __init__(self, llm: LLMJudge):
        self._llm = llm
        self._rouge = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )
        self._smoother = SmoothingFunction().method1


    def bleu(self, reference: str, hypothesis: str) -> float:
        if not hypothesis.strip():
            return 0.0
        score: float = sentence_bleu(  # type: ignore[assignment]
            [reference.lower().split()],
            hypothesis.lower().split(),
            smoothing_function=self._smoother,
        )
        return score


    def rouge(self, reference: str, hypothesis: str) -> dict[str, float]:
        if not hypothesis.strip():
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        scores = self._rouge.score(reference, hypothesis)
        return {
            "rouge1": round(scores["rouge1"].fmeasure, 4),
            "rouge2": round(scores["rouge2"].fmeasure, 4),
            "rougeL": round(scores["rougeL"].fmeasure, 4),
        }


    def llm_judge(self, question: str, reference: str, hypothesis: str) -> JudgeScore:
        return self._llm.judge(question=question, reference=reference, hypothesis=hypothesis)


    def evaluate_all(
        self,
        question: str,
        reference: str,
        hypothesis: str,
        strategy: str,
        model: str,
        latency_ms: float,
    ) -> EvalResult:
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


    @staticmethod
    def summarize(results: list[EvalResult]) -> dict[str, float]:
        n = len(results)
        if n == 0:
            return {}
        return {
            "n": n,
            "bleu_mean":         round(sum(r.bleu        for r in results) / n, 4),
            "rouge1_mean":       round(sum(r.rouge1      for r in results) / n, 4),
            "rouge2_mean":       round(sum(r.rouge2      for r in results) / n, 4),
            "rougeL_mean":       round(sum(r.rougeL      for r in results) / n, 4),
            "judge_score_mean":  round(sum(r.judge_score for r in results) / n, 2),
            "judge_accuracy":    round(sum(r.judge_correct for r in results) / n, 4),
            "latency_ms_mean":   round(sum(r.latency_ms  for r in results) / n, 1),
        }