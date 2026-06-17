"""
src/llm_eval/benchmark/prompt_strategies/zero_shot.py
"""
from __future__ import annotations
from llm_eval.benchmark.prompt_strategies.base import PromptConfig, PromptStrategy



class ZeroShotStrategy:
    """Zero-shot: apenas a questão e as alternativas, sem exemplos."""


    def build_prompt(self, question: str, choices: list[str]) -> str:
        choices_text = "\n".join(f"  {chr(65+i)}) {c}" for i, c in enumerate(choices))
        return (
            f"Question: {question}\n\n"
            f"Choices:\n{choices_text}\n\n"
            "Answer with ONLY the letter (A, B, C, or D):\n"
        )


    def get_config(self) -> PromptConfig:
        return PromptConfig(
            temperature=0.0,
            max_tokens=5,
            system_instruction=(
                "You are a precise assistant that answers multiple-choice questions. "
                "Respond with ONLY the letter of the correct answer (A, B, C, or D)."
            ),
        )



assert isinstance(ZeroShotStrategy(), PromptStrategy)