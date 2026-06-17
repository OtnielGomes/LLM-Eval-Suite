"""
src/llm_eval/benchmark/prompt_strategies/chain_of_thought.py
"""
from __future__ import annotations
import re
from llm_eval.benchmark.prompt_strategies.base import PromptConfig, PromptStrategy



class ChainOfThoughtStrategy:
    """CoT: instrui o modelo a raciocinar passo a passo antes da resposta."""


    def build_prompt(self, question: str, choices: list[str]) -> str:
        choices_text = "\n".join(f"  {chr(65+i)}) {c}" for i, c in enumerate(choices))
        return (
            f"Question: {question}\n\n"
            f"Choices:\n{choices_text}\n\n"
            "Think step by step, then end your response with:\n"
            "Therefore, the answer is: "
        )


    def get_config(self) -> PromptConfig:
        return PromptConfig(
            temperature=0.2,
            max_tokens=512,
            system_instruction=(
                "You are an expert reasoning assistant. "
                "Think step by step to solve the problem. "
                "Always end your response with: 'Therefore, the answer is: '"
            ),
        )


    @staticmethod
    def parse_answer(cot_response: str) -> str:
        match = re.search(r"the answer is[:\s]+([A-D])", cot_response, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        letters = re.findall(r"\b([A-D])\b", cot_response)
        return letters[-1].upper() if letters else ""



assert isinstance(ChainOfThoughtStrategy(), PromptStrategy)