"""
src/llm_eval/benchmark/prompt_strategies/few_shot.py
"""
from __future__ import annotations
from llm_eval.benchmark.prompt_strategies.base import PromptConfig, PromptStrategy


EXAMPLES = [
    {
        "question": "What is the chemical formula for water?",
        "choices":  ["H2O2", "H2O", "CO2", "NaCl"],
        "answer":   "B",
        "reason":   "Water consists of 2 hydrogen atoms and 1 oxygen atom.",
    },
    {
        "question": "Which planet is closest to the Sun?",
        "choices":  ["Venus", "Mars", "Mercury", "Earth"],
        "answer":   "C",
        "reason":   "Mercury is the innermost planet of the Solar System.",
    },
    {
        "question": "What is the derivative of x²?",
        "choices":  ["x", "2x", "x²", "2x²"],
        "answer":   "B",
        "reason":   "By the power rule, d/dx(x²) = 2x.",
    },
]



class FewShotStrategy:

    def build_prompt(self, question: str, choices: list[str]) -> str:
        parts = ["Here are some example questions and answers:\n"]
        for ex in EXAMPLES:
            ex_choices = "\n".join(f"  {chr(65+i)}) {c}" for i, c in enumerate(ex["choices"]))
            parts.append(
                f"Question: {ex['question']}\n"
                f"Choices:\n{ex_choices}\n"
                f"Answer: {ex['answer']}\n"
                f"Reason: {ex['reason']}\n"
            )
        target_choices = "\n".join(f"  {chr(65+i)}) {c}" for i, c in enumerate(choices))
        parts.append(
            f"Now answer this question:\n"
            f"Question: {question}\n"
            f"Choices:\n{target_choices}\n"
            f"Answer: "
        )
        return "\n".join(parts)


    def get_config(self) -> PromptConfig:
        return PromptConfig(
            temperature=0.0,
            max_tokens=10,
            system_instruction=(
                "You are a precise assistant. "
                "Answer with ONLY the letter (A, B, C, or D). No explanation."
            ),
        )



assert isinstance(FewShotStrategy(), PromptStrategy)