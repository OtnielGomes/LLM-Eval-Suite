"""
src/llm_eval/benchmark/prompt_strategies/zero_shot.py
======================================================
Zero-shot prompt strategy for multiple-choice benchmark evaluation.

Sends only the question and answer choices to the model — no worked examples,
no reasoning instructions. Serves as the performance baseline against which
``FewShotStrategy`` and ``ChainOfThoughtStrategy`` are compared.

Design decisions:
    - **``max_tokens=5``**: the tightest budget across all strategies. The
      model is instructed to reply with a single letter (1 token); 5 tokens
      provides a safe margin for leading spaces or punctuation without
      allowing verbose explanations.
    - **``temperature=0.0``**: maximum determinism. Zero-shot relies entirely
      on the model's parametric knowledge — stochasticity would only introduce
      noise into the benchmark results.
    - **Dual format constraint**: the answer instruction appears both in the
      user prompt (``"Answer with ONLY the letter..."``) and in the system
      prompt, reducing format non-compliance across models that weight the two
      roles differently.

Trade-offs vs other strategies:
    Lower token cost and fastest latency, at the expense of accuracy on
    complex reasoning tasks where examples or step-by-step instructions
    provide meaningful signal.
"""
from __future__ import annotations

from llm_eval.benchmark.prompt_strategies.base import PromptConfig, PromptStrategy


class ZeroShotStrategy:
    """Zero-shot prompt strategy: question and choices only, no examples.

    The simplest strategy in the suite. Formats the question and lettered
    choices into a minimal prompt and constrains the response to a single
    letter via both the user prompt and the system instruction.

    Acts as the performance baseline in comparative benchmark runs —
    accuracy improvements from ``FewShotStrategy`` and
    ``ChainOfThoughtStrategy`` are measured relative to this strategy.

    Satisfies the :class:`~llm_eval.benchmark.prompt_strategies.base.PromptStrategy`
    protocol without inheritance — verified by the ``assert isinstance`` check
    at the bottom of this module.
    """

    def build_prompt(self, question: str, choices: list[str]) -> str:
        """Build a minimal zero-shot prompt for a multiple-choice question.

        Formats choices as lettered options (A–D) and appends a single-line
        answer instruction. No examples or reasoning scaffolding are included.

        Args:
            question: The question text from the benchmark dataset.
            choices:  List of answer option strings. Typically 4 items
                      (A through D), but the method handles any length.

        Returns:
            Formatted prompt string with choices and a direct answer instruction.

        Example:
            >>> strategy = ZeroShotStrategy()
            >>> print(strategy.build_prompt(
            ...     question="What is the capital of France?",
            ...     choices=["Berlin", "Madrid", "Paris", "Rome"]
            ... ))
            Question: What is the capital of France?
            <BLANKLINE>
            Choices:
              A) Berlin
              B) Madrid
              C) Paris
              D) Rome
            <BLANKLINE>
            Answer with ONLY the letter (A, B, C, or D):
        """
        choices_text = "\n".join(
            f"  {chr(65+i)}) {c}" for i, c in enumerate(choices)
        )
        return (
            f"Question: {question}\n\n"
            f"Choices:\n{choices_text}\n\n"
            "Answer with ONLY the letter (A, B, C, or D):\n"
        )

    def get_config(self) -> PromptConfig:
        """Return generation parameters optimised for zero-shot single-letter responses.

        Uses the tightest ``max_tokens`` budget in the strategy suite (``5``),
        reflecting that a correct zero-shot response is a single letter.
        The small budget minimises API cost on large benchmark runs and
        prevents the model from appending unsolicited explanations that
        would complicate answer extraction.

        Returns:
            ``PromptConfig`` with:

            - ``temperature=0.0``   — fully deterministic, no noise in baseline
            - ``max_tokens=5``      — single-letter response with safety margin
            - ``system_instruction`` — reinforces letter-only constraint at model level

        Note:
            ``max_tokens=5`` rather than ``1`` accounts for models that emit
            a leading space or newline before the letter token. A budget of
            ``1`` would truncate such responses before the letter is reached.
        """
        return PromptConfig(
            temperature=0.0,
            # 5 tokens: 1 letter + margin for leading whitespace/newline tokens
            # that some models emit before the answer character
            max_tokens=5,
            system_instruction=(
                "You are a precise assistant that answers multiple-choice questions. "
                "Respond with ONLY the letter of the correct answer (A, B, C, or D)."
            ),
        )


# Verify Protocol conformance at import time without requiring inheritance.
# Raises AssertionError immediately if build_prompt() or get_config() are
# removed or renamed, surfacing the contract violation on import rather than
# at runtime during a benchmark run.
assert isinstance(ZeroShotStrategy(), PromptStrategy)