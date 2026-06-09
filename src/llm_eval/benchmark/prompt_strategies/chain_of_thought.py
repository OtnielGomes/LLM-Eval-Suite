"""
src/llm_eval/benchmark/prompt_strategies/chain_of_thought.py
=============================================================
Chain-of-thought (CoT) prompt strategy for multi-choice benchmark evaluation.

Instructs the model to reason step by step through the problem before
committing to a final answer. CoT prompting consistently improves accuracy
on reasoning-heavy tasks (mathematics, logic, science) at the cost of higher
token usage and slightly longer latency compared to zero-shot.

The strategy relies on a constrained response format — every answer must end
with the literal phrase ``"Therefore, the answer is: <letter>"`` — which
:meth:`ChainOfThoughtStrategy.parse_answer` uses as an anchor for extraction.

Trade-offs vs other strategies:

.. list-table::
   :header-rows: 1

   * - Strategy
     - Accuracy (reasoning tasks)
     - Token usage
     - Latency
   * - ZeroShot
     - Baseline
     - Low
     - Fast
   * - FewShot
     - + moderate
     - Medium (examples in prompt)
     - Moderate
   * - **ChainOfThought**
     - **+ high**
     - **High (reasoning in response)**
     - **Slower**

Reference:
    Wei et al., 2022 — "Chain-of-Thought Prompting Elicits Reasoning in LLMs"
    https://arxiv.org/abs/2201.11903
"""
from __future__ import annotations

import re

from llm_eval.benchmark.prompt_strategies.base import PromptConfig, PromptStrategy


class ChainOfThoughtStrategy:
    """Chain-of-thought prompt strategy.

    Formats the question and choices with an explicit reasoning instruction,
    then anchors answer extraction on the phrase
    ``"Therefore, the answer is: <letter>"``.

    The system instruction reinforces the format constraint at the model
    level, while the user prompt repeats it inline. The dual instruction
    approach reduces format non-compliance across models that give lower
    weight to system prompts.

    Satisfies the :class:`~llm_eval.benchmark.prompt_strategies.base.PromptStrategy`
    protocol without inheritance — verified by the ``assert isinstance`` check
    at the bottom of this module.
    """

    def build_prompt(self, question: str, choices: list[str]) -> str:
        """Build a chain-of-thought prompt for a multiple-choice question.

        Formats choices as lettered options (A-D) and appends an explicit
        reasoning instruction with the required response anchor phrase.
        The anchor ``"Therefore, the answer is: "`` is left open-ended so
        the model completes it with a single letter.

        Args:
            question: The question text from the benchmark dataset.
            choices:  List of answer option strings. Typically 4 items
                      (A through D), but the method handles any length.

        Returns:
            Formatted prompt string with reasoning instruction and answer anchor.

        Example:
            >>> strategy = ChainOfThoughtStrategy()
            >>> prompt = strategy.build_prompt(
            ...     question="What is 2 + 2?",
            ...     choices=["3", "4", "5", "6"]
            ... )
            >>> print(prompt)
            Question: What is 2 + 2?
            <BLANKLINE>
            Choices:
              A) 3
              B) 4
              C) 5
              D) 6
            <BLANKLINE>
            Think step by step, then end your response with:
            Therefore, the answer is:
        """
        choices_text = "\n".join(f"  {chr(65+i)}) {c}" for i, c in enumerate(choices))
        return (
            f"Question: {question}\n\n"
            f"Choices:\n{choices_text}\n\n"
            "Think step by step, then end your response with:\n"
            "Therefore, the answer is: "
        )

    def get_config(self) -> PromptConfig:
        """Return generation parameters optimised for chain-of-thought reasoning.

        Uses a slightly higher temperature than zero-shot (``0.2`` vs ``0.1``)
        to allow the model some flexibility in its reasoning path, while
        staying deterministic enough for reproducible evaluation.

        ``max_tokens=512`` provides sufficient budget for a step-by-step
        reasoning trace followed by the answer anchor. Reducing this value
        risks truncating the response before the anchor phrase, causing
        :meth:`parse_answer` to fall back to the last-letter heuristic.

        Returns:
            ``PromptConfig`` with:

            - ``temperature=0.2``   — slightly exploratory for reasoning
            - ``max_tokens=512``    — budget for CoT trace + answer anchor
            - ``system_instruction`` — reinforces the anchor format at model level
        """
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
        """Extract the final answer letter from a chain-of-thought response.

        Uses a two-stage extraction strategy:

        1. **Anchor match** (primary): searches for the pattern
           ``"the answer is: <letter>"`` (case-insensitive, flexible spacing).
           Returns the matched letter when found.

        2. **Last-letter heuristic** (fallback): if the anchor is absent
           (e.g. the model truncated its response or deviated from the format),
           extracts all standalone ``A``-``D`` letters and returns the last one.
           The last mentioned letter is typically the conclusion of the
           reasoning chain.

        Returns an empty string when neither strategy finds a letter, signalling
        to the evaluator that the response should be scored as incorrect.

        Args:
            cot_response: Raw text response generated by the model.

        Returns:
            Uppercase letter (``"A"``-``"D"``) of the extracted answer,
            or ``""`` if no answer letter could be identified.

        Example:
            >>> ChainOfThoughtStrategy.parse_answer(
            ...     "The sum is 4. Therefore, the answer is: B"
            ... )
            'B'
            >>> ChainOfThoughtStrategy.parse_answer(
            ...     "It could be A or C, but mostly C makes sense."
            ... )
            'C'
            >>> ChainOfThoughtStrategy.parse_answer("The model forgot the format.")
            ''
        """
        # Primary: anchor phrase — most reliable signal
        match = re.search(r"the answer is[:\s]+([A-D])", cot_response, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        # Fallback: last standalone A–D letter in the response —
        # typically the conclusion of the reasoning chain when the anchor is missing
        letters = re.findall(r"\b([A-D])\b", cot_response)
        return letters[-1].upper() if letters else ""


# Verify Protocol conformance at import time without requiring inheritance.
# This acts as a lightweight integration test: if build_prompt() or
# get_config() are ever removed or renamed, this line raises AssertionError
# immediately on import rather than failing silently at runtime.
assert isinstance(ChainOfThoughtStrategy(), PromptStrategy)