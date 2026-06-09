"""
src/llm_eval/benchmark/prompt_strategies/few_shot.py
=====================================================
Few-shot prompt strategy for multiple-choice benchmark evaluation.

Prepends three fixed worked examples (question → choices → answer → reason)
before the target question. The examples demonstrate the expected response
format and anchor the model's output to a single letter, enabling reliable
answer extraction without a dedicated parser.

Few-shot prompting improves accuracy over zero-shot by reducing format
non-compliance and providing implicit task framing. The trade-off is a larger
input prompt (roughly 3× the token count of zero-shot) on every call.

Design decisions:
    - **Fixed examples**: the three examples in ``EXAMPLES`` span chemistry,
      astronomy, and mathematics — three distinct MMLU subject domains — to
      reduce the risk of in-context bias toward any single subject area.
    - **``max_tokens=10``**: the model is instructed to reply with a single
      letter. A strict token cap prevents verbose outputs and reduces API cost
      on batch evaluation runs.
    - **``temperature=0.0``**: maximum determinism. Few-shot examples already
      constrain the output format; stochasticity provides no benefit here.

Reference:
    Brown et al., 2020 — "Language Models are Few-Shot Learners" (GPT-3)
    https://arxiv.org/abs/2005.14165
"""
from __future__ import annotations

from llm_eval.benchmark.prompt_strategies.base import PromptConfig, PromptStrategy


# ---------------------------------------------------------------------------
# Fixed few-shot examples
# ---------------------------------------------------------------------------

# Three examples spanning distinct MMLU subject domains (chemistry, astronomy,
# mathematics) to minimise in-context subject bias in the model's responses.
# Each example includes a brief "reason" field that demonstrates the expected
# reasoning pattern without triggering full chain-of-thought generation.
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


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------


class FewShotStrategy:
    """Few-shot prompt strategy with 3 fixed worked examples.

    Formats each example as a ``Question / Choices / Answer / Reason`` block
    and appends the target question with an open ``"Answer: "`` anchor that
    the model completes with a single letter.

    The ``reason`` field in each example provides just enough context to
    demonstrate correct reasoning without inflating the token budget as
    heavily as a full chain-of-thought trace would.

    Satisfies the :class:`~llm_eval.benchmark.prompt_strategies.base.PromptStrategy`
    protocol without inheritance — verified by the ``assert isinstance`` check
    at the bottom of this module.
    """

    def build_prompt(self, question: str, choices: list[str]) -> str:
        """Build a few-shot prompt with 3 worked examples followed by the target question.

        Iterates over ``EXAMPLES`` to construct the demonstration block, then
        appends the target question with an open ``"Answer: "`` anchor. The
        anchor is left incomplete so the model fills it with a single letter
        (A–D), matching the pattern shown in the examples.

        Args:
            question: The target question text from the benchmark dataset.
            choices:  List of answer option strings for the target question.
                      Typically 4 items (A through D).

        Returns:
            Formatted prompt string containing the 3 worked examples followed
            by the target question and an open answer anchor.

        Example output structure::

            Here are some example questions and answers:

            Question: What is the chemical formula for water?
            Choices:
              A) H2O2
              B) H2O
              ...
            Answer: B
            Reason: Water consists of 2 hydrogen atoms and 1 oxygen atom.

            [... 2 more examples ...]

            Now answer this question:
            Question: <target question>
            Choices:
              A) ...
            Answer:
        """
        parts = ["Here are some example questions and answers:\n"]
        for ex in EXAMPLES:
            ex_choices = "\n".join(
                f"  {chr(65+i)}) {c}" for i, c in enumerate(ex["choices"])
            )
            parts.append(
                f"Question: {ex['question']}\n"
                f"Choices:\n{ex_choices}\n"
                f"Answer: {ex['answer']}\n"
                f"Reason: {ex['reason']}\n"
            )
        target_choices = "\n".join(
            f"  {chr(65+i)}) {c}" for i, c in enumerate(choices)
        )
        parts.append(
            f"Now answer this question:\n"
            f"Question: {question}\n"
            f"Choices:\n{target_choices}\n"
            f"Answer: "
        )
        return "\n".join(parts)

    def get_config(self) -> PromptConfig:
        """Return generation parameters optimised for few-shot single-letter responses.

        Uses ``temperature=0.0`` for maximum determinism — the worked examples
        already constrain the output format, so stochasticity provides no
        benefit and only risks format non-compliance.

        ``max_tokens=10`` is intentionally strict: the model is instructed to
        reply with a single letter (1 token). The small budget prevents
        verbose explanations, reduces API cost on batch runs, and makes the
        response trivially parseable without a regex extractor.

        Returns:
            ``PromptConfig`` with:

            - ``temperature=0.0``   — fully deterministic output
            - ``max_tokens=10``     — enforces single-letter response
            - ``system_instruction`` — explicit letter-only constraint
        """
        return PromptConfig(
            temperature=0.0,
            # A single letter costs 1 token; 10 tokens provides a safe margin
            # in case the model emits a space or punctuation before the letter
            max_tokens=10,
            system_instruction=(
                "You are a precise assistant. "
                "Answer with ONLY the letter (A, B, C, or D). No explanation."
            ),
        )


# Verify Protocol conformance at import time without requiring inheritance.
# Raises AssertionError immediately if build_prompt() or get_config() are
# removed or renamed, surfacing the contract violation on import rather than
# at runtime during a benchmark run.
assert isinstance(FewShotStrategy(), PromptStrategy)