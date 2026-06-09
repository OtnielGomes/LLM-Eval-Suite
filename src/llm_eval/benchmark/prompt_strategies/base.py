"""
src/llm_eval/benchmark/prompt_strategies/base.py
================================================
Base abstractions for all prompting strategies.

Defines two building blocks shared across every strategy implementation:

- ``PromptConfig``: a dataclass holding LLM generation parameters
  (temperature, max_tokens, system_instruction) that are forwarded to
  ``client.complete()`` in ``run_benchmark.py``.

- ``PromptStrategy``: a ``Protocol`` (structural subtyping) that declares
  the minimum interface a strategy must implement. Any class with
  ``build_prompt()`` and ``get_config()`` satisfies the protocol — no
  inheritance required.

Design rationale — Protocol vs ABC:
    ``PromptStrategy`` uses ``typing.Protocol`` instead of ``abc.ABC`` for
    two reasons:

    1. **Structural subtyping (duck typing)**: third-party or test-only
       strategy classes satisfy the interface without importing this module,
       reducing coupling.
    2. **Runtime checkability**: the ``@runtime_checkable`` decorator enables
       ``isinstance(obj, PromptStrategy)`` checks in ``run_benchmark.py``
       and test fixtures without requiring explicit registration.

    The trade-off is that Protocols do not enforce implementation at class
    definition time — a missing method is only caught when the method is
    called or an ``isinstance`` check is performed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Generation parameters
# ---------------------------------------------------------------------------


@dataclass
class PromptConfig:
    """Generation parameters associated with a prompt strategy.

    Returned by :meth:`PromptStrategy.get_config` and forwarded directly to
    ``client.complete()`` in ``run_benchmark.py``. Each strategy can override
    the defaults to reflect its specific generation requirements — for example,
    ``ChainOfThoughtStrategy`` may use a higher ``max_tokens`` budget to
    accommodate step-by-step reasoning.

    Attributes:
        temperature:        Sampling temperature passed to the LLM.
                            Lower values (``0.0``–``0.3``) yield more
                            deterministic outputs; higher values increase
                            diversity. Defaults to ``0.1``.
        max_tokens:         Maximum number of tokens the model may generate.
                            Controls output length and API cost. Defaults
                            to ``1024``.
        system_instruction: Optional system prompt injected as the first
                            message in the chat. Empty string means no system
                            prompt is sent. Defaults to ``""``.

    Example:
        >>> config = PromptConfig(temperature=0.0, max_tokens=512)
        >>> config.system_instruction
        ''
    """
    temperature: float = 0.1
    max_tokens: int = 1024
    system_instruction: str = ""


# ---------------------------------------------------------------------------
# Strategy protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class PromptStrategy(Protocol):
    """Structural interface that every prompt strategy must satisfy.

    Any class that implements :meth:`build_prompt` and :meth:`get_config`
    conforms to this protocol — inheritance from ``PromptStrategy`` is not
    required. This enables third-party or test-only strategies to satisfy
    the interface without importing this module.

    The ``@runtime_checkable`` decorator allows ``isinstance`` checks at
    runtime, which is used in ``run_benchmark.py`` to validate strategies
    retrieved from the ``STRATEGIES`` registry.

    Implementors:
        - :class:`~llm_eval.benchmark.prompt_strategies.zero_shot.ZeroShotStrategy`
        - :class:`~llm_eval.benchmark.prompt_strategies.few_shot.FewShotStrategy`
        - :class:`~llm_eval.benchmark.prompt_strategies.chain_of_thought.ChainOfThoughtStrategy`

    Example — implementing a custom strategy without inheritance::

        class MyStrategy:
            def build_prompt(self, question: str, choices: list[str]) -> str:
                opts = "\\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices))
                return f"Answer briefly.\\n\\n{question}\\n{opts}"

            def get_config(self) -> PromptConfig:
                return PromptConfig(temperature=0.0, max_tokens=128)

        assert isinstance(MyStrategy(), PromptStrategy)  # True — no inheritance needed
    """

    def build_prompt(self, question: str, choices: list[str]) -> str:
        """Construct the prompt text to be sent to the LLM.

        Receives the raw question and its answer choices and returns a
        fully formatted string ready to be passed to ``client.complete()``.
        The formatting style (e.g. labelling choices as A/B/C/D, adding
        few-shot examples, or prepending chain-of-thought instructions)
        is the primary differentiator between strategies.

        Args:
            question: The question text from the benchmark dataset.
            choices:  List of answer option strings, typically 4 items
                      corresponding to options A through D.

        Returns:
            Formatted prompt string ready for the LLM.
        """
        ...

    def get_config(self) -> PromptConfig:
        """Return the generation parameters for this strategy.

        Provides strategy-specific values for temperature, max_tokens, and
        system_instruction that are forwarded to ``client.complete()``.
        Strategies with longer expected outputs (e.g. chain-of-thought)
        should return a larger ``max_tokens`` budget here.

        Returns:
            ``PromptConfig`` with generation parameters appropriate for
            this strategy.
        """
        ...