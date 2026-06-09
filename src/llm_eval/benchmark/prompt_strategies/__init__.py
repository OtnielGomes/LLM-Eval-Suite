"""
src/llm_eval/benchmark/prompt_strategies/__init__.py
=====================================================
Public API for the prompt strategies sub-package.

Exposes the base classes and all concrete strategy implementations so that
consumers can import directly from the sub-package root without needing to
know the internal module structure:

    # Preferred — import from sub-package root
    from llm_eval.benchmark.prompt_strategies import ZeroShotStrategy

    # Instead of the verbose internal path
    from llm_eval.benchmark.prompt_strategies.zero_shot import ZeroShotStrategy

Available strategies:

.. list-table::
   :header-rows: 1

   * - Class
     - Description
   * - ``ZeroShotStrategy``
     - Prompts the model with only the question and answer choices, no examples.
   * - ``FewShotStrategy``
     - Prepends a fixed set of worked examples before the target question.
   * - ``ChainOfThoughtStrategy``
     - Instructs the model to reason step-by-step before selecting an answer.

Adding a new strategy:
    1. Create ``src/llm_eval/benchmark/prompt_strategies/<name>.py`` and
       implement a class that inherits from ``PromptStrategy``.
    2. Add an import and an ``__all__`` entry in this file.
    3. Register the strategy in the ``STRATEGIES`` dict in ``run_benchmark.py``.
"""
from llm_eval.benchmark.prompt_strategies.base import PromptConfig, PromptStrategy
from llm_eval.benchmark.prompt_strategies.zero_shot import ZeroShotStrategy
from llm_eval.benchmark.prompt_strategies.few_shot import FewShotStrategy
from llm_eval.benchmark.prompt_strategies.chain_of_thought import ChainOfThoughtStrategy

__all__ = [
    "PromptConfig",
    "PromptStrategy",
    "ZeroShotStrategy",
    "FewShotStrategy",
    "ChainOfThoughtStrategy",
]