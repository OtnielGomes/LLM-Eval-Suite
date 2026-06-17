"""
src/llm_eval/benchmark/prompt_strategies/__init__.py
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