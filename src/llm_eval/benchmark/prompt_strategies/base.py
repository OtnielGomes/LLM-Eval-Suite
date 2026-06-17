"""
01_benchmark/prompt_strategies/base.py

Base protocol for all prompting strategies.
Any class that implements build_prompt() and get_config() satisfies the protocol — no mandatory inheritance.
"""

from __future__ import annotations


from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable



@dataclass
class PromptConfig:
 
    temperature: float = 0.1
    max_tokens: int = 1024
    system_instruction: str = ""



@runtime_checkable
class PromptStrategy(Protocol):
    


    def build_prompt(self, question: str, choices: list[str]) -> str:
        ...


    def get_config(self) -> PromptConfig:
        ...