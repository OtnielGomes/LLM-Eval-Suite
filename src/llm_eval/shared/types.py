"""
src/llm_eval/shared/types.py

Canonical types shared between all modules.

"""
from dataclasses import dataclass, field
from pydantic import BaseModel


@dataclass
class LLMResponse:
    model: str
    prompt: str
    response: str
    latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0


class JudgeScore(BaseModel):
    
    score: int        
    reason: str
    is_correct: bool

@dataclass
class EvalResult:
    question: str
    expected: str
    predicted: str
    strategy: str
    model: str
    latency_ms: float
    bleu: float = 0.0
    rouge1: float = 0.0
    rouge2: float = 0.0
    rougeL: float = 0.0
    judge_score: int = 0
    judge_reason: str = ""
    judge_correct: bool = False


    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class RAGResult:
    question: str
    answer: str
    contexts: list[str] = field(default_factory=list)
    ground_truth: str = ""
    strategy: str = "naive"
    latency_ms: float = 0.0

