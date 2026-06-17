"""
src/llm_eval/clients/ollama_cliente.py
"""
from __future__ import annotations


import json
import time
from openai import OpenAI


from llm_eval.shared.types import LLMResponse, JudgeScore
from llm_eval.shared.config import OLLAMA_BASE_URL, OLLAMA_LLM_MODEL, JUDGE_PROMPT_TEMPLATE



class OllamaClient:
    """Wrapper sobre a API OpenAI-compatible do Ollama (localhost)."""


    def __init__(self, model: str = OLLAMA_LLM_MODEL):
        self.client = OpenAI(base_url=f"{OLLAMA_BASE_URL}/v1", api_key="ollama")
        self.model_name = model


    def complete(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        start = time.monotonic()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        r = self.client.chat.completions.create(
            model=self.model_name, messages=messages,
            temperature=temperature, max_tokens=max_tokens,
        )
        content = r.choices[0].message.content
        if content is None:
            raise ValueError(f"Ollama returned content=None for model='{self.model_name}'.")
        return LLMResponse(
            model=self.model_name, prompt=prompt, response=content,
            latency_ms=(time.monotonic() - start) * 1000,
        )


    def judge(self, question: str, reference: str, hypothesis: str) -> JudgeScore:
        prompt = (
            JUDGE_PROMPT_TEMPLATE.format(
                question=question, reference=reference, hypothesis=hypothesis
            )
            + '\n\nReturn ONLY valid JSON. '
            'Format: {"score": <1-5>, "reason": "", "is_correct": }'
        )
        resp = self.complete(prompt, temperature=0)
        try:
            start = resp.response.index("{")
            end   = resp.response.rindex("}") + 1
            return JudgeScore(**json.loads(resp.response[start:end]))
        except Exception as e:
            return JudgeScore(score=1, reason=f"parse_error: {e}", is_correct=False)


    def stream(self, prompt: str, system: str = ""):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        for chunk in self.client.chat.completions.create(
            model=self.model_name, messages=messages, stream=True
        ):
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


    def count_tokens(self, text: str) -> int:
        return len(text.split())