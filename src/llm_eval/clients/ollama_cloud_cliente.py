"""
src/llm_eval/clients/ollama_cloud_cliente.py
"""
from __future__ import annotations


import json
import os
import time


from dotenv import load_dotenv
from openai import OpenAI


from llm_eval.shared.types import LLMResponse, JudgeScore
from llm_eval.shared.config import (
    OLLAMA_CLOUD_BASE_URL,
    OLLAMA_CLOUD_MODEL,
    JUDGE_PROMPT_TEMPLATE,
)


load_dotenv()



class OllamaCloudClient:
    """Wrapper sobre a API cloud do Ollama via interface OpenAI-compatible."""


    def __init__(self, model: str = OLLAMA_CLOUD_MODEL):
        # FIX: lê OLLAMA_API_KEY via os.getenv() em tempo de instanciação (lazy),
        # NÃO via OLLAMA_CLOUD_API_KEY importada do config.py.
        # A constante importada é congelada no momento do import do módulo,
        # o que impede que patch.dict("os.environ") e monkeypatch.setenv
        # funcionem corretamente nos testes.
        api_key = os.getenv("OLLAMA_API_KEY")
        if not api_key:
            raise ValueError(
                "OLLAMA_API_KEY not found. "
                "Create it at https://ollama.com/settings/keys and add it to .env."
            )
        self.client = OpenAI(base_url=OLLAMA_CLOUD_BASE_URL, api_key=api_key)
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
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = r.choices[0].message.content
        if content is None:
            raise ValueError(
                f"Ollama Cloud returned content=None for model='{self.model_name}'."
            )
        return LLMResponse(
            model=self.model_name,
            prompt=prompt,
            response=content,
            latency_ms=(time.monotonic() - start) * 1000,
        )


    def judge(self, question: str, reference: str, hypothesis: str) -> JudgeScore:
        json_format = 'Format: {"score": <1-5>, "reason": "", "is_correct": }'
        prompt = (
            JUDGE_PROMPT_TEMPLATE.format(
                question=question, reference=reference, hypothesis=hypothesis
            )
            + "\n\nReturn ONLY valid JSON. " + json_format
        )
        resp = self.complete(prompt, temperature=0)
        try:
            start = resp.response.index("{")
            end = resp.response.rindex("}") + 1
            return JudgeScore(**json.loads(resp.response[start:end]))
        except Exception as e:
            return JudgeScore(
                score=1,
                reason=f"parse_error: {e} | raw='{resp.response[:100]}'",
                is_correct=False,
            )


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