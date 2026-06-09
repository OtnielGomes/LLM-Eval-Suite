"""
src/llm_eval/clients/ollama_cloud_cliente.py
============================================
LLM client for remote inference via Ollama Cloud.

Communicates with the Ollama Cloud API using the OpenAI-compatible REST
interface. Functionally mirrors ``OllamaClient`` (localhost) but targets
Ollama's hosted infrastructure, enabling access to large models (e.g.
``llama3.3:70b``) without local GPU hardware.

This is the **default backend** in ``run_benchmark.py`` when neither
``--local`` nor ``--gemini`` is passed, as it combines zero cost with
access to production-grade model sizes.

Architecture note:
    Like ``OllamaClient``, Ollama Cloud does not support native structured
    output (``response_schema``). The :meth:`judge` method therefore uses
    prompt-based JSON extraction with manual parsing, identical to the
    local client strategy.

    ``OLLAMA_API_KEY`` is read via ``os.getenv()`` at instantiation time
    (lazy evaluation) rather than imported as a module-level constant. This
    is intentional: a frozen import-time constant would prevent
    ``patch.dict("os.environ")`` and ``monkeypatch.setenv`` from working
    correctly in tests.

Prerequisites:
    - Ollama Cloud account: https://ollama.com
    - API key set in ``.env``: ``OLLAMA_API_KEY=<your-key>``

Configuration (resolved from ``shared/config.py``):
    OLLAMA_CLOUD_BASE_URL  : Ollama Cloud API endpoint
    OLLAMA_CLOUD_MODEL     : Default cloud model, e.g. ``"llama3.3:70b"``
    JUDGE_PROMPT_TEMPLATE  : Shared evaluation rubric template (score 1–5)

Dependencies:
    openai >= 1.0, python-dotenv
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

    """Wrapper around the Ollama Cloud OpenAI-compatible API.

    Provides the same interface as ``OllamaClient`` and ``GeminiClient``
    so that all three backends are interchangeable with a one-line swap
    in ``run_benchmark.py`` and ``evaluate_rag.py``.

    Attributes:
        client (OpenAI): SDK instance configured for Ollama Cloud.
        model_name (str): Model identifier used in every API call.
    """

    def __init__(self, model: str = OLLAMA_CLOUD_MODEL):
        
        """Initialise the client and authenticate with Ollama Cloud.

        Reads ``OLLAMA_API_KEY`` from the environment at instantiation time
        rather than at module import time. This lazy evaluation ensures that
        test fixtures using ``monkeypatch.setenv`` or
        ``patch.dict("os.environ")`` can inject the key before the client
        is created, without needing to reload the module.

        Args:
            model: Ollama Cloud model identifier.
                   Defaults to ``OLLAMA_CLOUD_MODEL`` from ``config.py``.

        Raises:
            ValueError: If ``OLLAMA_API_KEY`` is not set in the environment.
                        The error message includes the URL to generate a key.
        """

        api_key = os.getenv("OLLAMA_API_KEY")
        if not api_key:
            raise ValueError(
                "OLLAMA_API_KEY not found."
                "Create one at https://ollama.com/settings/keys and add it to .env."
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
        
        """Generate a text response from Ollama Cloud synchronously.

        Constructs a Chat Completions message list from the optional system
        prompt and user prompt, calls the Ollama Cloud endpoint, and returns
        the result as an ``LLMResponse``.

        Args:
            prompt:      User message text sent to the model.
            system:      Optional system prompt to set model behaviour.
                         Omitted from the message list when empty.
            temperature: Sampling temperature. Values in ``[0.0, 0.3]``
                         produce more deterministic outputs.
            max_tokens:  Maximum number of tokens in the generated response.

        Returns:
            ``LLMResponse`` with the model name, original prompt, generated
            text, and end-to-end latency in milliseconds.

        Raises:
            ValueError: If the API returns ``content=None``, which may indicate
                        a quota limit, a content policy block, or a transient
                        cloud-side error.
        """

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

        """Evaluate a model-generated answer against a reference using LLM-as-judge.

        Builds an evaluation prompt from ``JUDGE_PROMPT_TEMPLATE`` and appends
        an explicit JSON format instruction. Because Ollama Cloud does not
        support native structured output, the response is parsed by locating
        the outermost ``{...}`` block in the raw text.

        The raw prefix of the response is included in the fallback ``reason``
        field (first 100 characters) to aid debugging when parse failures occur
        in batch evaluation runs.

        Args:
            question:   The original question posed to the model under evaluation.
            reference:  The ground-truth / expected answer.
            hypothesis: The model-generated answer to be evaluated.

        Returns:
            ``JudgeScore`` with a numeric score (1–5), textual justification,
            and a boolean correctness verdict.

        Note:
            On JSON parse failure, returns a fallback ``JudgeScore`` with
            ``score=1`` and the exception message plus raw response prefix as
            the reason. This prevents a single malformed response from aborting
            a batch evaluation run.
        """

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

        """Stream the model response token-by-token as a Python generator.

        Yields text fragments incrementally as Ollama Cloud generates them,
        enabling progressive display without waiting for the full response.

        Args:
            prompt: User message text sent to the model.
            system: Optional system prompt. Omitted from the message list
                    when empty.

        Yields:
            str: Non-empty text fragments (delta chunks) as they arrive.
                 ``None`` and empty-string chunks are silently skipped.

        Example:
            >>> for token in client.stream("Explain chain-of-thought prompting"):
            ...     print(token, end="", flush=True)
        """

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

        """Estimate the token count for the given text using whitespace tokenisation.

        Ollama Cloud does not expose a dedicated token-counting endpoint.
        This method uses word-splitting as a fast approximation suitable for
        rough context-length and cost estimation.

        Args:
            text: Input text to estimate token count for.

        Returns:
            Approximate token count (number of whitespace-separated words).

        Note:
            Sub-word tokenisers (BPE, SentencePiece) typically produce 20–40 %
            more tokens than whitespace splits for English text, and
            significantly more for code or non-Latin scripts. For precise
            counts, use a model-specific tokeniser such as ``tiktoken``.
        """
        
        return len(text.split())
