"""
src/llm_eval/clients/ollama_cliente.py
=======================================
LLM client for local inference via Ollama.

Communicates with the Ollama daemon's OpenAI-compatible REST API running at
``localhost:11434``. Provides the same interface as ``GeminiClient`` and
``OllamaCloudClient`` so that backends are interchangeable with a one-line
swap in ``run_benchmark.py``.

Architecture note:
    Ollama does not support native structured output (``response_schema``), so
    the :meth:`judge` method uses prompt-based JSON extraction followed by
    manual parsing instead of the SDK-level ``response.parsed`` pattern used
    in ``GeminiClient``.

Prerequisites:
    - Ollama daemon running: ``ollama serve``
    - Target model pulled:   ``ollama pull llama3.1:8b``

Configuration (resolved from ``shared/config.py``):
    OLLAMA_BASE_URL        : Ollama daemon address, e.g. ``"http://localhost:11434"``
    OLLAMA_LLM_MODEL       : Default generation model, e.g. ``"llama3.1:8b"``
    JUDGE_PROMPT_TEMPLATE  : Shared evaluation rubric template (score 1–5)

Dependencies:
    openai >= 1.0
"""

from __future__ import annotations

import json
import time
from openai import OpenAI

from llm_eval.shared.types import LLMResponse, JudgeScore
from llm_eval.shared.config import OLLAMA_BASE_URL, OLLAMA_LLM_MODEL, JUDGE_PROMPT_TEMPLATE


class OllamaClient:

    """Wrapper around the Ollama OpenAI-compatible API (localhost).

    Uses the ``openai`` SDK pointed at ``localhost:11434`` instead of the
    OpenAI servers. This works because Ollama implements the same
    ``/v1/chat/completions`` REST interface, making the SDK fully reusable
    without any additional dependencies.

    Attributes:
        client (OpenAI): SDK instance configured for the local Ollama daemon.
        model_name (str): Model identifier used in every API call.
    """

    def __init__(self, model: str = OLLAMA_LLM_MODEL):

        """Args:
            model: Ollama model identifier to use for generation and evaluation.
                   Must be available locally (``ollama pull <model>``).
                   Defaults to ``OLLAMA_LLM_MODEL`` from ``config.py``.

        Note:
            ``api_key="ollama"`` is a required placeholder — the ``openai``
            SDK refuses to instantiate without a non-empty key, but the
            Ollama daemon ignores its value entirely.
        """

        # Ollama's OpenAI-compatible endpoint lives at /v1 under the base URL
        self.client = OpenAI(base_url=f"{OLLAMA_BASE_URL}/v1", api_key="ollama")
        self.model_name = model

    def complete(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        
        """Generate a text response synchronously.

        Constructs a Chat Completions message list from the optional system
        prompt and user prompt, calls the Ollama daemon, and returns the
        result as an ``LLMResponse``.

        Args:
            prompt:      User message text sent to the model.
            system:      Optional system prompt to set model behaviour.
                         Omitted from the message list when empty.
            temperature: Sampling temperature. Values in ``[0.0, 0.3]``
                         produce more deterministic outputs; higher values
                         increase creativity and variability.
            max_tokens:  Maximum number of tokens in the generated response.
                         Controls output length and prevents runaway generation.

        Returns:
            ``LLMResponse`` with the model name, original prompt, generated
            text, and end-to-end latency in milliseconds.

        Raises:
            ValueError: If the model returns ``content=None``, which can occur
                        when the Ollama daemon is overloaded or the model is
                        not fully loaded.
        """

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

        """Evaluate a model-generated answer against a reference using LLM-as-judge.

        Builds an evaluation prompt from ``JUDGE_PROMPT_TEMPLATE`` (the shared
        rubric defined in ``config.py``) and appends an explicit JSON format
        instruction, since Ollama does not support native structured output.

        The response text is scanned for the outermost ``{...}`` block and
        parsed with ``json.loads``. This handles models that wrap JSON in
        markdown code fences or add explanatory text before or after the object.

        Args:
            question:   The original question posed to the model under evaluation.
            reference:  The ground-truth / expected answer.
            hypothesis: The model-generated answer to be evaluated.

        Returns:
            ``JudgeScore`` with a numeric score (1–5), textual justification,
            and a boolean correctness verdict.

        Note:
            On JSON parse failure, returns a fallback ``JudgeScore`` with
            ``score=1`` (lowest valid score) and the exception message as the
            reason, rather than raising. This prevents a single malformed
            response from aborting a batch evaluation run.
        """

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

        """Stream the model response token-by-token as a Python generator.

        Unlike :meth:`complete`, which blocks until the full response is ready,
        ``stream`` yields text fragments incrementally as the model generates
        them. Useful for long-form outputs where progressive display improves
        perceived latency.

        Args:
            prompt: User message text sent to the model.
            system: Optional system prompt. Omitted from the message list when empty.

        Yields:
            str: Non-empty text fragments (delta chunks) as they arrive from
                 the Ollama daemon. ``None`` and empty-string chunks are
                 silently skipped.

        Example:
            >>> for token in client.stream("Explain RAG in detail"):
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

        Ollama's OpenAI-compatible API does not expose a dedicated token-counting
        endpoint (unlike the native Gemini SDK). This method uses word-splitting
        as a fast approximation suitable for rough cost and context-length
        estimation.

        Args:
            text: Input text to estimate token count for.

        Returns:
            Approximate token count (number of whitespace-separated words).

        Note:
            This is a lower-bound approximation. Sub-word tokenisers (BPE, SentencePiece)
            typically produce 20–40 % more tokens than whitespace splits for English text,
            and significantly more for code or non-Latin scripts. For precise counts,
            use a model-specific tokeniser such as ``tiktoken`` for Llama-based models.
        """
        
        return len(text.split())
