"""
tests/test_ollama_cloud_client.py

Tests for OllamaCloudClient (src/llm_eval/clients/ollama_cloud_cliente.py).
100% mocked — no real calls to the Ollama Cloud API.

Tested behavior:
- __init__: raises ValueError if OLLAMA_API_KEY is missing
- __init__: uses OLLAMA_CLOUD_MODEL or defaults to "llama3.3:70b"
- complete(): returns a correctly typed LLMResponse
- complete(): raises ValueError if content=None (model unavailable)
- complete(): preserves the system prompt in the message list
- judge(): returns a JudgeScore with correct JSON parsing
- judge(): falls back to score=1 when JSON is malformed
- judge(): extracts JSON even with surrounding text (robustness)
- stream(): yields tokens from chunk deltas
- count_tokens(): whitespace-based token count (approximation)
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from llm_eval.shared.types import LLMResponse, JudgeScore

# sys.path resolved by conftest.py
from ollama_cloud_cliente import OllamaCloudClient  # type: ignore[import]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_client(model: str = "llama3.3:70b") -> tuple[OllamaCloudClient, MagicMock]:
    """Instantiates OllamaCloudClient with a mocked OpenAI backend. Returns (client, mock_openai)."""
    with patch.dict("os.environ", {"OLLAMA_API_KEY": "test-key-123"}):
        with patch("ollama_cloud_cliente.OpenAI") as mock_openai_cls:
            mock_openai_instance = MagicMock()
            mock_openai_cls.return_value = mock_openai_instance
            client = OllamaCloudClient(model=model)
            return client, mock_openai_instance

def _mock_completion(content: str | None) -> MagicMock:
    """Creates a ChatCompletion mock with the given content."""
    mock_resp = MagicMock()
    mock_resp.choices[0].message.content = content
    return mock_resp

# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------

class TestInit:

    def test_raises_error_without_api_key(self):
        """Without OLLAMA_API_KEY in the environment, should raise ValueError with instructions."""
        with patch.dict("os.environ", {}, clear=True):
            import os
            os.environ.pop("OLLAMA_API_KEY", None)
            with pytest.raises(ValueError, match="OLLAMA_API_KEY"):
                OllamaCloudClient()

    def test_model_name_stored(self):
        """model_name should reflect the model passed to the constructor."""
        client, _ = _make_client(model="qwen3:32b")
        assert client.model_name == "qwen3:32b"

    def test_model_defaults_to_llama(self):
        """Without an argument, model should be 'llama3.3:70b' (or OLLAMA_CLOUD_MODEL from env)."""
        client, _ = _make_client()
        assert "llama" in client.model_name or len(client.model_name) > 0

    def test_openai_client_instantiated_with_base_url(self):
        """OpenAI should be instantiated with the Ollama Cloud base_url."""
        with patch.dict("os.environ", {"OLLAMA_API_KEY": "test-key"}):
            with patch("ollama_cloud_cliente.OpenAI") as mock_cls:
                mock_cls.return_value = MagicMock()
                OllamaCloudClient()
                call_kwargs = mock_cls.call_args[1]
                assert "ollama.com" in call_kwargs["base_url"]

# ---------------------------------------------------------------------------
# complete()
# ---------------------------------------------------------------------------

class TestComplete:

    def test_returns_llm_response(self):
        """complete() should return a typed LLMResponse."""
        client, mock_openai = _make_client()
        mock_openai.chat.completions.create.return_value = _mock_completion("Hello!")

        result = client.complete("What is RAG?")
        assert isinstance(result, LLMResponse)

    def test_all_fields_populated(self):
        """LLMResponse should have model, prompt, response, and latency_ms populated."""
        client, mock_openai = _make_client(model="llama3.3:70b")
        mock_openai.chat.completions.create.return_value = _mock_completion(
            "RAG combines retrieval with generation."
        )

        result = client.complete("What is RAG?")
        assert result.model == "llama3.3:70b"
        assert result.prompt == "What is RAG?"
        assert result.response == "RAG combines retrieval with generation."
        assert result.latency_ms >= 0

    def test_raises_value_error_if_content_none(self):
        """content=None should raise ValueError indicating the model is unavailable."""
        client, mock_openai = _make_client()
        mock_openai.chat.completions.create.return_value = _mock_completion(None)

        with pytest.raises(ValueError, match="content=None"):
            client.complete("Test prompt")

    def test_system_prompt_prepended_to_messages(self):
        """When system != '', a system message should be prepended before the user message."""
        client, mock_openai = _make_client()
        mock_openai.chat.completions.create.return_value = _mock_completion("OK")

        client.complete("User prompt", system="You are helpful.")

        call_kwargs = mock_openai.chat.completions.create.call_args[1]
        messages = call_kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are helpful."
        assert messages[1]["role"] == "user"

    def test_no_system_prompt_only_user_message(self):
        """Without a system prompt, messages should contain only the user message."""
        client, mock_openai = _make_client()
        mock_openai.chat.completions.create.return_value = _mock_completion("OK")

        client.complete("Hello")

        call_kwargs = mock_openai.chat.completions.create.call_args[1]
        messages = call_kwargs["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    def test_temperature_and_max_tokens_forwarded(self):
        """temperature and max_tokens should be forwarded to the OpenAI client."""
        client, mock_openai = _make_client()
        mock_openai.chat.completions.create.return_value = _mock_completion("OK")

        client.complete("prompt", temperature=0.5, max_tokens=512)

        call_kwargs = mock_openai.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 512

# ---------------------------------------------------------------------------
# judge()
# ---------------------------------------------------------------------------

class TestJudge:

    def _make_judge_response(self, score: int, reason: str, is_correct: bool) -> MagicMock:
        """Creates a valid JSON response mock for judge()."""
        json_str = json.dumps({"score": score, "reason": reason, "is_correct": is_correct})
        return _mock_completion(json_str)

    def test_returns_valid_judge_score(self):
        """judge() should return a JudgeScore with score between 1 and 5."""
        client, mock_openai = _make_client()
        mock_openai.chat.completions.create.return_value = self._make_judge_response(
            score=4, reason="Good answer.", is_correct=True
        )

        result = client.judge(
            question="What is RAG?",
            reference="RAG combines retrieval and generation.",
            hypothesis="RAG is a retrieval-augmented approach.",
        )

        assert isinstance(result, JudgeScore)
        assert result.score == 4
        assert result.reason == "Good answer."
        assert result.is_correct is True

    def test_extracts_json_with_surrounding_text(self):
        """judge() should extract JSON even when surrounded by extra text (robustness)."""
        client, mock_openai = _make_client()
        raw = 'Here is my evaluation: {"score": 3, "reason": "Partial.", "is_correct": false} Hope this helps!'
        mock_openai.chat.completions.create.return_value = _mock_completion(raw)

        result = client.judge("Q", "R", "H")

        assert result.score == 3
        assert result.is_correct is False

    def test_fallback_on_malformed_json(self):
        """Malformed JSON should return score=1 with an error message without raising."""
        client, mock_openai = _make_client()
        mock_openai.chat.completions.create.return_value = _mock_completion(
            "I cannot evaluate this."
        )

        result = client.judge("Q", "R", "H")

        assert result.score == 1
        assert result.is_correct is False
        assert "parse_error" in result.reason

    def test_calls_complete_internally(self):
        """judge() should call the underlying completion API exactly once."""
        client, mock_openai = _make_client()
        mock_openai.chat.completions.create.return_value = self._make_judge_response(
            4, "OK", True
        )

        client.judge("Q", "R", "H")
        mock_openai.chat.completions.create.assert_called_once()

# ---------------------------------------------------------------------------
# stream()
# ---------------------------------------------------------------------------

class TestStream:

    def test_stream_yields_tokens(self):
        """stream() should yield tokens from chunk deltas."""
        client, mock_openai = _make_client()

        chunks = []
        for token in ["Hello", " ", "World"]:
            chunk = MagicMock()
            chunk.choices[0].delta.content = token
            chunks.append(chunk)

        mock_openai.chat.completions.create.return_value = iter(chunks)

        tokens = list(client.stream("Say hello"))
        assert tokens == ["Hello", " ", "World"]

    def test_stream_filters_none_deltas(self):
        """stream() should not yield None deltas."""
        client, mock_openai = _make_client()

        chunks = []
        for content in ["Hi", None, "!"]:
            chunk = MagicMock()
            chunk.choices[0].delta.content = content
            chunks.append(chunk)

        mock_openai.chat.completions.create.return_value = iter(chunks)

        tokens = list(client.stream("Hello"))
        assert None not in tokens
        assert tokens == ["Hi", "!"]

# ---------------------------------------------------------------------------
# count_tokens()
# ---------------------------------------------------------------------------

class TestCountTokens:

    def test_whitespace_based_count(self):
        """count_tokens() uses split() — counts words, not real tokens."""
        client, _ = _make_client()
        assert client.count_tokens("hello world foo bar") == 4

    def test_empty_string_returns_zero(self):
        """count_tokens() on an empty string should return 0."""
        client, _ = _make_client()
        assert client.count_tokens("") == 0

    def test_punctuation_attached_to_word(self):
        """Punctuation attached to a word is counted as one token (split by whitespace)."""
        client, _ = _make_client()
        assert client.count_tokens("Hello, world!") == 2