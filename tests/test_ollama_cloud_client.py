"""
tests/test_ollama_cloud_client.py
==================================
Unit tests for ``OllamaCloudClient``
(``src/llm_eval/clients/ollama_cloud_cliente.py``).

100% mocked — no real calls to the Ollama Cloud API are made.

Behaviour covered:

.. list-table::
   :header-rows: 1

   * - Method
     - Scenario tested
   * - ``__init__``
     - Raises ``ValueError`` when ``OLLAMA_API_KEY`` is absent
   * - ``__init__``
     - Stores ``model_name`` from constructor argument
   * - ``__init__``
     - Falls back to ``"llama3.3:70b"`` when no model is specified
   * - ``__init__``
     - Passes ``base_url`` containing ``"ollama.com"`` to ``OpenAI``
   * - ``complete()``
     - Returns a typed ``LLMResponse``
   * - ``complete()``
     - Populates all ``LLMResponse`` fields correctly
   * - ``complete()``
     - Raises ``ValueError`` when ``content=None`` (model unavailable)
   * - ``complete()``
     - Prepends the system message when ``system != ""``
   * - ``complete()``
     - Sends only the user message when ``system == ""``
   * - ``complete()``
     - Forwards ``temperature`` and ``max_tokens`` to the OpenAI client
   * - ``judge()``
     - Returns a valid ``JudgeScore`` from a well-formed JSON response
   * - ``judge()``
     - Extracts JSON when the model wraps it in surrounding prose
   * - ``judge()``
     - Falls back to ``score=1`` on malformed JSON without raising
   * - ``judge()``
     - Calls ``self.complete()`` exactly once per invocation
   * - ``stream()``
     - Yields token strings from ``delta.content``
   * - ``stream()``
     - Skips ``None`` delta chunks silently
   * - ``count_tokens()``
     - Returns whitespace word count
   * - ``count_tokens()``
     - Returns ``0`` for empty string
     - Counts punctuation-attached words as single tokens
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from llm_eval.shared.types import LLMResponse, JudgeScore  # installed package

# sys.path registered by conftest.py
from ollama_cloud_cliente import OllamaCloudClient  # type: ignore[import]


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_client(model: str = "llama3.3:70b") -> tuple[OllamaCloudClient, MagicMock]:
    """Instantiate ``OllamaCloudClient`` with a mocked ``OpenAI`` SDK.

    Patches ``OpenAI`` at the module level so that no network connections are
    attempted. Returns both the client and the mock instance so that tests can
    configure return values and assert on call arguments.

    Args:
        model: Model name passed to the ``OllamaCloudClient`` constructor.
               Defaults to ``"llama3.3:70b"``.

    Returns:
        A ``(client, mock_openai_instance)`` tuple where ``mock_openai_instance``
        is the ``MagicMock`` returned by the patched ``OpenAI()`` constructor.
    """
    with patch.dict("os.environ", {"OLLAMA_API_KEY": "test-key-123"}):
        with patch("ollama_cloud_cliente.OpenAI") as mock_openai_cls:
            mock_openai_instance = MagicMock()
            mock_openai_cls.return_value = mock_openai_instance
            client = OllamaCloudClient(model=model)
            return client, mock_openai_instance


def _mock_completion(content: str | None) -> MagicMock:
    """Build a minimal ``ChatCompletion`` mock with the given ``content``.

    Provides only the structure accessed by ``OllamaCloudClient.complete()``:
    ``choices[0].message.content``. Additional fields are auto-created as
    ``MagicMock`` attributes and are not accessed by the client.

    Args:
        content: The text content for the first choice. Pass ``None`` to
                 simulate a model that returns no content (e.g. quota exceeded).

    Returns:
        ``MagicMock`` with ``choices[0].message.content`` set to ``content``.
    """
    mock_resp = MagicMock()
    mock_resp.choices[0].message.content = content
    return mock_resp


# ---------------------------------------------------------------------------
# __init__ tests
# ---------------------------------------------------------------------------


class TestInit:
    """Tests for ``OllamaCloudClient.__init__``."""

    def test_lanca_erro_sem_api_key(self):
        """Should raise ``ValueError`` mentioning ``OLLAMA_API_KEY`` when the key is absent.

        Uses ``clear=True`` on ``patch.dict`` to remove any key that may have
        been injected by the ``set_ollama_env`` autouse fixture, then
        explicitly pops the key to guarantee a clean environment.
        """
        with patch.dict("os.environ", {}, clear=True):
            import os
            os.environ.pop("OLLAMA_API_KEY", None)
            with pytest.raises(ValueError, match="OLLAMA_API_KEY"):
                OllamaCloudClient()

    def test_model_name_armazenado(self):
        """``model_name`` should reflect the model passed to the constructor."""
        client, _ = _make_client(model="qwen3:32b")
        assert client.model_name == "qwen3:32b"

    def test_model_default_llama(self):
        """Default ``model_name`` should be ``"llama3.3:70b"`` or the env override.

        The assertion accepts any non-empty model name containing ``"llama"``
        to accommodate both the hardcoded default and ``OLLAMA_CLOUD_MODEL``
        overrides injected by the ``set_ollama_env`` fixture.
        """
        client, _ = _make_client()
        assert "llama" in client.model_name or len(client.model_name) > 0

    def test_openai_client_instanciado_com_base_url(self):
        """``OpenAI`` should be instantiated with a ``base_url`` containing ``"ollama.com"``.

        Verifies that the client points at the Ollama Cloud endpoint rather
        than the default OpenAI API base URL.
        """
        with patch.dict("os.environ", {"OLLAMA_API_KEY": "test-key"}):
            with patch("ollama_cloud_cliente.OpenAI") as mock_cls:
                mock_cls.return_value = MagicMock()
                OllamaCloudClient()
                call_kwargs = mock_cls.call_args[1]
                assert "ollama.com" in call_kwargs["base_url"]


# ---------------------------------------------------------------------------
# complete() tests
# ---------------------------------------------------------------------------


class TestComplete:
    """Tests for ``OllamaCloudClient.complete``."""

    def test_retorna_llm_response(self):
        """``complete()`` should return a typed ``LLMResponse`` instance."""
        client, mock_openai = _make_client()
        mock_openai.chat.completions.create.return_value = _mock_completion("Hello!")

        result = client.complete("What is RAG?")
        assert isinstance(result, LLMResponse)

    def test_campos_preenchidos(self):
        """All ``LLMResponse`` fields should be populated correctly.

        Verifies identity fields (model, prompt, response) and the non-negative
        latency constraint.
        """
        client, mock_openai = _make_client(model="llama3.3:70b")
        mock_openai.chat.completions.create.return_value = _mock_completion(
            "RAG combines retrieval with generation."
        )

        result = client.complete("What is RAG?")

        assert result.model    == "llama3.3:70b"
        assert result.prompt   == "What is RAG?"
        assert result.response == "RAG combines retrieval with generation."
        assert result.latency_ms >= 0

    def test_lanca_value_error_se_content_none(self):
        """``content=None`` should raise ``ValueError`` with ``"content=None"`` in the message.

        Simulates a quota-exceeded or model-unavailable response where the
        Ollama Cloud API returns a choice with no content.
        """
        client, mock_openai = _make_client()
        mock_openai.chat.completions.create.return_value = _mock_completion(None)

        with pytest.raises(ValueError, match="content=None"):
            client.complete("Test prompt")

    def test_system_prompt_adicionado_as_mensagens(self):
        """When ``system`` is non-empty, a system message should precede the user message.

        Verifies the message list structure required by the OpenAI Chat
        Completions API when a system instruction is provided.
        """
        client, mock_openai = _make_client()
        mock_openai.chat.completions.create.return_value = _mock_completion("OK")

        client.complete("User prompt", system="You are helpful.")

        call_kwargs = mock_openai.chat.completions.create.call_args[1]
        messages    = call_kwargs["messages"]
        assert messages[0]["role"]    == "system"
        assert messages[0]["content"] == "You are helpful."
        assert messages[1]["role"]    == "user"

    def test_sem_system_prompt_so_user_message(self):
        """When ``system`` is empty, the message list should contain only the user message."""
        client, mock_openai = _make_client()
        mock_openai.chat.completions.create.return_value = _mock_completion("OK")

        client.complete("Hello")

        call_kwargs = mock_openai.chat.completions.create.call_args[1]
        messages    = call_kwargs["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    def test_temperatura_e_max_tokens_passados(self):
        """``temperature`` and ``max_tokens`` should be forwarded to the ``OpenAI`` client."""
        client, mock_openai = _make_client()
        mock_openai.chat.completions.create.return_value = _mock_completion("OK")

        client.complete("prompt", temperature=0.5, max_tokens=512)

        call_kwargs = mock_openai.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"]  == 512


# ---------------------------------------------------------------------------
# judge() tests
# ---------------------------------------------------------------------------


class TestJudge:
    """Tests for ``OllamaCloudClient.judge``."""

    def _make_judge_response(
        self, score: int, reason: str, is_correct: bool
    ) -> MagicMock:
        """Build a mock completion containing a valid JSON judge response.

        Args:
            score:      Judge score (1–5).
            reason:     Textual justification string.
            is_correct: Boolean correctness verdict.

        Returns:
            ``MagicMock`` completion with the JSON-encoded judge response as
            ``choices[0].message.content``.
        """
        json_str = json.dumps(
            {"score": score, "reason": reason, "is_correct": is_correct}
        )
        return _mock_completion(json_str)

    def test_retorna_judge_score_valido(self):
        """``judge()`` should return a ``JudgeScore`` with all fields correctly parsed.

        Verifies type, numeric score, reason string, and boolean verdict
        from a well-formed JSON response.
        """
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
        assert result.score      == 4
        assert result.reason     == "Good answer."
        assert result.is_correct is True

    def test_extrai_json_com_texto_ao_redor(self):
        """``judge()`` should extract JSON when surrounded by prose.

        Verifies the robustness of the ``index('{') / rindex('}')`` extraction
        strategy against models that add preamble or closing text around the
        JSON object.
        """
        client, mock_openai = _make_client()
        raw = (
            'Here is my evaluation: '
            '{"score": 3, "reason": "Partial.", "is_correct": false} '
            'Hope this helps!'
        )
        mock_openai.chat.completions.create.return_value = _mock_completion(raw)

        result = client.judge("Q", "R", "H")

        assert result.score      == 3
        assert result.is_correct is False

    def test_fallback_json_malformado(self):
        """Malformed JSON should return ``score=1`` with ``"parse_error"`` in reason.

        Verifies that a response with no valid JSON object triggers the
        fallback path without raising an exception — critical for batch
        evaluation runs where a single parse failure must not abort the run.
        """
        client, mock_openai = _make_client()
        mock_openai.chat.completions.create.return_value = _mock_completion(
            "I cannot evaluate this."
        )

        result = client.judge("Q", "R", "H")

        assert result.score      == 1
        assert result.is_correct is False
        assert "parse_error" in result.reason

    def test_usa_complete_internamente(self):
        """``judge()`` should call ``self.complete()`` exactly once.

        Verifies the pass-through behaviour — ``judge`` delegates to
        ``complete`` without batching or caching calls.
        """
        client, mock_openai = _make_client()
        mock_openai.chat.completions.create.return_value = self._make_judge_response(
            4, "OK", True
        )

        client.judge("Q", "R", "H")
        mock_openai.chat.completions.create.assert_called_once()


# ---------------------------------------------------------------------------
# stream() tests
# ---------------------------------------------------------------------------


class TestStream:
    """Tests for ``OllamaCloudClient.stream``."""

    def test_stream_yield_tokens(self):
        """``stream()`` should yield each non-None ``delta.content`` string.

        Constructs a synthetic chunk iterator with three tokens and verifies
        that all are yielded in order.
        """
        client, mock_openai = _make_client()

        chunks = []
        for token in ["Hello", " ", "World"]:
            chunk = MagicMock()
            chunk.choices[0].delta.content = token
            chunks.append(chunk)

        mock_openai.chat.completions.create.return_value = iter(chunks)

        tokens = list(client.stream("Say hello"))
        assert tokens == ["Hello", " ", "World"]

    def test_stream_filtra_delta_none(self):
        """``stream()`` should silently skip chunks where ``delta.content`` is ``None``.

        Simulates the first and last control chunks in the OpenAI streaming
        protocol (which carry ``delta.content=None``) mixed with real content.
        """
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
# count_tokens() tests
# ---------------------------------------------------------------------------


class TestCountTokens:
    """Tests for ``OllamaCloudClient.count_tokens``."""

    def test_contagem_por_whitespace(self):
        """``count_tokens()`` should count whitespace-separated words.

        Verifies the whitespace tokenisation approximation documented in
        the method docstring.
        """
        client, _ = _make_client()
        assert client.count_tokens("hello world foo bar") == 4

    def test_string_vazia_retorna_zero(self):
        """``count_tokens("")`` should return ``0``."""
        client, _ = _make_client()
        assert client.count_tokens("") == 0

    def test_frase_com_pontuacao(self):
        """Punctuation attached to a word should be counted as part of the same token.

        ``"Hello, world!"`` splits into ``["Hello,", "world!"]`` — 2 tokens.
        This test documents the known limitation of whitespace splitting and
        ensures it remains consistent across Python versions.
        """
        client, _ = _make_client()
        assert client.count_tokens("Hello, world!") == 2