"""OpenAI provider with automatic factlens hallucination scoring.

Wraps the OpenAI Python SDK and evaluates every response using SGI
(when context is provided) or DGI (context-free).

Example:
    >>> from factlens.providers.openai import FactlensOpenAI
    >>> llm = FactlensOpenAI(api_key="sk-...")
    >>> resp = llm.chat("What is the capital of France?", context="Paris is the capital.")
    >>> resp.factlens_score.flagged
    False
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from factlens._internal.embeddings import DEFAULT_MODEL
from factlens.evaluate import evaluate
from factlens.providers._base import LLMResponse

if TYPE_CHECKING:
    import openai

logger = logging.getLogger(__name__)


def _get_openai_client(api_key: str) -> openai.OpenAI:
    """Lazily import and instantiate the OpenAI client.

    Args:
        api_key: OpenAI API key.

    Returns:
        An authenticated ``openai.OpenAI`` client instance.

    Raises:
        ImportError: If the ``openai`` package is not installed.
    """
    try:
        import openai as _openai
    except ImportError as exc:
        msg = (
            "The 'openai' package is required for FactlensOpenAI. "
            "Install it with: pip install 'factlens[openai]'"
        )
        raise ImportError(msg) from exc
    return _openai.OpenAI(api_key=api_key)


class FactlensOpenAI:
    """OpenAI LLM provider with built-in factlens scoring.

    Wraps the OpenAI chat completions API and automatically evaluates
    each response for hallucination risk.

    Args:
        api_key: OpenAI API key.
        model: Chat model to use for generation. Defaults to ``"gpt-4o"``.
        factlens_model: Sentence-transformer model for factlens scoring.
            Defaults to ``DEFAULT_MODEL``.

    Examples:
        >>> llm = FactlensOpenAI(api_key="sk-...")
        >>> resp = llm.chat("Summarize this document.", context="The document text.")
        >>> print(resp.factlens_score.explanation)
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        factlens_model: str = DEFAULT_MODEL,
    ) -> None:
        self._client = _get_openai_client(api_key)
        self._model = model
        self._factlens_model = factlens_model

    def chat(
        self,
        prompt: str,
        context: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send a chat completion request and score the response.

        Args:
            prompt: The user message content.
            context: Optional source document. When provided, SGI scoring
                is used; otherwise DGI scoring is applied.
            **kwargs: Additional keyword arguments forwarded to the
                OpenAI ``chat.completions.create`` call.

        Returns:
            LLMResponse containing the generated text, model identifier,
            usage metadata, and a factlens hallucination score.

        Raises:
            openai.OpenAIError: If the API call fails.

        Examples:
            >>> llm = FactlensOpenAI(api_key="sk-...")
            >>> resp = llm.chat("What causes tides?")
            >>> resp.text
            'Tides are primarily caused by...'
        """
        messages: list[dict[str, str]] = [{"role": "user", "content": prompt}]

        logger.debug("Calling OpenAI model=%s prompt_len=%d", self._model, len(prompt))

        completion = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            **kwargs,
        )

        choice = completion.choices[0]
        text = choice.message.content or ""

        usage: dict[str, Any] = {}
        if completion.usage is not None:
            usage = {
                "prompt_tokens": completion.usage.prompt_tokens,
                "completion_tokens": completion.usage.completion_tokens,
                "total_tokens": completion.usage.total_tokens,
            }

        score = evaluate(
            question=prompt,
            response=text,
            context=context,
            model=self._factlens_model,
        )

        logger.info(
            "OpenAI response scored: method=%s value=%.3f flagged=%s",
            score.method,
            score.value,
            score.flagged,
        )

        return LLMResponse(
            text=text,
            model=self._model,
            usage=usage,
            factlens_score=score,
        )

    def complete(
        self,
        prompt: str,
        context: str | None = None,
    ) -> LLMResponse:
        """Generate a completion for the given prompt.

        Convenience method that delegates to :meth:`chat`.

        Args:
            prompt: The user prompt or instruction.
            context: Optional source document for grounded evaluation.

        Returns:
            LLMResponse with generated text and factlens score.
        """
        return self.chat(prompt, context=context)
