"""Anthropic Claude provider with automatic factlens hallucination scoring.

Wraps the Anthropic Python SDK and evaluates every response using SGI
(when context is provided) or DGI (context-free).

Example:
    >>> from factlens.providers.anthropic import FactlensAnthropic
    >>> llm = FactlensAnthropic(api_key="sk-ant-...")
    >>> resp = llm.chat("What is X?", context="X is defined as Y.")
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
    import anthropic

logger = logging.getLogger(__name__)


def _get_anthropic_client(api_key: str) -> anthropic.Anthropic:
    """Lazily import and instantiate the Anthropic client.

    Args:
        api_key: Anthropic API key.

    Returns:
        An authenticated ``anthropic.Anthropic`` client instance.

    Raises:
        ImportError: If the ``anthropic`` package is not installed.
    """
    try:
        import anthropic as _anthropic
    except ImportError as exc:
        msg = (
            "The 'anthropic' package is required for FactlensAnthropic. "
            "Install it with: pip install 'factlens[anthropic]'"
        )
        raise ImportError(msg) from exc
    return _anthropic.Anthropic(api_key=api_key)


class FactlensAnthropic:
    """Anthropic Claude provider with built-in factlens scoring.

    Wraps the Anthropic messages API and automatically evaluates
    each response for hallucination risk.

    Args:
        api_key: Anthropic API key.
        model: Claude model to use for generation. Defaults to
            ``"claude-sonnet-4-20250514"``.
        factlens_model: Sentence-transformer model for factlens scoring.
            Defaults to ``DEFAULT_MODEL``.

    Example:
        >>> llm = FactlensAnthropic(api_key="sk-ant-...")
        >>> resp = llm.chat("Summarize this.", context="Source text here.")
        >>> print(resp.factlens_score.explanation)
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        factlens_model: str = DEFAULT_MODEL,
    ) -> None:
        self._client = _get_anthropic_client(api_key)
        self._model = model
        self._factlens_model = factlens_model

    def chat(
        self,
        prompt: str,
        context: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send a message to Claude and score the response.

        Args:
            prompt: The user message content.
            context: Optional source document. When provided, SGI scoring
                is used; otherwise DGI scoring is applied.
            **kwargs: Additional keyword arguments forwarded to the
                Anthropic ``messages.create`` call.

        Returns:
            LLMResponse containing the generated text, model identifier,
            usage metadata, and a factlens hallucination score.

        Raises:
            anthropic.APIError: If the API call fails.

        Example:
            >>> llm = FactlensAnthropic(api_key="sk-ant-...")
            >>> resp = llm.chat("Explain photosynthesis.")
            >>> resp.text
            'Photosynthesis is the process by which...'
        """
        messages: list[dict[str, str]] = [{"role": "user", "content": prompt}]

        logger.debug("Calling Anthropic model=%s prompt_len=%d", self._model, len(prompt))

        max_tokens = kwargs.pop("max_tokens", 4096)

        message = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            messages=messages,
            **kwargs,
        )

        text = ""
        for block in message.content:
            if hasattr(block, "text"):
                text += block.text

        usage: dict[str, Any] = {
            "input_tokens": message.usage.input_tokens,
            "output_tokens": message.usage.output_tokens,
        }

        score = evaluate(
            question=prompt,
            response=text,
            context=context,
            model=self._factlens_model,
        )

        logger.info(
            "Anthropic response scored: method=%s value=%.3f flagged=%s",
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
