"""Base types and protocol for factlens LLM providers.

Defines the ``LLMResponse`` dataclass returned by all providers and the
``BaseLLMProvider`` protocol that every concrete provider must satisfy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from factlens.score import FactlensScore


@dataclass(slots=True)
class LLMResponse:
    """Unified response container for all LLM provider calls.

    Attributes:
        text: The generated text content from the LLM.
        model: The model identifier used for generation.
        usage: Provider-specific usage metadata (tokens, cost, etc.).
        factlens_score: Optional factlens evaluation result attached
            after hallucination scoring.

    Examples:
        >>> resp = LLMResponse(text="Hello!", model="gpt-4o", usage={})
        >>> resp.factlens_score is None
        True
    """

    text: str
    model: str
    usage: dict[str, Any] = field(default_factory=dict)
    factlens_score: FactlensScore | None = None


@runtime_checkable
class BaseLLMProvider(Protocol):
    """Protocol defining the interface all factlens providers implement.

    Providers wrap third-party LLM SDKs and automatically attach a
    ``FactlensScore`` to every response, enabling inline hallucination
    detection without changing application code.

    Examples:
        >>> def use_provider(provider: BaseLLMProvider) -> None:
        ...     resp = provider.complete("Summarize this.", context="Source text.")
        ...     if resp.factlens_score and resp.factlens_score.flagged:
        ...         print("Review recommended!")
    """

    def complete(
        self,
        prompt: str,
        context: str | None = None,
    ) -> LLMResponse:
        """Generate a completion for the given prompt.

        Args:
            prompt: The user prompt or instruction.
            context: Optional source document for grounded evaluation.
                When provided, SGI scoring is used; otherwise DGI.

        Returns:
            LLMResponse with generated text and factlens score.
        """
        ...

    def chat(
        self,
        messages: list[dict[str, str]],
        context: str | None = None,
    ) -> LLMResponse:
        """Generate a chat completion from a message history.

        Args:
            messages: List of message dicts with ``role`` and ``content`` keys.
            context: Optional source document for grounded evaluation.
                When provided, SGI scoring is used; otherwise DGI.

        Returns:
            LLMResponse with generated text and factlens score.
        """
        ...
