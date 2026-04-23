"""LLM provider wrappers with built-in factlens scoring.

Each provider wraps a third-party LLM SDK and automatically evaluates
every response for hallucination risk using SGI (when context is provided)
or DGI (context-free).
"""

from __future__ import annotations

from factlens.providers._base import BaseLLMProvider, LLMResponse

__all__ = [
    "BaseLLMProvider",
    "LLMResponse",
]
