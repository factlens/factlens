"""Semantic Kernel integration for factlens.

Provides a function invocation filter that evaluates function results
for hallucination risk.
"""

from __future__ import annotations

from factlens.integrations.semantic_kernel.filter import FactlensFilter

__all__ = [
    "FactlensFilter",
]
