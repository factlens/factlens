"""AutoGen integration for factlens.

Provides a reply checker that evaluates agent messages for hallucination risk.
"""

from __future__ import annotations

from factlens.integrations.autogen.checker import FactlensChecker

__all__ = [
    "FactlensChecker",
]
