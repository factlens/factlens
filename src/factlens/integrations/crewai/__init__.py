"""CrewAI integration for factlens.

Provides a tool that CrewAI agents can use to self-verify their outputs.
"""

from __future__ import annotations

from factlens.integrations.crewai.tool import FactlensTool

__all__ = [
    "FactlensTool",
]
