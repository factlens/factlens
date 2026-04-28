"""CrewAI tool for agent self-verification via factlens.

Allows CrewAI agents to verify their own outputs for hallucination
risk before presenting them to users or other agents.

Example:
    >>> from factlens.integrations.crewai import FactlensTool
    >>> tool = FactlensTool()
    >>> result = tool._run(
    ...     question="What is X?",
    ...     response="X is Y.",
    ...     context="According to the docs, X is Y.",
    ... )
    >>> print(result)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from factlens._internal.embeddings import DEFAULT_MODEL
from factlens.evaluate import evaluate

if TYPE_CHECKING:
    from factlens.score import FactlensScore

logger = logging.getLogger(__name__)


def _validate_crewai_available() -> None:
    """Verify that the crewai package is importable.

    Raises:
        ImportError: If the ``crewai`` package is not installed.
    """
    try:
        import crewai  # noqa: F401
    except ImportError as exc:
        msg = (
            "The 'crewai' package is required for FactlensTool. "
            "Install it with: pip install 'factlens[crewai]'"
        )
        raise ImportError(msg) from exc


class FactlensTool:
    """CrewAI tool for verifying LLM outputs using factlens.

    Extends the CrewAI tool pattern to let agents self-verify their
    outputs. The tool evaluates a question-response pair (with optional
    context) and returns a human-readable verification summary.

    Args:
        name: Tool name visible to the agent. Defaults to
            ``"factlens_verify"``.
        description: Tool description for agent tool selection.
        factlens_model: Sentence-transformer model for factlens scoring.
            Defaults to ``DEFAULT_MODEL``.

    Example:
        >>> from factlens.integrations.crewai import FactlensTool
        >>> tool = FactlensTool()
        >>> # Agent uses the tool to verify its own output
        >>> result = tool._run(
        ...     question="What causes rain?",
        ...     response="Rain is caused by condensation.",
        ...     context="Water cycle: evaporation, condensation, precipitation.",
        ... )
        >>> "PASS" in result or "FLAGGED" in result
        True
    """

    name: str = "factlens_verify"
    description: str = (
        "Verify an LLM response for hallucination risk. "
        "Provide the question, response, and optionally the source context. "
        "Returns a verification result with a grounding score."
    )

    def __init__(
        self,
        name: str = "factlens_verify",
        description: str | None = None,
        factlens_model: str = DEFAULT_MODEL,
    ) -> None:
        self.name = name
        if description is not None:
            self.description = description
        self._factlens_model = factlens_model

    def _run(
        self,
        question: str,
        response: str,
        context: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Evaluate a response for hallucination risk.

        Args:
            question: The original question or prompt.
            response: The LLM-generated response to verify.
            context: Optional source document. When provided, SGI scoring
                is used; otherwise DGI scoring is applied.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            A formatted string containing the verification result,
            including method, score, status, and explanation.

        Example:
            >>> tool = FactlensTool()
            >>> result = tool._run("What is 2+2?", "2+2 is 4.")
            >>> isinstance(result, str)
            True
        """
        logger.debug(
            "FactlensTool._run question_len=%d response_len=%d context=%s",
            len(question),
            len(response),
            "provided" if context else "none",
        )

        score: FactlensScore = evaluate(
            question=question,
            response=response,
            context=context,
            model=self._factlens_model,
        )

        status = "FLAGGED" if score.flagged else "PASS"

        result = (
            f"Factlens Verification Result\n"
            f"----------------------------\n"
            f"Method: {score.method.upper()}\n"
            f"Score: {score.value:.3f} (normalized: {score.normalized:.3f})\n"
            f"Status: {status}\n"
            f"Explanation: {score.explanation}\n"
        )

        if score.flagged:
            result += (
                "\nRecommendation: This response may contain hallucinated content. "
                "Consider revising with verified sources.\n"
            )

        logger.info(
            "FactlensTool result: method=%s value=%.3f status=%s",
            score.method,
            score.value,
            status,
        )

        return result
