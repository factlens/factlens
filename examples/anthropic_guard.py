# /// script
# requires-python = ">=3.10"
# dependencies = ["factlens[anthropic]"]
# ///
"""Anthropic provider with factlens hallucination guard.

Requires: ``pip install factlens[anthropic]``

Demonstrates wrapping Anthropic's Claude API with factlens scoring.
Each response is automatically evaluated for hallucination risk.
"""

import os

from factlens.evaluate import evaluate


def anthropic_with_factlens(question: str, context: str | None = None) -> None:
    """Call Anthropic Claude and score the response with factlens."""
    try:
        import anthropic
    except ImportError:
        print("Install anthropic: pip install factlens[anthropic]")
        raise SystemExit(1)

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("Set ANTHROPIC_API_KEY environment variable.")
        raise SystemExit(1)

    client = anthropic.Anthropic(api_key=api_key)

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=512,
        messages=[{"role": "user", "content": question}],
    )

    response_text = message.content[0].text

    score = evaluate(
        question=question,
        response=response_text,
        context=context,
    )

    print(f"Question: {question}")
    print(f"Response: {response_text[:200]}...")
    print(f"Method:   {score.method.upper()}")
    print(f"Score:    {score.value:.3f}")
    print(f"Flagged:  {score.flagged}")
    print(f"Explain:  {score.explanation}")
    print()


if __name__ == "__main__":
    print("=== Anthropic + Factlens ===\n")

    # With context (SGI)
    anthropic_with_factlens(
        question="What is CRISPR?",
        context=(
            "CRISPR-Cas9 is a genome editing tool adapted from a bacterial "
            "immune defense system. It allows precise DNA modifications."
        ),
    )

    # Without context (DGI)
    anthropic_with_factlens(
        question="What causes auroras?",
    )
