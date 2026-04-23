# /// script
# requires-python = ">=3.10"
# dependencies = ["factlens[openai]"]
# ///
"""OpenAI provider with factlens hallucination guard.

Requires: ``pip install factlens[openai]``

Uses FactlensOpenAI to wrap OpenAI chat completions with automatic
hallucination scoring. Every response includes a factlens score.
"""

import os

from factlens.providers.openai import FactlensOpenAI

if __name__ == "__main__":
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("Set OPENAI_API_KEY environment variable to run this example.")
        raise SystemExit(1)

    llm = FactlensOpenAI(api_key=api_key, model="gpt-4o")

    # With context — uses SGI scoring
    print("=== With Context (SGI) ===\n")
    context = (
        "The Eiffel Tower was built for the 1889 World's Fair. "
        "It stands 330 metres tall and is located on the Champ de Mars in Paris."
    )
    resp = llm.chat("How tall is the Eiffel Tower?", context=context)
    print(f"Response: {resp.text}")
    print(f"Score:    {resp.factlens_score.method.upper()} = {resp.factlens_score.value:.3f}")
    print(f"Flagged:  {resp.factlens_score.flagged}")
    print(f"Explain:  {resp.factlens_score.explanation}\n")

    # Without context — uses DGI scoring
    print("=== Without Context (DGI) ===\n")
    resp = llm.chat("What is the speed of sound?")
    print(f"Response: {resp.text}")
    print(f"Score:    {resp.factlens_score.method.upper()} = {resp.factlens_score.value:.3f}")
    print(f"Flagged:  {resp.factlens_score.flagged}")
    print(f"Explain:  {resp.factlens_score.explanation}")
