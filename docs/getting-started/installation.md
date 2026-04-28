# Installation

## Requirements

- Python 3.10 or later
- A sentence-transformer model (downloaded automatically on first use)

## Install from PyPI

```bash
pip install factlens
```

This installs the core library with SGI, DGI, calibration, and CLI support. The default embedding model (`all-mpnet-base-v2`) is downloaded automatically on first use via the `sentence-transformers` package.

## Optional Extras

factlens uses optional dependency groups to keep the core install lightweight. Install only what you need:

=== "Provider extras"

    ```bash
    pip install "factlens[openai]"       # FactlensOpenAI provider
    pip install "factlens[anthropic]"    # FactlensAnthropic provider
    pip install "factlens[google]"       # FactlensGemini provider
    ```

=== "Integration extras"

    ```bash
    pip install "factlens[langchain]"        # LangChain evaluator + callback
    pip install "factlens[crewai]"           # CrewAI tool
    pip install "factlens[semantic-kernel]"  # Semantic Kernel filter
    pip install "factlens[autogen]"          # AutoGen checker
    ```

=== "Everything"

    ```bash
    pip install "factlens[all]"
    ```

## Install from Source

```bash
git clone https://github.com/factlens/factlens.git
cd factlens
pip install -e ".[dev]"
```

The editable install (`-e`) is recommended for development --- changes to source files take effect immediately without reinstalling.

## Verify Installation

```python
import factlens
print(factlens.__version__)  # e.g., 2026.4.22
```

Run a quick smoke test:

```python
from factlens import compute_sgi

result = compute_sgi(
    question="What is the capital of France?",
    context="France is in Western Europe. Its capital is Paris.",
    response="The capital of France is Paris.",
)
print(result.flagged)      # False
print(result.value)        # ~1.2 (varies by model)
print(result.explanation)  # Human-readable interpretation
```

!!! note "First-run model download"
    The first call to any scoring function triggers a one-time download of the `all-mpnet-base-v2` sentence-transformer model (~420 MB). Subsequent calls use the cached model.

## Embedding Model Selection

factlens defaults to `all-mpnet-base-v2` (768 dimensions, ~420 MB). You can use any sentence-transformer model by passing the `model` parameter:

```python
from factlens import compute_sgi

result = compute_sgi(
    question="...",
    context="...",
    response="...",
    model="all-mpnet-base-v2",  # 768 dimensions, higher quality
)
```

!!! warning "Model consistency"
    All scoring and calibration operations must use the same embedding model. Mixing models produces meaningless scores because the geometry of the embedding space differs between models.

## System Requirements

| Component | Minimum | Recommended |
|---|---|---|
| RAM | 1 GB | 4 GB |
| Disk | 200 MB (model cache) | 500 MB |
| GPU | Not required | Not required |
| Python | 3.10 | 3.12+ |

factlens runs entirely on CPU. The sentence-transformer inference is fast enough (~5ms per embedding) that GPU acceleration is unnecessary for typical workloads.
