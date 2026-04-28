# Contributing to factlens

factlens is a research-driven project. Contributions that extend its reach without compromising the geometric core are welcome.

## Architecture

The codebase has a deliberate structure:

```
src/factlens/
├── _internal/          # Geometric engine (embeddings, geometry, thresholds)
├── sgi.py              # Semantic Grounding Index (context-required)
├── dgi.py              # Directional Grounding Index (context-free)
├── evaluate.py         # High-level API (auto-selects SGI or DGI)
├── calibrate.py        # Domain-specific calibration
├── score.py            # Result types
├── providers/          # LLM provider connectors (OpenAI, Anthropic, Google)
├── integrations/       # Framework integrations (LangChain, CrewAI, etc.)
├── cli/                # Command-line interface
└── data/               # Bundled calibration dataset
```

The `_internal/` directory is the geometric engine. It should remain small, deterministic, and dependency-light. Everything outside `_internal/` builds on it.

## Where contributions help most

### 1. Embedding backends

factlens currently uses sentence-transformers. Other backends would expand reach:

- **OpenAI embeddings** (`text-embedding-3-small`, `text-embedding-3-large`)
- **Cohere Embed**
- **Voyage AI**
- **Local GGUF models** via llama-cpp-python

Each backend needs: an `encode_texts()` implementation, model registration in `VALIDATED_MODELS` (after testing), and calibration data showing AUROC on the confabulation benchmark.

### 2. LLM provider connectors

Connectors wrap LLM API calls with automatic factlens scoring. Current: OpenAI, Anthropic, Google. Wanted:

- **Mistral**
- **Groq**
- **Together AI**
- **AWS Bedrock**
- **Azure OpenAI**

A connector lives in `src/factlens/providers/` and must: accept the same arguments as the native SDK, inject scoring transparently, and return results with `.factlens_score` attached.

### 3. Framework integrations

Integrations plug factlens into existing LLM frameworks. Current: LangChain, CrewAI, Semantic Kernel, AutoGen. Wanted:

- **LlamaIndex**
- **Haystack**
- **DSPy**
- **Instructor**

An integration lives in `src/factlens/integrations/` and should follow the framework's extension pattern (callbacks, evaluators, tools, filters).

### 4. Domain calibration datasets

DGI accuracy depends on calibration quality. Generic calibration achieves AUROC ~0.76. Domain-specific calibration reaches 0.90-0.99. Contributions of verified (question, response) pairs in specific domains are valuable:

- Format: CSV with columns `domain`, `question`, `grounded_response`
- Minimum 30 pairs per domain
- Each response must be factually verified by a domain expert
- License: CC BY 4.0 (to match the benchmark)

### 5. Geometric methods

New scoring methods beyond SGI and DGI. This is the hardest contribution — it requires understanding the embedding geometry. If you have a paper or a novel geometric approach to grounding verification, open an issue to discuss before implementing.

## Development setup

```bash
git clone https://github.com/factlens/factlens.git
cd factlens
pip install -e ".[dev,docs]"
```

### Running checks

```bash
ruff check src/ tests/       # Lint
ruff format src/ tests/      # Format
mypy src/factlens/           # Type check
pytest                       # Tests (fast, no model loading)
pytest -m slow               # Tests that load embedding models
```

### Code standards

- **Type hints everywhere.** `mypy --strict` must pass.
- **Google-style docstrings** with Args/Returns/Raises.
- **No `Any` in public APIs.** Internal code may use it sparingly.
- Tests go in `tests/` mirroring the `src/` structure.

## Pull request process

1. **Open an issue first** for anything beyond a typo fix. This avoids wasted effort.
2. **One concern per PR.** A new connector is one PR. A new integration is one PR.
3. **Include tests.** New code without tests will not be merged.
4. **Run the full check suite** before opening the PR.
5. **Update docstrings** if you change public API signatures.

## What we don't accept

- Changes to `_internal/geometry.py` or `_internal/thresholds.py` without a paper or empirical evidence.
- Dependencies that pull in large frameworks (PyTorch is already the ceiling).
- "Improvements" that make scores non-deterministic.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
