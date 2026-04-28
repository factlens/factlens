# API Reference

This page provides the complete API reference for factlens. All public classes and functions are documented with their signatures, parameters, return types, and examples.

For auto-generated documentation from source docstrings, ensure `mkdocstrings` is configured in your MkDocs build.

## Core Functions

### compute_sgi

::: factlens.sgi.compute_sgi

### compute_dgi

::: factlens.dgi.compute_dgi

### evaluate

::: factlens.evaluate.evaluate

### evaluate_batch

::: factlens.evaluate.evaluate_batch

### calibrate

::: factlens.calibrate.calibrate

## Core Classes

### SGI

::: factlens.sgi.SGI

### DGI

::: factlens.dgi.DGI

## Result Types

### SGIResult

::: factlens.score.SGIResult

### DGIResult

::: factlens.score.DGIResult

### FactlensScore

::: factlens.score.FactlensScore

### CalibrationResult

::: factlens.calibrate.CalibrationResult

## Providers

### FactlensOpenAI

::: factlens.providers.openai.FactlensOpenAI

### FactlensAnthropic

::: factlens.providers.anthropic.FactlensAnthropic

### FactlensGemini

::: factlens.providers.google.FactlensGemini

## Integrations

### FactlensEvaluator (LangChain)

::: factlens.integrations.langchain.evaluator.FactlensEvaluator

### FactlensCallback (LangChain)

::: factlens.integrations.langchain.callback.FactlensCallback

### FactlensTool (CrewAI)

::: factlens.integrations.crewai.tool.FactlensTool

### FactlensFilter (Semantic Kernel)

::: factlens.integrations.semantic_kernel.filter.FactlensFilter

### FactlensChecker (AutoGen)

::: factlens.integrations.autogen.checker.FactlensChecker

## Internal Modules

!!! note "Internal API"
    The following modules are internal implementation details. They are documented here for completeness but are not part of the public API and may change without notice.

### Geometry Primitives

::: factlens._internal.geometry

### Thresholds

::: factlens._internal.thresholds

## Constants

| Constant | Value | Module | Description |
|---|---|---|---|
| `SGI_STRONG_PASS` | 1.20 | `factlens._internal.thresholds` | SGI strong pass threshold |
| `SGI_REVIEW` | 0.95 | `factlens._internal.thresholds` | SGI review/flag threshold |
| `DGI_PASS` | 0.30 | `factlens._internal.thresholds` | DGI pass threshold |
| `DEFAULT_MODEL` | `"all-mpnet-base-v2"` | `factlens._internal.embeddings` | Default sentence-transformer model |

## Type Summary

| Type | Description | Key fields |
|---|---|---|
| `SGIResult` | SGI computation result | `value`, `normalized`, `flagged`, `q_dist`, `ctx_dist` |
| `DGIResult` | DGI computation result | `value`, `normalized`, `flagged` |
| `FactlensScore` | Unified evaluation result | `value`, `normalized`, `flagged`, `method`, `explanation`, `detail` |
| `CalibrationResult` | DGI calibration output | `model`, `n_pairs`, `embedding_dim`, `mu_hat`, `concentration` |
| `LLMResponse` | Provider response wrapper | `text`, `model`, `usage`, `factlens_score` |
