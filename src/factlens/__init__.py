"""Factlens — A geometric lens on factual accuracy.

Deterministic LLM hallucination detection via embedding geometry.
No second LLM. Auditable. EU AI Act compliant.

Quick start::

    >>> from factlens import compute_sgi, compute_dgi, evaluate
    >>>
    >>> # With context (RAG verification) — uses SGI
    >>> result = compute_sgi(
    ...     question="What is the capital of France?",
    ...     context="France is in Western Europe. Its capital is Paris.",
    ...     response="The capital of France is Paris.",
    ... )
    >>> result.flagged
    False
    >>>
    >>> # Without context — uses DGI
    >>> result = compute_dgi(
    ...     question="What causes seasons?",
    ...     response="Seasons are caused by Earth's 23.5-degree axial tilt.",
    ... )
    >>> result.flagged
    False
    >>>
    >>> # Auto-select method
    >>> score = evaluate(question="Q?", response="A.", context="Source.")
    >>> score.method
    'sgi'

Embedding models:

    factlens defaults to ``all-mpnet-base-v2`` (768 dims), which was used
    for DGI calibration and the confabulation benchmark. For faster scoring,
    pass ``model="all-MiniLM-L6-v2"`` (384 dims, ~5x faster).

    Any sentence-transformers model works. See ``factlens.VALIDATED_MODELS``
    for models with published AUROC numbers.

References:
    Marin (2025). Semantic Grounding Index. arXiv:2512.13771.
    Marin (2026). A Geometric Taxonomy of Hallucinations. arXiv:2602.13224.
    Marin (2026). Rotational Dynamics of Factual Constraint Processing. arXiv:2603.13259.
"""

from factlens._internal.embeddings import DEFAULT_MODEL, VALIDATED_MODELS, ModelInfo, reset_cache
from factlens._version import __version__
from factlens.calibrate import CalibrationResult, calibrate
from factlens.dgi import DGI, compute_dgi, reset_calibration_cache
from factlens.evaluate import evaluate, evaluate_batch
from factlens.score import DGIResult, FactlensScore, SGIResult
from factlens.sgi import SGI, compute_sgi

__all__ = [
    "DEFAULT_MODEL",
    "DGI",
    "SGI",
    "VALIDATED_MODELS",
    "CalibrationResult",
    "DGIResult",
    "FactlensScore",
    "ModelInfo",
    "SGIResult",
    "__version__",
    "calibrate",
    "compute_dgi",
    "compute_sgi",
    "evaluate",
    "evaluate_batch",
    "reset_cache",
    "reset_calibration_cache",
]
