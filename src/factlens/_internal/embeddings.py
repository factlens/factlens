"""Embedding model management with lazy loading and process-level caching.

The embedding model is the most expensive resource in factlens. This module
ensures it is loaded exactly once per process and reused across all scoring
calls. The model is loaded lazily on first use — importing factlens does
not trigger a download or GPU allocation.

Validated models (tested in published research):

    - ``all-mpnet-base-v2`` (default): 768 dims, 109M params.
      Used in arXiv:2602.13224 for DGI calibration and the confabulation
      benchmark. Recommended for production use.
    - ``all-MiniLM-L6-v2``: 384 dims, 22M params.
      Used in arXiv:2512.13771 for SGI experiments. Faster, lower
      memory, suitable when latency matters more than accuracy.

Any sentence-transformers compatible model works, but default thresholds
(SGI_REVIEW, DGI_PASS) were calibrated against the validated models.
Using an unvalidated model without domain-specific calibration may
produce unreliable scores.

Thread safety:
    The global cache uses module-level state. In multi-threaded applications,
    the first thread to call ``get_encoder()`` initializes the model; subsequent
    threads reuse it. The ``SentenceTransformer.encode()`` method is thread-safe.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


# ── Model registry ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ModelInfo:
    """Metadata for a validated embedding model."""

    name: str
    dims: int
    params_m: int
    description: str


VALIDATED_MODELS: dict[str, ModelInfo] = {
    "all-mpnet-base-v2": ModelInfo(
        name="all-mpnet-base-v2",
        dims=768,
        params_m=109,
        description=(
            "Default. Used for DGI calibration (arXiv:2602.13224) "
            "and the confabulation benchmark. Best accuracy."
        ),
    ),
    "all-MiniLM-L6-v2": ModelInfo(
        name="all-MiniLM-L6-v2",
        dims=384,
        params_m=22,
        description=(
            "Used for SGI experiments (arXiv:2512.13771). "
            "5x faster, half the memory. Good for latency-sensitive pipelines."
        ),
    ),
}
"""Models tested in published factlens research.

Any sentence-transformers model works, but these have validated
threshold calibrations and published AUROC numbers.
"""

DEFAULT_MODEL: str = "all-mpnet-base-v2"
"""Default sentence transformer model.

Changed from ``all-MiniLM-L6-v2`` in v2026.4.28. The bundled DGI
reference pairs and thresholds were calibrated with this model.
"""


# ── Module-level cache ──────────────────────────────────────────────────────

_encoder: SentenceTransformer | None = None
_encoder_model_name: str | None = None


def get_encoder(model_name: str = DEFAULT_MODEL) -> Any:
    """Load a sentence transformer model, caching for process lifetime.

    The model is downloaded on first use (cached to ``~/.cache/torch/``
    by sentence-transformers) and kept in memory for the duration of the
    process. Changing ``model_name`` between calls loads the new model
    and replaces the cache.

    Args:
        model_name: HuggingFace model name or local path. Must be
            compatible with the sentence-transformers library.

    Returns:
        A ``SentenceTransformer`` instance ready for ``.encode()``.

    Raises:
        ImportError: If ``sentence-transformers`` is not installed.
    """
    global _encoder, _encoder_model_name

    if _encoder is not None and _encoder_model_name == model_name:
        return _encoder

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        msg = (
            "sentence-transformers is required for factlens scoring. "
            "Install with: pip install factlens"
        )
        raise ImportError(msg) from exc

    if model_name not in VALIDATED_MODELS:
        logger.warning(
            "Model '%s' is not validated. Default thresholds (SGI_REVIEW, DGI_PASS) "
            "were calibrated with validated models (%s). Consider domain-specific "
            "calibration for reliable results. See: factlens.calibrate()",
            model_name,
            ", ".join(VALIDATED_MODELS),
        )

    logger.info("Loading embedding model: %s", model_name)
    _encoder = SentenceTransformer(model_name)
    _encoder_model_name = model_name
    logger.info("Embedding model loaded: %s", model_name)
    return _encoder


def encode_texts(
    texts: list[str],
    model_name: str = DEFAULT_MODEL,
) -> NDArray[np.float32]:
    """Encode a list of texts into embedding vectors.

    Args:
        texts: Strings to encode. Empty strings produce zero vectors.
        model_name: Sentence transformer model to use.

    Returns:
        Array of shape ``(len(texts), embedding_dim)`` with float32 values.
        Embeddings are NOT L2-normalized (raw encoder output).
    """
    encoder = get_encoder(model_name)
    embeddings: NDArray[np.float32] = encoder.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    return embeddings


def reset_cache() -> None:
    """Clear the cached encoder. Useful for testing or memory management."""
    global _encoder, _encoder_model_name
    _encoder = None
    _encoder_model_name = None
