"""Specialty-aware filtering (Section 5.2 / Algorithm 1).

The reading clinician's specialty is represented by a single centroid embedding
``B`` (mean of the canonical descriptors of that specialty's seed concepts).
Each candidate clinical feature ``A`` is scored by cosine similarity to ``B``;
features above threshold ``tau`` are kept and the top-``K`` are returned, ranked
by descending similarity. This is a direct, dependency-light implementation of
Algorithm 1 in the paper (the embedding function is injected so the same logic
works with MedGemma embeddings or any other encoder).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

DEFAULT_TAU = 0.75  # paper's selected operating point (Table 4)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """cos(theta) = (A . B) / (||A|| ||B||).  Returns 0.0 if either is a zero vector."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def build_specialty_embedding(concept_embeddings: Sequence[np.ndarray]) -> np.ndarray:
    """Centroid (mean) of the seed-concept embeddings for a specialty -> vector B."""
    mat = np.asarray(concept_embeddings, dtype=float)
    if mat.ndim != 2 or mat.shape[0] == 0:
        raise ValueError("expected a non-empty 2-D array of concept embeddings")
    return mat.mean(axis=0)


@dataclass
class ScoredFeature:
    feature: object
    similarity: float


def rank_and_select(
    features: Sequence[object],
    specialty_vec: np.ndarray,
    embed: Callable[[object], np.ndarray],
    tau: float = DEFAULT_TAU,
    K: int | None = None,
) -> list[ScoredFeature]:
    """Algorithm 1: embed -> cosine-similarity -> threshold -> sort desc -> top-K.

    Parameters
    ----------
    features      : extracted clinical features F = {f1, ..., fn}.
    specialty_vec : the specialty embedding V_s (vector B).
    embed         : function mapping a feature to its embedding vector V_f.
    tau           : similarity threshold (default 0.75).
    K             : number of top features to retain (None = keep all above tau).
    """
    scored: list[ScoredFeature] = []
    for f in features:
        v_f = embed(f)
        sim = cosine_similarity(v_f, specialty_vec)
        if sim > tau:
            scored.append(ScoredFeature(f, sim))
    scored.sort(key=lambda s: s.similarity, reverse=True)
    if K is not None:
        scored = scored[:K]
    return scored
