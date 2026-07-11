"""End-to-end feature selection shared by inference and evaluation scripts."""
from __future__ import annotations

import numpy as np

from .embeddings import TextEmbedder
from .schemas import ClinicalFeature, ClinicalRecord
from .specialty_filter import rank_and_select, select_with_safety_layer


def select_specialty_features(
    record: ClinicalRecord,
    specialty_vector: np.ndarray,
    embedder: TextEmbedder,
    *,
    tau: float,
    K: int,
    safety_layer: bool = True,
) -> list[ClinicalFeature]:
    cache: dict[str, np.ndarray] = {}

    def embed(feature: ClinicalFeature) -> np.ndarray:
        if feature.feature_id not in cache:
            cache[feature.feature_id] = embedder.encode([feature.text])[0]
        return cache[feature.feature_id]

    if safety_layer:
        return select_with_safety_layer(record.features, specialty_vector, embed, tau=tau, K=K)
    return [item.feature for item in rank_and_select(record.features, specialty_vector, embed, tau=tau, K=K)]


def select_agnostic_features(
    record: ClinicalRecord, *, K: int, safety_layer: bool = True
) -> list[ClinicalFeature]:
    mandatory = [feature for feature in record.features if feature.mandatory] if safety_layer else []
    optional = [feature for feature in record.features if feature not in mandatory]
    return mandatory + optional[:K]
