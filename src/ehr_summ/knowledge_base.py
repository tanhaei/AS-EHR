"""Specialty knowledge-base loading and centroid construction."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .embeddings import TextEmbedder


@dataclass(frozen=True)
class KnowledgeConcept:
    specialty: str
    concept_id: str
    descriptor: str
    mandatory: bool = False
    synonyms: tuple[str, ...] = ()


def load_knowledge_base(path: str | Path) -> list[KnowledgeConcept]:
    concepts: list[KnowledgeConcept] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            concepts.append(
                KnowledgeConcept(
                    specialty=row["specialty"].strip(),
                    concept_id=row["concept_id"].strip(),
                    descriptor=row["descriptor"].strip(),
                    mandatory=row.get("mandatory", "").strip().lower() in {"1", "true", "yes"},
                    synonyms=tuple(
                        item.strip()
                        for item in row.get("synonyms", "").split("|")
                        if item.strip()
                    ),
                )
            )
    if not concepts:
        raise ValueError("knowledge base is empty")
    return concepts


def build_centroids(
    concepts: Iterable[KnowledgeConcept], embedder: TextEmbedder
) -> dict[str, np.ndarray]:
    grouped: dict[str, list[str]] = {}
    for concept in concepts:
        grouped.setdefault(concept.specialty, []).append(concept.descriptor)
    return {
        specialty: embedder.encode(descriptors).mean(axis=0)
        for specialty, descriptors in grouped.items()
    }
