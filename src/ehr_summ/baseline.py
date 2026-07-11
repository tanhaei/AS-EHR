"""Matched-length extractive baseline with optional BioBERT embeddings."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .embeddings import TextEmbedder
from .preprocessing import split_sentences, tokenize_clinical


@dataclass
class ExtractiveResult:
    summary: str
    selected_sentence_indices: list[int]


def extractive_summary(note: str, query_terms: Sequence[str], embedder: TextEmbedder,
                       *, max_words: int = 200) -> ExtractiveResult:
    sentences = split_sentences(note)
    if not sentences:
        return ExtractiveResult("", [])
    query = " ".join(query_terms) or "clinical summary"
    sentence_vectors = embedder.encode(sentences)
    query_vector = embedder.encode([query])[0]
    scores = sentence_vectors @ query_vector
    ranked = np.argsort(-scores)
    selected: list[int] = []
    words = 0
    for index in ranked:
        sentence_words = len(tokenize_clinical(sentences[int(index)]))
        if selected and words + sentence_words > max_words:
            continue
        selected.append(int(index))
        words += sentence_words
        if words >= max_words:
            break
    selected.sort()
    return ExtractiveResult(" ".join(sentences[index] for index in selected), selected)

