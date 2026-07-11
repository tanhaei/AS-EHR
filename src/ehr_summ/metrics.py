"""Evaluation metrics used in the paper.

Implements:
  * Precision, Recall, F1 (Eqs. 1-2) over "clinically required features".
  * ROUGE-N (Eq. 3): clipped n-gram overlap between a candidate and one or
    more reference summaries.
  * Macro / weighted / pooled aggregation helpers used to reproduce the
    headline numbers (pooled F1 = 0.83, macro ROUGE-N = 0.855, etc.).
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable, Sequence

from .preprocessing import tokenize_clinical


# --------------------------------------------------------------------------- #
# Precision / Recall / F1  (Eqs. 1-2)
# --------------------------------------------------------------------------- #
def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Return (precision, recall, F1) from raw counts.

    Edge cases follow the usual convention: a metric whose denominator is 0
    is defined as 0.0.
    """
    if tp < 0 or fp < 0 or fn < 0:
        raise ValueError("counts must be non-negative")
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return precision, recall, f1


def prf_from_sets(predicted, gold) -> tuple[float, float, float]:
    """Precision/Recall/F1 by comparing a *set* of predicted features against a
    checklist-derived gold set (the paper's "clinically required features").
    """
    predicted, gold = set(predicted), set(gold)
    tp = len(predicted & gold)
    fp = len(predicted - gold)
    fn = len(gold - predicted)
    return precision_recall_f1(tp, fp, fn)


# --------------------------------------------------------------------------- #
# ROUGE-N  (Eq. 3)
# --------------------------------------------------------------------------- #
def _ngrams(tokens: Sequence[str], n: int) -> Counter:
    if n < 1:
        raise ValueError("n must be >= 1")
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def rouge_n(candidate: Sequence[str], references: Iterable[Sequence[str]], n: int = 1) -> float:
    """ROUGE-N as written in Eq. (3): summed clipped match counts over all
    reference summaries, divided by the total number of reference n-grams.

    ``references`` may be a single token sequence or an iterable of sequences.
    """
    # Allow a single reference passed as a flat token list.
    refs = list(references)
    if refs and isinstance(refs[0], str):
        refs = [refs]  # a single reference given as list[str]

    cand_ngrams = _ngrams(candidate, n)
    total_match = 0
    total_ref = 0
    for ref in refs:
        ref_ngrams = _ngrams(ref, n)
        for gram, ref_count in ref_ngrams.items():
            total_match += min(cand_ngrams.get(gram, 0), ref_count)
        total_ref += sum(ref_ngrams.values())
    return total_match / total_ref if total_ref > 0 else 0.0


# --------------------------------------------------------------------------- #
# Aggregation helpers
# --------------------------------------------------------------------------- #
def macro_average(values: Sequence[float]) -> float:
    """Unweighted mean across specialties (paper's 'macro-average')."""
    values = list(values)
    if not values:
        raise ValueError("empty sequence")
    return sum(values) / len(values)


def weighted_average(values: Sequence[float], weights: Sequence[float]) -> float:
    """Record-count-weighted mean (approximates the pooled estimate)."""
    values, weights = list(values), list(weights)
    if len(values) != len(weights):
        raise ValueError("values and weights length mismatch")
    denom = sum(weights)
    if denom == 0:
        raise ValueError("weights sum to zero")
    return sum(v * w for v, w in zip(values, weights)) / denom


def rouge_l(candidate: Sequence[str], reference: Sequence[str]) -> float:
    """ROUGE-L recall based on the longest common subsequence."""
    if not reference:
        return 0.0
    previous = [0] * (len(reference) + 1)
    for token in candidate:
        current = [0]
        for index, ref_token in enumerate(reference, 1):
            if token == ref_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(current[-1], previous[index]))
        previous = current
    return previous[-1] / len(reference)


def text_rouge(candidate: str, reference: str) -> dict[str, float]:
    """Persian-normalized ROUGE-1/2/L with an explicit tokenizer."""
    candidate_tokens = tokenize_clinical(candidate)
    reference_tokens = tokenize_clinical(reference)
    return {
        "rouge_1": rouge_n(candidate_tokens, reference_tokens, n=1),
        "rouge_2": rouge_n(candidate_tokens, reference_tokens, n=2),
        "rouge_l": rouge_l(candidate_tokens, reference_tokens),
    }
