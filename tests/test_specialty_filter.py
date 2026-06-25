"""Tests for ehr_summ.specialty_filter (Algorithm 1)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from ehr_summ.specialty_filter import (
    DEFAULT_TAU,
    build_specialty_embedding,
    cosine_similarity,
    rank_and_select,
)

TOL = 1e-9


def test_cosine_identical():
    v = np.array([1.0, 2.0, 3.0])
    assert abs(cosine_similarity(v, v) - 1.0) < TOL


def test_cosine_orthogonal():
    assert abs(cosine_similarity([1, 0], [0, 1])) < TOL


def test_cosine_zero_vector():
    assert cosine_similarity([0, 0], [1, 1]) == 0.0


def test_centroid():
    emb = [[0.0, 0.0], [2.0, 4.0]]
    c = build_specialty_embedding(emb)
    assert np.allclose(c, [1.0, 2.0])


def test_default_tau_is_paper_value():
    assert abs(DEFAULT_TAU - 0.75) < TOL


def test_rank_and_select_threshold_and_topk():
    # specialty vector points along +x
    spec = np.array([1.0, 0.0])
    features = ["aligned1", "aligned2", "orthogonal", "weak"]
    embeds = {
        "aligned1": np.array([1.0, 0.05]),    # sim ~1.0
        "aligned2": np.array([1.0, 0.20]),    # sim ~0.98
        "orthogonal": np.array([0.0, 1.0]),   # sim 0 -> dropped by tau
        "weak": np.array([0.5, 0.9]),         # sim ~0.49 -> dropped by tau
    }
    out = rank_and_select(features, spec, embed=lambda f: embeds[f], tau=DEFAULT_TAU, K=5)
    kept = [s.feature for s in out]
    assert kept == ["aligned1", "aligned2"]          # only above-threshold, sorted desc
    assert out[0].similarity >= out[1].similarity     # ranking is descending


def test_topk_truncation():
    spec = np.array([1.0, 0.0])
    feats = [f"f{i}" for i in range(5)]
    # all highly aligned, decreasing similarity
    embeds = {f"f{i}": np.array([1.0, 0.01 * i]) for i in range(5)}
    out = rank_and_select(feats, spec, embed=lambda f: embeds[f], tau=0.5, K=3)
    assert len(out) == 3
    sims = [s.similarity for s in out]
    assert sims == sorted(sims, reverse=True)
