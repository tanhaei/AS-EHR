"""Tests for ehr_summ.stats."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from ehr_summ.stats import (
    benjamini_hochberg,
    bootstrap_ci,
    cluster_bootstrap_ci,
    cohens_d_paired,
    paired_test,
    rank_biserial,
)

TOL = 1e-9


def test_bootstrap_ci_contains_mean():
    rng = np.random.default_rng(0)
    data = rng.normal(0.83, 0.05, size=500)
    est, lo, hi = bootstrap_ci(data, n_boot=500, seed=1)
    assert lo < est < hi
    assert abs(est - data.mean()) < TOL


def test_bootstrap_ci_narrows_with_n():
    rng = np.random.default_rng(0)
    small = rng.normal(0, 1, 30)
    large = rng.normal(0, 1, 3000)
    _, lo_s, hi_s = bootstrap_ci(small, n_boot=400, seed=2)
    _, lo_l, hi_l = bootstrap_ci(large, n_boot=400, seed=2)
    assert (hi_l - lo_l) < (hi_s - lo_s)


def test_cluster_bootstrap_keeps_repeated_patient_records_together():
    values = np.array([0.8, 0.9, 0.4, 0.5])
    clusters = np.array(["p1", "p1", "p2", "p2"])
    estimate, low, high = cluster_bootstrap_ci(values, clusters, n_boot=200, seed=4)
    assert abs(estimate - 0.65) < TOL
    assert low <= estimate <= high


def test_cohens_d_sign_and_scale():
    x = np.array([1.0, 1.0, 1.0, 1.0])
    y = np.array([0.0, 0.0, 0.0, 0.0])
    # constant difference -> sd(diff)=0 -> guarded to 0.0
    assert cohens_d_paired(x, y) == 0.0
    rng = np.random.default_rng(0)
    a = rng.normal(0.83, 0.05, 400)
    b = a - 0.11  # constant shift would give 0; add noise
    b = b + rng.normal(0, 0.05, 400)
    d = cohens_d_paired(a, b)
    assert d > 0  # a > b on average


def test_paired_test_detects_difference():
    rng = np.random.default_rng(0)
    a = rng.normal(0.83, 0.05, 300)
    b = rng.normal(0.72, 0.05, 300)
    res = paired_test(a, b)
    assert res.p_value < 0.001
    assert res.mean_x > res.mean_y


def test_paired_test_handles_identical_pairs():
    values = np.array([0.2, 0.4, 0.6])
    result = paired_test(values, values)
    assert result.p_value == 1.0
    assert result.effect_size == 0.0


def test_rank_biserial_range():
    x = np.array([2.0, 3.0, 4.0, 5.0])
    y = np.array([1.0, 1.0, 1.0, 1.0])
    rb = rank_biserial(x, y)
    assert 0.0 < rb <= 1.0


def test_benjamini_hochberg_basic():
    # one clearly significant, rest null
    pvals = [0.0001, 0.2, 0.4, 0.6, 0.8, 0.9]
    adj, reject = benjamini_hochberg(pvals, q=0.05)
    assert reject[0] is True
    assert all(0.0 <= a <= 1.0 for a in adj)
    # adjusted >= raw for the smallest p (BH inflates)
    assert adj[0] >= pvals[0]


def test_benjamini_hochberg_all_significant():
    pvals = [1e-6, 1e-5, 1e-6, 1e-6, 1e-5, 1e-6]
    _, reject = benjamini_hochberg(pvals, q=0.05)
    assert all(reject)
