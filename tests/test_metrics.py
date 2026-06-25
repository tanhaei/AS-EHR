"""Tests for ehr_summ.metrics (pytest-style; also runnable via tests/run_tests.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ehr_summ.metrics import (
    macro_average,
    precision_recall_f1,
    prf_from_sets,
    rouge_n,
    weighted_average,
)

TOL = 1e-9


def test_prf_perfect():
    p, r, f1 = precision_recall_f1(tp=10, fp=0, fn=0)
    assert (p, r, f1) == (1.0, 1.0, 1.0)


def test_prf_known_values():
    # tp=8, fp=2, fn=2 -> p=0.8, r=0.8, f1=0.8
    p, r, f1 = precision_recall_f1(8, 2, 2)
    assert abs(p - 0.8) < TOL and abs(r - 0.8) < TOL and abs(f1 - 0.8) < TOL


def test_prf_zero_denominator():
    assert precision_recall_f1(0, 0, 0) == (0.0, 0.0, 0.0)


def test_prf_from_sets():
    pred = {"a", "b", "c"}
    gold = {"b", "c", "d"}
    p, r, f1 = prf_from_sets(pred, gold)
    # tp=2, fp=1, fn=1 -> p=r=f1=2/3
    assert abs(p - 2 / 3) < TOL and abs(f1 - 2 / 3) < TOL


def test_rouge_n_identical():
    toks = "the patient has hypertension".split()
    assert abs(rouge_n(toks, toks, n=1) - 1.0) < TOL
    assert abs(rouge_n(toks, toks, n=2) - 1.0) < TOL


def test_rouge_n_partial():
    cand = "patient has hypertension".split()
    ref = "patient has diabetes".split()
    # unigrams: matches {patient, has} = 2 of 3 reference unigrams
    assert abs(rouge_n(cand, ref, n=1) - 2 / 3) < TOL


def test_rouge_n_clipping():
    cand = "a a a".split()
    ref = "a a b".split()
    # reference has 'a' x2, 'b' x1 -> match min(3,2)+min(0,1)=2 over 3
    assert abs(rouge_n(cand, ref, n=1) - 2 / 3) < TOL


def test_macro_matches_paper_rouge():
    rouge = [0.89, 0.82, 0.84, 0.87, 0.83, 0.88]
    assert abs(macro_average(rouge) - 0.855) < 1e-6


def test_weighted_pooled_f1():
    f1 = [0.86, 0.79, 0.82, 0.84, 0.80, 0.85]
    n = [220, 130, 150, 160, 110, 230]
    assert abs(weighted_average(f1, n) - 0.83) < 0.005
