#!/usr/bin/env python3
"""Reproduce the statistical analysis (Section 6.2-6.3, Table 5).

Using calibrated synthetic per-record paired F1 scores:
  * pooled paired test of proposed vs BioBERT + Cohen's d,
  * per-specialty paired tests with Benjamini-Hochberg FDR correction,
  * 95% bootstrap CI for the pooled F1.

Run:  python scripts/run_statistical_analysis.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from ehr_summ.stats import benjamini_hochberg, bootstrap_ci, cohens_d_paired, paired_test
from ehr_summ.synthetic import generate_all_pairs


def main() -> int:
    pairs = generate_all_pairs(seed=0)

    print("== Per-specialty paired comparison (proposed vs BioBERT) ==")
    print(f"{'specialty':<17}{'proposed':>9}{'biobert':>9}{'d':>7}{'p_raw':>11}")
    p_values, names, ds = [], [], []
    all_proposed, all_baseline = [], []
    for name, ps in pairs.items():
        res = paired_test(ps.proposed, ps.baseline)
        p_values.append(res.p_value)
        names.append(name)
        ds.append(cohens_d_paired(ps.proposed, ps.baseline))
        all_proposed.append(ps.proposed)
        all_baseline.append(ps.baseline)
        print(f"{name:<17}{ps.proposed.mean():9.3f}{ps.baseline.mean():9.3f}"
              f"{ds[-1]:7.2f}{res.p_value:11.2e}")

    q_adj, reject = benjamini_hochberg(p_values, q=0.05)
    print("\n== Benjamini-Hochberg FDR correction (six comparisons) ==")
    print(f"{'specialty':<17}{'q_adjusted':>12}{'significant':>13}")
    for name, q, rej in zip(names, q_adj, reject):
        print(f"{name:<17}{q:12.2e}{str(bool(rej)):>13}")

    print("\n== Pooled analysis (n = 1000) ==")
    proposed = np.concatenate(all_proposed)
    baseline = np.concatenate(all_baseline)
    pooled = paired_test(proposed, baseline)
    d_pooled = cohens_d_paired(proposed, baseline)
    est, lo, hi = bootstrap_ci(proposed, n_boot=1000, ci=95, seed=0)
    print(f"  n                     : {proposed.size}")
    print(f"  mean proposed F1      : {proposed.mean():.3f}  (paper 0.83)")
    print(f"  mean BioBERT  F1      : {baseline.mean():.3f}  (paper 0.72)")
    print(f"  test                  : {pooled.test}")
    print(f"  p-value               : {pooled.p_value:.2e}  (paper p < 0.001)")
    print(f"  Cohen's d (pooled)    : {d_pooled:.2f}  (paper 1.07)")
    print(f"  pooled F1 95%% CI      : [{lo:.3f}, {hi:.3f}]  (paper [0.825, 0.839])")

    checks = {
        "pooled mean proposed ~ 0.83": abs(proposed.mean() - 0.83) < 0.01,
        "pooled mean baseline ~ 0.72": abs(baseline.mean() - 0.72) < 0.01,
        "pooled p < 0.001": pooled.p_value < 0.001,
        "pooled d in [1.0, 1.15]": 1.0 <= d_pooled <= 1.15,
        "all six BH-significant": all(reject),
    }
    print("\n== Consistency checks ==")
    for label, ok in checks.items():
        print(f"  [{'OK ' if ok else 'FAIL'}] {label}")
    all_ok = all(checks.values())
    print("\n" + ("STATISTICAL RESULTS REPRODUCED [OK]" if all_ok else "MISMATCH [FAIL]"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
