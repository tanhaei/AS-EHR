#!/usr/bin/env python3
"""Recompute the paper's derived/aggregate quantities from the per-specialty
tables and check internal consistency against the reported headline numbers.

Run:  python scripts/reproduce_tables.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# allow "import ehr_summ" when run from a checkout without installation
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ehr_summ import data_loader as dl
from ehr_summ.metrics import macro_average, weighted_average


def _check(label: str, got: float, expected: float, tol: float) -> bool:
    ok = abs(got - expected) <= tol
    flag = "OK " if ok else "FAIL"
    print(f"  [{flag}] {label}: computed={got:.4f}  paper={expected:.4f}  (tol={tol})")
    return ok


def main() -> int:
    dist = dl.specialty_distribution()
    metrics = dl.specialty_metrics()
    feedback = dl.expert_feedback()
    n = dist["n_records"].tolist()

    all_ok = True
    print("== Corpus composition (Table 3) ==")
    total = int(dist["n_records"].sum())
    all_ok &= _check("total records", total, 1000, 0)
    cs_counts = (dist["n_records"] * dist["pct_code_switched"] / 100.0)
    all_ok &= _check("code-switched records", float(cs_counts.sum()), 707, 1.0)
    pooled_cs_pct = 100.0 * cs_counts.sum() / total
    all_ok &= _check("pooled %% code-switched", pooled_cs_pct, 70.7, 0.1)

    print("\n== Pooled / macro metrics (Tables 6, A.8; Abstract) ==")
    f1_macro = macro_average(metrics["f1"])
    f1_pooled = weighted_average(metrics["f1"], n)
    all_ok &= _check("pooled F1 (record-weighted)", f1_pooled, 0.83, 0.005)
    all_ok &= _check("macro F1", f1_macro, 0.83, 0.01)

    rouge_macro = macro_average(metrics["rouge_n"])
    all_ok &= _check("macro-average ROUGE-N", rouge_macro, 0.855, 0.001)

    print("\n== Code-switched term accuracy proxy (Section 5.4) ==")
    cs_acc_weighted = weighted_average(metrics["code_switched_term_accuracy"], cs_counts)
    print("  NOTE: record-count weighting is only a proxy; exact term-level pooling requires")
    print("        correct/total term counts for every specialty.")
    all_ok &= _check("record-weighted proxy", cs_acc_weighted, 0.852, 0.005)

    print("\n== Reader study (Table 7; Abstract) ==")
    all_ok &= _check("total experts", int(feedback["n_experts"].sum()), 130, 0)
    all_ok &= _check(
        "expert-weighted satisfaction",
        weighted_average(feedback["satisfaction_mean"], feedback["n_experts"]),
        4.4,
        0.05,
    )
    all_ok &= _check(
        "expert-weighted coverage",
        weighted_average(feedback["coverage_mean"], feedback["n_experts"]),
        4.5,
        0.05,
    )

    print("\n== Baseline comparison (Table 5) ==")
    base = dl.baseline_comparison().set_index("specialty")
    per = base.drop(index="Pooled")
    all_ok &= _check("pooled proposed F1", weighted_average(per["proposed_f1"], n), 0.83, 0.005)
    all_ok &= _check("pooled BioBERT F1", weighted_average(per["biobert_f1"], n), 0.72, 0.005)

    print("\n" + ("AGGREGATE CONSISTENCY CHECKS PASSED" if all_ok else "INCONSISTENCY DETECTED"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
