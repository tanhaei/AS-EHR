#!/usr/bin/env python3
"""Demonstrate the threshold-selection procedure (Section 5.2, Table 4).

Sweeps tau over the paper's grid on a synthetic held-out tuning split and
reports precision/recall/F1 of "clinically required feature" retention,
selecting the F1-maximising operating point (the paper selects tau = 0.75).

Run:  python scripts/run_threshold_sweep.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from ehr_summ import data_loader as dl
from ehr_summ.metrics import precision_recall_f1
from ehr_summ.synthetic import generate_tuning_scores

GRID = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85]


def sweep(sims: np.ndarray, gold: np.ndarray):
    rows = []
    for tau in GRID:
        kept = sims > tau
        tp = int(np.sum(kept & gold))
        fp = int(np.sum(kept & ~gold))
        fn = int(np.sum(~kept & gold))
        p, r, f1 = precision_recall_f1(tp, fp, fn)
        rows.append((tau, p, r, f1))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-csv",
        help=(
            "Real tuning evidence with columns similarity,is_required. "
            "Omit for the labelled synthetic demo."
        ),
    )
    parser.add_argument("--output-csv")
    args = parser.parse_args()
    if args.input_csv:
        evidence = pd.read_csv(args.input_csv)
        missing = {"similarity", "is_required"} - set(evidence.columns)
        if missing:
            raise ValueError(f"threshold evidence missing columns: {sorted(missing)}")
        sims = evidence["similarity"].to_numpy(float)
        gold = evidence["is_required"].astype(bool).to_numpy()
        label = "REAL TUNING EVIDENCE"
    else:
        sims, gold = generate_tuning_scores(seed=0, n=600)
        label = "SYNTHETIC DEMONSTRATION"
    rows = sweep(sims, gold)

    print(f"{label} threshold sweep:")
    print(f"{'tau':>5} {'precision':>10} {'recall':>8} {'f1':>7}")
    for tau, p, r, f1 in rows:
        print(f"{tau:5.2f} {p:10.2f} {r:8.2f} {f1:7.2f}")

    best_tau = max(rows, key=lambda row: row[3])[0]
    print(f"\nSelected operating point (max F1): tau = {best_tau:.2f}")
    if args.output_csv:
        output = Path(args.output_csv)
        output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows, columns=["tau", "precision", "recall", "f1"]).to_csv(output, index=False)
        print(f"Wrote sweep evidence to {output}")

    print("\nPaper-reported sweep (Table 4):")
    print(dl.threshold_sensitivity().to_string(index=False))
    paper_best = dl.threshold_sensitivity().sort_values("f1", ascending=False).iloc[0]["tau"]
    print(f"Paper selected: tau = {paper_best:.2f}")

    # The exact arg-max depends on the true (non-public) similarity
    # distribution; what reproduces robustly is (a) the monotone trade-off
    # precision-up / recall-down as tau increases, and (b) an F1 maximum in
    # the paper's plateau region {0.70, 0.75} (F1 = 0.84 / 0.85 in Table 4).
    precisions = [r[1] for r in rows]
    recalls = [r[2] for r in rows]
    prec_monotone = all(b >= a for a, b in zip(precisions, precisions[1:]))
    rec_monotone = all(b <= a for a, b in zip(recalls, recalls[1:]))
    in_plateau = best_tau in (0.70, 0.75)

    print("\n== Consistency checks ==")
    print(f"  [{'OK ' if prec_monotone else 'FAIL'}] precision increases with tau")
    print(f"  [{'OK ' if rec_monotone else 'FAIL'}] recall decreases with tau")
    if not args.input_csv:
        print(f"  [{'OK ' if in_plateau else 'FAIL'}] F1 maximised in paper plateau {{0.70, 0.75}}")
    ok = prec_monotone and rec_monotone and (in_plateau if not args.input_csv else True)
    message = ("Real threshold evidence analyzed" if args.input_csv
               else "Synthetic threshold demonstration completed")
    print("\n" + (message + " [OK]" if ok else "Trade-off check failed [FAIL]"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
