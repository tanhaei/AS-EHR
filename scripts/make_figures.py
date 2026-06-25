#!/usr/bin/env python3
"""Regenerate the paper's figures from the reported per-specialty data.

Produces (under figures/):
  * fig7a_radar.png        - holistic performance radar (Figure 7a)
  * fig7b_bar.png          - precision/recall/F1 grouped bars (Figure 7b)
  * fig8_bubble.png        - satisfaction vs coverage bubbles (Figure 8)
  * table5_baseline.png    - proposed vs BioBERT F1 by specialty

Run:  python scripts/make_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np

from ehr_summ import data_loader as dl

OUT = Path(__file__).resolve().parents[1] / "figures"
OUT.mkdir(exist_ok=True)


def fig7a_radar() -> Path:
    m = dl.specialty_metrics()
    axes = ["precision", "recall", "f1", "rouge_n", "code_switched_term_accuracy"]
    labels = ["Precision", "Recall", "F1", "ROUGE-N", "Accuracy"]
    angles = np.linspace(0, 2 * np.pi, len(axes), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    for _, row in m.iterrows():
        vals = [row[a] for a in axes]
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=1.5, label=row["specialty"])
        ax.fill(angles, vals, alpha=0.05)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0.70, 0.92)
    ax.set_title("Holistic Performance Profile (Figure 7a)", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.10), fontsize=8)
    path = OUT / "fig7a_radar.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def fig7b_bar() -> Path:
    m = dl.specialty_metrics()
    specialties = m["specialty"].tolist()
    x = np.arange(len(specialties))
    w = 0.27
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w, m["precision"], w, label="Precision")
    ax.bar(x, m["recall"], w, label="Recall")
    ax.bar(x + w, m["f1"], w, label="F1-Score")
    ax.set_xticks(x)
    ax.set_xticklabels(specialties, rotation=30, ha="right")
    ax.set_ylim(0.70, 0.95)
    ax.set_ylabel("Metric Score")
    ax.set_title("Detailed Metric Breakdown (Figure 7b)")
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    path = OUT / "fig7b_bar.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def fig8_bubble() -> Path:
    f = dl.expert_feedback()
    fig, ax = plt.subplots(figsize=(8, 6))
    sizes = (f["n_experts"] / f["n_experts"].max() * 1200).tolist()
    ax.scatter(f["satisfaction_mean"], f["coverage_mean"], s=sizes, alpha=0.6,
               edgecolors="black", linewidths=0.8)
    for _, row in f.iterrows():
        ax.annotate(f"{row['specialty']}\n(n={int(row['n_experts'])})",
                    (row["satisfaction_mean"], row["coverage_mean"]),
                    fontsize=8, ha="center", va="center")
    lim = [4.1, 4.8]
    ax.plot(lim, lim, "k--", alpha=0.4, linewidth=0.8)
    ax.set_xlim(lim)
    ax.set_ylim(4.2, 4.8)
    ax.set_xlabel("Satisfaction Level (1-5)")
    ax.set_ylabel("Summary Coverage Score (1-5)")
    ax.set_title("Expert Feedback: Satisfaction vs. Coverage (Figure 8)")
    ax.grid(linestyle=":", alpha=0.5)
    path = OUT / "fig8_bubble.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def table5_baseline() -> Path:
    b = dl.baseline_comparison()
    per = b[b["specialty"] != "Pooled"]
    x = np.arange(len(per))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w / 2, per["proposed_f1"], w, label="Proposed (MedGemma 27B)")
    ax.bar(x + w / 2, per["biobert_f1"], w, label="BioBERT (extractive)")
    ax.set_xticks(x)
    ax.set_xticklabels(per["specialty"], rotation=30, ha="right")
    ax.set_ylim(0.60, 0.92)
    ax.set_ylabel("F1-Score")
    ax.set_title("Proposed vs BioBERT baseline (Table 5)")
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    path = OUT / "table5_baseline.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> int:
    paths = [fig7a_radar(), fig7b_bar(), fig8_bubble(), table5_baseline()]
    print("Wrote figures:")
    for p in paths:
        print(f"  {p.relative_to(OUT.parent)}  ({p.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
