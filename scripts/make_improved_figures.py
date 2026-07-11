#!/usr/bin/env python3
"""Generate reviewer-friendly alternatives to the legacy radar/bubble figures."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ehr_summ import data_loader as dl

OUT = Path(__file__).resolve().parents[1] / "figures"
OUT.mkdir(exist_ok=True)


def _save(fig, stem: str) -> list[Path]:
    paths = []
    for extension in ("png", "pdf", "svg"):
        path = OUT / f"{stem}.{extension}"
        fig.savefig(path, dpi=300 if extension == "png" else None, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def f1_confidence_forest() -> list[Path]:
    frame = dl.f1_confidence_intervals()
    frame = frame[frame["specialty"] != "Pooled"].iloc[::-1]
    y = np.arange(len(frame))
    left = frame["f1"] - frame["ci_low"]
    right = frame["ci_high"] - frame["f1"]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.errorbar(frame["f1"], y, xerr=np.vstack([left, right]), fmt="o", capsize=4,
                color="#245b78", ecolor="#7895a5")
    ax.set_yticks(y, frame["specialty"])
    ax.set_xlabel("F1 score with reported 95% confidence interval")
    ax.set_xlim(0.74, 0.89)
    ax.grid(axis="x", linestyle=":", alpha=0.45)
    ax.set_title("Clinical-feature retention by specialty")
    return _save(fig, "improved_f1_confidence_forest")


def reader_feedback_bars() -> list[Path]:
    frame = dl.expert_feedback()
    x = np.arange(len(frame))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(x - width / 2, frame["satisfaction_mean"], width,
           yerr=frame["satisfaction_sd"], capsize=3, label="Satisfaction")
    ax.bar(x + width / 2, frame["coverage_mean"], width,
           yerr=frame["coverage_sd"], capsize=3, label="Coverage")
    ax.set_xticks(x, frame["specialty"], rotation=25, ha="right")
    ax.set_ylim(3.5, 5.1)
    ax.set_ylabel("Mean clinician rating (1-5)")
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    for index, count in enumerate(frame["n_experts"]):
        ax.text(index, 3.54, f"n={int(count)}", ha="center", va="bottom", fontsize=8)
    ax.set_title("Clinician ratings by specialty")
    return _save(fig, "improved_reader_feedback")


def baseline_difference_plot() -> list[Path]:
    frame = dl.baseline_comparison()
    frame = frame[frame["specialty"] != "Pooled"].iloc[::-1]
    y = np.arange(len(frame))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.scatter(frame["delta_f1"], y, color="#8b3a3a")
    for x_value, y_value, effect in zip(frame["delta_f1"], y, frame["cohens_d"]):
        ax.text(x_value + 0.001, y_value, f"d={effect:.2f}", va="center", fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y, frame["specialty"])
    ax.set_xlim(0, 0.14)
    ax.set_xlabel("Reported paired F1 difference (proposed - BioBERT)")
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    ax.set_title("Reported baseline differences")
    return _save(fig, "improved_baseline_differences")


def main() -> int:
    paths = f1_confidence_forest() + reader_feedback_bars() + baseline_difference_plot()
    print("Wrote improved figure alternatives:")
    for path in paths:
        print(f"  {path.relative_to(OUT.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
