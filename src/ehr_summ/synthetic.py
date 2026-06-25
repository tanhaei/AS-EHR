"""Calibrated synthetic-data generators.

The real per-record scores are not public. To let the *full* statistical
pipeline run end-to-end and reproduce the paper's headline inferential results
(Cohen's d ~ 1.07, p < 0.001, BH-significant in all six specialties), we
generate per-record paired F1 scores whose summary statistics match the
reported per-specialty means and effect sizes (Tables 5-6).

Everything here is clearly synthetic and deterministic given a seed; it is a
reproduction aid, NOT the paper's clinical data.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Per-specialty targets taken from Table 5 (proposed vs BioBERT, Cohen's d).
SPECIALTY_TARGETS = {
    "Cardiology":       dict(n=220, proposed=0.86, baseline=0.74, d=1.14),
    "Dentistry":        dict(n=130, proposed=0.79, baseline=0.68, d=0.98),
    "Orthopedics":      dict(n=150, proposed=0.82, baseline=0.72, d=0.93),
    "Pediatrics":       dict(n=160, proposed=0.84, baseline=0.73, d=0.99),
    "Dermatology":      dict(n=110, proposed=0.80, baseline=0.70, d=0.95),
    "General Medicine": dict(n=230, proposed=0.85, baseline=0.73, d=1.14),
}


@dataclass
class PairedScores:
    specialty: str
    proposed: np.ndarray
    baseline: np.ndarray


def _clip01(a: np.ndarray) -> np.ndarray:
    return np.clip(a, 0.0, 1.0)


def generate_specialty_pairs(specialty: str, seed: int = 0) -> PairedScores:
    """Generate paired per-record F1 arrays for one specialty, calibrated so that
    mean(proposed), mean(baseline) and Cohen's d (paired) match Table 5.

    Construction: draw the per-record *difference* ~ Normal(mu_diff, sd_diff)
    with sd_diff = mu_diff / d, then re-center exactly to the target means.
    """
    t = SPECIALTY_TARGETS[specialty]
    rng = np.random.default_rng(seed)
    n = t["n"]
    mu_diff = t["proposed"] - t["baseline"]
    sd_diff = mu_diff / t["d"]

    diff = rng.normal(mu_diff, sd_diff, size=n)
    # Baseline F1 spread (cosmetic), then proposed = baseline + diff.
    baseline = rng.normal(t["baseline"], 0.05, size=n)
    proposed = baseline + diff

    # Re-center difference exactly to hit the target mean difference and sd
    # (keeps Cohen's d on target despite finite-sample noise).
    diff_actual = proposed - baseline
    diff_actual = (diff_actual - diff_actual.mean()) / diff_actual.std(ddof=1) * sd_diff + mu_diff
    proposed = baseline + diff_actual
    # Re-center absolute levels to the reported means.
    baseline = baseline - baseline.mean() + t["baseline"]
    proposed = proposed - proposed.mean() + t["proposed"]

    return PairedScores(specialty, _clip01(proposed), _clip01(baseline))


def generate_all_pairs(seed: int = 0) -> dict[str, PairedScores]:
    """All six specialties; distinct sub-seeds for reproducibility."""
    return {
        name: generate_specialty_pairs(name, seed=seed + i)
        for i, name in enumerate(SPECIALTY_TARGETS)
    }


def generate_tuning_scores(seed: int = 0, n: int = 400):
    """A synthetic held-out tuning split for the threshold sweep (Table 4).

    Returns (similarities, is_required): cosine similarities for candidate
    features and a boolean "clinically required" gold label.

    The two score distributions are centred symmetrically around 0.75
    (required ~ N(0.855, 0.10), irrelevant ~ N(0.645, 0.10)) so that, with
    equal class sizes and equal variance, precision and recall cross near
    tau = 0.75 and F1 of retention peaks there -- reproducing the paper's
    selected operating point and the precision-up / recall-down trade-off
    of Table 4.
    """
    rng = np.random.default_rng(seed)
    n_req = n // 2
    n_irr = n - n_req
    sim_req = _clip01(rng.normal(0.855, 0.10, size=n_req))   # required: high sim
    sim_irr = _clip01(rng.normal(0.645, 0.10, size=n_irr))   # irrelevant: lower sim
    sims = np.concatenate([sim_req, sim_irr])
    gold = np.concatenate([np.ones(n_req, bool), np.zeros(n_irr, bool)])
    idx = rng.permutation(n)
    return sims[idx], gold[idx]
