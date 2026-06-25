"""Statistical-analysis pipeline (Section 6.2).

Implements the exact procedure described in the paper:
  * 95% bootstrap confidence intervals over records (B = 1000 resamples).
  * Paired comparison: Shapiro-Wilk normality test chooses a paired t-test,
    otherwise a Wilcoxon signed-rank test.
  * Effect sizes: Cohen's d (paired) or rank-biserial correlation.
  * Benjamini-Hochberg false-discovery-rate correction across the six
    per-specialty comparisons.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy import stats as sps


# --------------------------------------------------------------------------- #
# Bootstrap confidence interval
# --------------------------------------------------------------------------- #
def bootstrap_ci(
    values: Sequence[float],
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 1000,
    ci: float = 95.0,
    seed: int | None = 0,
) -> tuple[float, float, float]:
    """Return (point_estimate, ci_low, ci_high) by resampling records with replacement."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        raise ValueError("empty sample")
    boots = np.empty(n_boot)
    for i in range(n_boot):
        sample = values[rng.integers(0, n, size=n)]
        boots[i] = statistic(sample)
    alpha = (100.0 - ci) / 2.0
    low, high = np.percentile(boots, [alpha, 100.0 - alpha])
    return float(statistic(values)), float(low), float(high)


# --------------------------------------------------------------------------- #
# Effect sizes
# --------------------------------------------------------------------------- #
def cohens_d_paired(x: Sequence[float], y: Sequence[float]) -> float:
    """Cohen's d for paired samples: mean(diff) / sd(diff)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    diff = x - y
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 0 else 0.0


def rank_biserial(x: Sequence[float], y: Sequence[float]) -> float:
    """Matched-pairs rank-biserial correlation (effect size for Wilcoxon)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    diff = x - y
    diff = diff[diff != 0]
    if diff.size == 0:
        return 0.0
    ranks = sps.rankdata(np.abs(diff))
    r_plus = ranks[diff > 0].sum()
    r_minus = ranks[diff < 0].sum()
    total = ranks.sum()
    return float((r_plus - r_minus) / total)


# --------------------------------------------------------------------------- #
# Paired hypothesis test (auto-selected)
# --------------------------------------------------------------------------- #
@dataclass
class PairedResult:
    test: str
    statistic: float
    p_value: float
    effect_name: str
    effect_size: float
    mean_x: float
    mean_y: float


def paired_test(x: Sequence[float], y: Sequence[float], alpha_normal: float = 0.05) -> PairedResult:
    """Paired t-test if differences look normal (Shapiro-Wilk p > alpha),
    otherwise Wilcoxon signed-rank. Reports the matching effect size."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    diff = x - y

    # Shapiro-Wilk needs >= 3 points and some variance.
    normal = False
    if diff.size >= 3 and np.ptp(diff) > 0:
        try:
            _, p_norm = sps.shapiro(diff)
            normal = p_norm > alpha_normal
        except Exception:
            normal = False

    if normal:
        stat, p = sps.ttest_rel(x, y)
        return PairedResult("paired t-test", float(stat), float(p),
                            "cohens_d", cohens_d_paired(x, y), float(x.mean()), float(y.mean()))
    else:
        # zero_method="wilcox" drops zero-differences (scipy default behaviour varies by version)
        stat, p = sps.wilcoxon(x, y)
        return PairedResult("wilcoxon signed-rank", float(stat), float(p),
                            "rank_biserial", rank_biserial(x, y), float(x.mean()), float(y.mean()))


# --------------------------------------------------------------------------- #
# Benjamini-Hochberg FDR
# --------------------------------------------------------------------------- #
def benjamini_hochberg(p_values: Sequence[float], q: float = 0.05) -> tuple[list[float], list[bool]]:
    """Return (adjusted_q_values, reject_flags) for the BH-FDR procedure.

    adjusted q-values are the standard step-up BH-adjusted p-values (monotone,
    capped at 1.0); reject_flags indicate significance at level ``q``.
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    if m == 0:
        return [], []
    order = np.argsort(p)
    ranked = p[order]
    # BH adjusted values with monotonicity enforced from the largest down.
    adj = ranked * m / (np.arange(1, m + 1))
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0.0, 1.0)
    # restore original ordering
    adjusted = np.empty(m)
    adjusted[order] = adj
    reject = adjusted <= q
    return adjusted.tolist(), reject.tolist()
