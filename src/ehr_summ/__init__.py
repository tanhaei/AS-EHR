"""ehr_summ: reproduction harness for the specialty-aware EHR summarization paper.

This package implements the *methods* described in the paper:
  - Algorithm 1 (specialty-based feature ranking via cosine similarity),
  - the evaluation metrics (precision / recall / F1, ROUGE-N),
  - the statistical-analysis pipeline (bootstrap CIs, paired tests,
    Cohen's d, Benjamini-Hochberg FDR correction),
  - a calibrated synthetic-data generator so the full pipeline can run
    end-to-end without the proprietary BioArc data or fine-tuned weights.

The real clinical corpus and the fine-tuned MedGemma-27B weights are NOT
included (proprietary / privacy-restricted, per the paper's data-availability
statement). Numbers reported in the paper are stored under ``data/`` and can be
recomputed/verified for internal consistency.
"""

__version__ = "1.0.0"

from . import metrics, specialty_filter, stats, synthetic, data_loader  # noqa: F401
