# Specialty-Aware Multilingual EHR Summarization — Reproduction Repository

Reproduction harness for the paper **“Automatic Summarization of Electronic
Health Records Using Large Language Models: A Specialty-Aware, Multilingual
Integration with a Hospital Information System”** (MedGemma 27B + BioArc).

This repository provides runnable, tested implementations of the paper's
**methods, metrics, and statistical analysis**, the paper's **reported results**
as machine-readable CSVs, scripts that **recompute and verify** the headline
numbers for internal consistency, a **calibrated synthetic-data harness** so the
full statistical pipeline runs end-to-end, and code to **regenerate the figures**.

> **Important — what is and isn't here.**
> The clinical corpus (de-identified BioArc records, MIMIC-IV mix) and the
> fine-tuned MedGemma-27B weights are **not included**: they are proprietary and
> privacy-restricted (see the paper's *Availability of data and material*).
> Per-record gold scores are therefore not public either. To keep the pipeline
> fully executable, `ehr_summ.synthetic` generates per-record scores **calibrated
> to the paper's reported per-specialty means and effect sizes**. These are a
> reproduction aid, **not** the original clinical data, and are clearly labelled
> as synthetic throughout.

---

## Paper at a glance

- **Problem.** Information overload in EHRs increases clinician cognitive load;
  the burden is worse for low-resource languages such as **Persian**, where
  labelled clinical corpora and tooling are scarce.
- **System.** A fine-tuned **MedGemma 27B** model integrated as a prototype in
  the **BioArc** hospital information system. It processes structured fields
  (diagnoses, medications, labs coded with **ICD-11 / SNOMED-CT**) together with
  unstructured Persian / Persian–English **code-switched** notes.
- **Core idea — specialty-aware filtering.** Each specialty is a centroid
  embedding `B`; a candidate clinical feature `A` is scored by **cosine
  similarity**; features above threshold **τ = 0.75** are kept and the top-`K`
  are summarized (Algorithm 1).
- **Headline results (1,000 de-identified records, six specialties).**
  - Pooled **F1 = 0.83** (95% CI [0.825, 0.839]); macro **ROUGE-N = 0.855**.
  - Beats a matched extractive **BioBERT** baseline (0.83 vs 0.72; paired
    *p* < 0.001; **Cohen's d = 1.07**); per-specialty gains significant after
    **Benjamini–Hochberg** correction.
  - Specialty-aware **A/B ablation**: clinical-utility 4.25 vs 3.78 (d = 0.65).
  - Code-switched term accuracy **85.2%** on the 707 code-switched records.
  - Reader study: **130 clinicians**, satisfaction **4.4/5**, coverage **4.5/5**.
  - Single-A100 deployment: median latency **5.3 s/summary**.
- **Out of scope / future work.** Handwritten-Persian OCR, demographic fairness,
  and external validation across other EHR platforms.

---

## Repository layout

```
ehr-specialty-summarization/
├── README.md
├── pyproject.toml            # installable package (src layout) + pytest config
├── requirements.txt
├── run_all.sh               # run every script + the test suite
├── data/                    # paper-reported numbers (Tables 3–7, A.8, appendix)
│   ├── specialty_distribution.csv
│   ├── threshold_sensitivity.csv
│   ├── baseline_comparison.csv
│   ├── specialty_metrics.csv
│   ├── expert_feedback.csv
│   ├── f1_confidence_intervals.csv
│   ├── rouge_l_w.csv
│   └── specialty_embedding_construction.csv
├── src/ehr_summ/            # the library
│   ├── metrics.py            # Precision/Recall/F1 (Eqs. 1–2), ROUGE-N (Eq. 3)
│   ├── specialty_filter.py   # Algorithm 1: cosine-similarity ranking + τ + top-K
│   ├── stats.py              # bootstrap CI, paired t/Wilcoxon, Cohen's d, BH-FDR
│   ├── synthetic.py          # calibrated synthetic per-record scores
│   └── data_loader.py        # load the CSVs above
├── scripts/
│   ├── reproduce_tables.py        # recompute & verify paper aggregates
│   ├── run_threshold_sweep.py     # τ sweep → selection (Table 4 methodology)
│   ├── run_statistical_analysis.py# paired tests + Cohen's d + BH (Table 5)
│   └── make_figures.py            # regenerate Figures 7–8 + Table 5 chart
├── tests/                   # pytest-style tests + a stdlib runner
│   ├── test_metrics.py
│   ├── test_specialty_filter.py
│   ├── test_stats.py
│   ├── test_reproduce.py
│   └── run_tests.py          # `python tests/run_tests.py` (no pytest needed)
└── figures/                 # output of make_figures.py
```

---

## Installation

Requires Python ≥ 3.10.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# optional, editable install so `import ehr_summ` works anywhere:
pip install -e .
```

The scripts also add `src/` to `sys.path` automatically, so they run from a bare
checkout without installation.

---

## Quick start

Run everything (scripts + tests):

```bash
bash run_all.sh
```

Or individually:

```bash
python scripts/reproduce_tables.py          # verify paper's aggregate numbers
python scripts/run_threshold_sweep.py       # τ-selection methodology (Table 4)
python scripts/run_statistical_analysis.py  # paired tests, Cohen's d, BH-FDR
python scripts/make_figures.py              # write figures/*.png
```

### What `reproduce_tables.py` checks (recomputed from `data/`)

| Quantity | Computed | Paper |
|---|---|---|
| Total records | 1000 | 1000 |
| Code-switched records | 707 | 707 |
| Pooled F1 (record-weighted) | 0.833 | 0.83 |
| Macro-average ROUGE-N | 0.855 | 0.855 |
| Pooled code-switched accuracy | 0.853 | 0.852 |
| Total experts | 130 | 130 |
| Mean satisfaction / coverage | 4.37 / 4.48 | 4.4 / 4.5 |

### What `run_statistical_analysis.py` reproduces (calibrated synthetic data)

- Pooled **Cohen's d = 1.07** (paper 1.07), paired **p < 0.001**.
- All six per-specialty comparisons **significant after Benjamini–Hochberg**.
- Pooled F1 95% bootstrap CI ≈ [0.82, 0.84] (paper [0.825, 0.839]).

---

## Using the library

```python
import numpy as np
from ehr_summ.specialty_filter import build_specialty_embedding, rank_and_select
from ehr_summ.metrics import prf_from_sets, rouge_n
from ehr_summ.stats import paired_test, benjamini_hochberg, bootstrap_ci

# 1) Specialty centroid from seed-concept embeddings, then Algorithm 1.
specialty_vec = build_specialty_embedding([emb1, emb2, emb3])   # vector B
top = rank_and_select(features, specialty_vec, embed=my_encoder, tau=0.75, K=12)

# 2) Feature-level evaluation against a checklist-derived gold set.
p, r, f1 = prf_from_sets(predicted_features, gold_features)

# 3) Statistics.
res = paired_test(proposed_f1_per_record, baseline_f1_per_record)
q_adj, reject = benjamini_hochberg([res.p_value, ...], q=0.05)
est, lo, hi = bootstrap_ci(proposed_f1_per_record, n_boot=1000, ci=95)
```

To run the real experiment on your own data, replace `my_encoder` with a
MedGemma (or other) embedding function and feed real per-record gold sets to the
metrics — every interface is encoder-agnostic.

---

## Testing

```bash
pytest -q                 # if pytest is installed
python tests/run_tests.py # zero-dependency fallback runner
```

The suite covers metric correctness (including ROUGE-N n-gram clipping),
Algorithm 1 thresholding/ranking, the statistics (bootstrap, paired tests,
Cohen's d, BH-FDR), and end-to-end reproduction of the paper's aggregates and
the synthetic calibration. Current status: **29 passing**.

---

## Notes on faithfulness & limitations

- **Synthetic ≠ clinical.** The synthetic generator reproduces the *reported
  summary statistics*, not the underlying records. It cannot validate the
  model's clinical correctness — only that the analysis pipeline yields the
  reported inferential results given those statistics.
- **Threshold arg-max.** The exact F1-optimal τ depends on the true similarity
  distribution (not public). The sweep script verifies the robust, reproducible
  facts: the monotone precision↑/recall↓ trade-off and an F1 maximum in the
  paper's plateau region {0.70, 0.75}.
- **No PHI.** No protected health information is included anywhere in this repo.

## Citation

If you use this code, please cite the original paper (see the manuscript for the
full reference) and link back to this reproduction repository.

## License

MIT — see [LICENSE](LICENSE).
