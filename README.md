# Specialty-Aware Multilingual EHR Summarization

Public implementation and evidence pipeline for the manuscript:

> **Automatic Summarization of Electronic Health Records Using Large Language
> Models: A Specialty-Aware, Multilingual Integration with a Hospital
> Information System**

The repository contains two deliberately separated layers:

1. **Executable method implementation** — Persian normalization, patient-level
   splitting, explicit embedding extraction, specialty filtering, an optional
   safety layer, LoRA training, MedGemma inference, a matched extractive
   BioBERT baseline, record-level evaluation, reader-study analysis, and raw
   latency benchmarking.
2. **Legacy aggregate-consistency harness** — the paper-reported tables and
   clearly labelled synthetic demonstrations retained from version 1. These
   demonstrate the analysis functions but do **not** validate the clinical
   results.

## Evidence boundary

The public example records and reader ratings are synthetic. The BioArc corpus,
original patient-level outputs, and fine-tuned adapter are not included. No
clinical-performance claim can be recovered from the synthetic examples or
from `ehr_summ.synthetic`.

The repository can independently reproduce a paper result only after the
governed experiment artifacts described in
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) are supplied. The README and command
output use **demonstrate**, **aggregate consistency**, and **reproduce**
carefully to preserve this distinction.

## What is implemented

- Persian/Arabic Unicode, digit, whitespace, and half-space normalization.
- Persian, English, and code-switch language-profile detection.
- Public JSONL schemas with stable source-feature identifiers.
- Configurable adapter from governed BioArc/EHR JSON exports to canonical
  structured and note-derived clinical features.
- Deterministic patient-level train/tune/test splitting and leakage checks.
- Direct-identifier redaction reference code with an explicit validation warning.
- Three embedding backends:
  - dependency-light hashing encoder for tests only;
  - mean-pooled BioBERT-compatible encoder;
  - MedGemma hidden-state encoder with explicit layer, pooling, and L2 normalization.
- Specialty centroid construction from a versioned knowledge-base CSV.
- Original threshold + top-K filtering and an optional mandatory-fact safety channel.
- Grounded JSON prompt with source-feature traceability.
- MedGemma LoRA training and adapter-aware inference.
- Shared MedGemma model loading for generation and hidden-state embeddings, so
  the 27B checkpoint is not duplicated in GPU memory.
- Optional FastAPI adapter for the EHR JSON request/response boundary.
- Matched-word-budget extractive BioBERT baseline.
- Per-record Precision/Recall/F1, Persian-normalized ROUGE-1/2/L, unsupported
  feature rate, mandatory omission rate, and term-level code-switch accuracy.
- Patient-cluster bootstrap confidence intervals.
- Paired clinician outcomes and optional crossed mixed-effects analysis.
- Raw latency observations, median, p95, mean, and sequential throughput.
- CI and a dependency-light end-to-end dry-run.

## Quick start: no model download

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[test]
bash run_all.sh
```

The command validates six synthetic records, runs specialty-aware and
specialty-agnostic inference, runs the extractive baseline, evaluates the
predictions, analyzes a paired reader-study example, benchmarks raw latency,
regenerates figures, and runs the full test suite.

Dry-run results are written under `output/dry_run/` and are ignored by Git.
They are implementation checks, not manuscript results.

## Real MedGemma setup

Install the optional model and analysis dependencies:

```bash
pip install -r requirements-model.txt
```

Accept the model terms and authenticate with the model host before loading
`google/medgemma-27b-it`. Copy
[`configs/medgemma_lora.example.json`](configs/medgemma_lora.example.json), then
replace every value marked as illustrative with the original experiment log.

The exact values currently supported by the manuscript are isolated in
[`configs/paper_reported.json`](configs/paper_reported.json). Unknown values are
`null` on purpose, and this file fails training validation until they are
recovered. This prevents an example default from being mistaken for an
experimental fact.

### Prepare patient-level splits

```bash
export EHR_PATIENT_HASH_SALT='institution-held-secret'
python scripts/adapt_ehr_export.py \
  --input data/private/raw_export.jsonl \
  --output data/private/records.jsonl \
  --knowledge-base data/private/specialty_knowledge_base.csv \
  --field-map-json configs/bioarc_field_map.local.json

python scripts/prepare_splits.py \
  --input data/private/records.jsonl \
  --output-dir data/private/splits \
  --seed 2026

python scripts/validate_experiment.py \
  --train data/private/splits/train.jsonl \
  --tune data/private/splits/tune.jsonl \
  --test data/private/splits/test.jsonl
```

### Validate and run LoRA training

```bash
python scripts/train_lora.py \
  --config configs/medgemma_lora.local.json \
  --validate-only

python scripts/train_lora.py \
  --config configs/medgemma_lora.local.json
```

### Run the controlled ablation

The same model, records, generation settings, and output budget are used; only
the specialty filter changes.

```bash
python scripts/run_inference.py \
  --config configs/medgemma_lora.local.json \
  --backend medgemma \
  --condition specialty-aware

python scripts/run_inference.py \
  --config configs/medgemma_lora.local.json \
  --backend medgemma \
  --condition specialty-agnostic
```

### Run the extractive baseline

```bash
python scripts/run_biobert_baseline.py \
  --config configs/medgemma_lora.local.json \
  --backend biobert
```

### Serve the EHR integration boundary

```bash
pip install -e .[serve]
python scripts/serve_api.py \
  --config configs/medgemma_lora.local.json \
  --backend medgemma \
  --host 127.0.0.1 \
  --port 8000
```

The service exposes `GET /health` and `POST /v1/summaries`. The request body is
the canonical clinical-record schema plus an optional `condition` field. FastAPI
publishes the generated OpenAPI description at `/docs`; production deployment
still requires the hospital's authentication, authorization, audit logging,
rate limiting, and network controls.

### Evaluate and retain record-level evidence

```bash
python scripts/evaluate_predictions.py \
  --records data/private/splits/test.jsonl \
  --predictions output/medgemma_lora/predictions_aware.jsonl \
  --per-record-csv output/medgemma_lora/per_record_aware.csv \
  --summary-json output/medgemma_lora/summary_aware.json
```

Code-switch accuracy is pooled from the exact number of correct and total
terms—not from record counts. Confidence intervals resample patients rather
than assuming repeated records are independent.

### Analyze the blinded reader study

```bash
python scripts/analyze_reader_study.py \
  --ratings data/private/reader_ratings.csv \
  --output output/medgemma_lora/reader_analysis.json \
  --mixed-effects
```

Required columns are documented in [`data/DATA_DICTIONARY.md`](data/DATA_DICTIONARY.md).
The paired analysis retains relevance, clarity, utility, and task time. The
optional mixed model includes evaluator and record random effects.

### Benchmark latency

```bash
python scripts/benchmark_latency.py \
  --records data/private/splits/test.jsonl \
  --backend medgemma \
  --model-id google/medgemma-27b-it \
  --adapter-path output/medgemma_lora/final_adapter \
  --warmup 10 \
  --repeats 100 \
  --raw-csv output/medgemma_lora/latency_raw.csv \
  --summary-json output/medgemma_lora/latency_summary.json
```

Record batch size, input/output token counts, serving framework, concurrency,
GPU model, software versions, and warm/cold-cache conditions must accompany any
reported latency or throughput value.

## Repository layout

```text
configs/                    provenance-aware experiment configurations
data/
  examples/                 synthetic schema and dry-run data
  *.csv                     legacy paper-reported aggregate tables
scripts/
  adapt_ehr_export.py        configurable EHR-to-canonical adapter
  prepare_splits.py         patient-level splitting
  train_lora.py             real LoRA training / validation
  run_inference.py          aware and agnostic MedGemma inference
  run_biobert_baseline.py   matched extractive comparator
  evaluate_predictions.py  record-level evidence and cluster bootstrap
  compare_systems.py        real paired system comparison + BH-FDR
  create_manifest.py        SHA-256 evidence/config manifest
  analyze_reader_study.py   paired and mixed-effects analysis
  benchmark_latency.py      raw latency evidence
  serve_api.py              optional EHR-facing REST adapter
  validate_experiment.py    schema and leakage checks
  reproduce_tables.py       legacy aggregate consistency only
  run_threshold_sweep.py    synthetic method demonstration only
  run_statistical_analysis.py synthetic method demonstration only
  make_improved_figures.py reviewer-friendly CI and reader-study plots
src/ehr_summ/               reusable pipeline modules
tests/                      unit and end-to-end dry-run tests
```

## Knowledge base and safety layer

The public knowledge-base schema is:

```text
specialty, concept_id, descriptor, mandatory, synonyms
```

The example file contains only synthetic demonstration concepts. The original
650-concept list or an institution-approved version must be supplied for the
real experiment. If SNOMED CT content is distributed, its licensing terms must
be respected.

When `filter.safety_layer=true`, mandatory facts—such as allergies,
anticoagulation, pregnancy status, critical laboratory values, or other
institution-defined safety concepts—are retained independently of specialty
similarity. To reproduce the manuscript's original Algorithm 1 without this
extension, set it to `false` and report that choice.

## Legacy paper aggregates

The CSV files directly under `data/` contain values transcribed from the paper.
They support arithmetic checks and figure regeneration. They are not raw data.

- `scripts/reproduce_tables.py` checks arithmetic consistency.
- `scripts/run_threshold_sweep.py` runs a synthetic threshold demonstration; it
  may select 0.70 instead of the paper's 0.75.
- `scripts/run_statistical_analysis.py` uses synthetic differences calibrated
  to the reported means and effect sizes. Its p-values cannot validate the
  clinical experiment.

With an approved real tuning CSV, the same threshold script analyzes genuine
evidence instead:

```bash
python scripts/run_threshold_sweep.py \
  --input-csv data/private/threshold_evidence.csv \
  --output-csv output/medgemma_lora/threshold_sweep.csv
```

After evaluating proposed and comparator predictions separately, run the real
paired comparison:

```bash
python scripts/compare_systems.py \
  --proposed output/medgemma_lora/per_record_aware.csv \
  --comparator output/medgemma_lora/per_record_baseline.csv \
  --output output/medgemma_lora/paired_comparison.json
```

## Data governance

- Never commit direct patient identifiers, raw clinical exports, access tokens,
  model secrets, or governed adapters.
- `data/private/`, model weights, checkpoints, and outputs are ignored by Git.
- The provided redaction regexes are not a substitute for a validated
  institutional de-identification process.
- Even derived per-record scores require governance approval before release.

## Recommended manuscript availability statement

Until governed per-record artifacts are released, use wording such as:

> Code for preprocessing, specialty filtering, model training and inference,
> metric computation, statistical procedures, and table/figure regeneration is
> publicly available. Clinical records, original per-record text, and the
> fine-tuned adapter are restricted by institutional data-governance
> requirements. Synthetic examples are provided solely to validate software
> execution and do not reproduce the reported clinical findings.

## License and citation

Code is released under the MIT License. See [`CITATION.cff`](CITATION.cff) and
cite the associated manuscript when using the pipeline.
