#!/usr/bin/env bash
# Dependency-light aggregate checks, dry-run pipeline, figures, and tests.
set -e
echo "==> reproduce_tables";        python scripts/reproduce_tables.py
echo; echo "==> synthetic threshold demonstration";   python scripts/run_threshold_sweep.py
echo; echo "==> synthetic statistical demonstration"; python scripts/run_statistical_analysis.py
echo; echo "==> adapt synthetic EHR export"; EHR_PATIENT_HASH_SALT=synthetic-test-only python scripts/adapt_ehr_export.py --input data/examples/raw_ehr_export.jsonl --output output/dry_run/adapted_records.jsonl --knowledge-base data/examples/specialty_knowledge_base.csv --field-map-json configs/bioarc_field_map.example.json
echo; echo "==> validate example records"; python scripts/validate_experiment.py --records data/examples/records.jsonl
echo; echo "==> hash dry-run artifacts"; python scripts/create_manifest.py --output output/dry_run/manifest.json configs/dry_run.json data/examples/records.jsonl data/examples/specialty_knowledge_base.csv
echo; echo "==> specialty-aware dry-run"; python scripts/run_inference.py --config configs/dry_run.json --backend dry-run --condition specialty-aware
echo; echo "==> specialty-agnostic dry-run"; python scripts/run_inference.py --config configs/dry_run.json --backend dry-run --condition specialty-agnostic
echo; echo "==> extractive baseline dry-run"; python scripts/run_biobert_baseline.py --config configs/dry_run.json --backend dry-run
echo; echo "==> evaluate dry-run"; python scripts/evaluate_predictions.py --records data/examples/records.jsonl --predictions output/dry_run/predictions_aware.jsonl --per-record-csv output/dry_run/per_record_scores.csv --summary-json output/dry_run/evaluation_summary.json
echo; echo "==> reader study dry-run"; python scripts/analyze_reader_study.py --ratings data/examples/reader_ratings.csv --output output/dry_run/reader_analysis.json
echo; echo "==> latency dry-run"; python scripts/benchmark_latency.py --records data/examples/records.jsonl --raw-csv output/dry_run/latency_raw.csv --summary-json output/dry_run/latency_summary.json --warmup 1 --repeats 5
echo; echo "==> make_figures";      python scripts/make_figures.py
echo; echo "==> make improved figure alternatives"; python scripts/make_improved_figures.py
echo; echo "==> tests";             python tests/run_tests.py
