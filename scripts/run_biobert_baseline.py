#!/usr/bin/env python3
"""Run a reproducible, matched-word-budget extractive baseline."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ehr_summ.baseline import extractive_summary
from ehr_summ.config import deep_get, load_config
from ehr_summ.embeddings import HashingEmbedder, TransformerMeanPoolEmbedder
from ehr_summ.preprocessing import normalize_persian
from ehr_summ.schemas import SummaryPrediction, load_records, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--backend", choices=["dry-run", "biobert"], default="dry-run")
    args = parser.parse_args()
    config = load_config(args.config)
    model_id = deep_get(config, "baseline.model_id", "dmis-lab/biobert-base-cased-v1.2")
    embedder = (HashingEmbedder(int(deep_get(config, "embedding.dimension", 256)))
                if args.backend == "dry-run" else TransformerMeanPoolEmbedder(model_id))
    predictions = []
    for record in load_records(deep_get(config, "data.input_jsonl")):
        started = time.perf_counter()
        result = extractive_summary(record.note, [feature.text for feature in record.features], embedder,
                                    max_words=int(deep_get(config, "baseline.max_words", 200)))
        normalized_summary = normalize_persian(result.summary).casefold()
        included = [feature.feature_id for feature in record.features
                    if normalize_persian(feature.text).casefold() in normalized_summary]
        predictions.append(SummaryPrediction(record.record_id, "biobert-extractive", result.summary,
                                             included, model_id, time.perf_counter() - started))
    output = deep_get(config, "baseline.output_jsonl")
    write_jsonl(output, (prediction.__dict__ for prediction in predictions))
    print(f"Wrote {len(predictions)} baseline predictions to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

