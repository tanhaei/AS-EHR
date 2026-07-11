# Reproducibility status and release checklist

This document prevents software availability from being confused with evidence
availability.

## Current public status

| Claim or component | Public implementation | Original evidence included |
|---|---:|---:|
| Persian normalization | Yes | Not required |
| Patient-level splitting | Yes | No original split manifest |
| MedGemma LoRA training | Yes | No original adapter/log |
| MedGemma inference | Yes | No original predictions |
| Hidden-state embedding | Yes, explicit | Original layer/pooling unknown |
| Specialty filter | Yes | Original 650 concepts absent |
| Mandatory safety layer | Yes, optional extension | Not part of reported experiment |
| BioBERT extractive baseline | Yes | Original baseline predictions absent |
| Feature F1 and ROUGE | Yes | Original per-record values absent |
| Code-switch term accuracy | Yes | Original correct/total counts absent |
| Reader-study analysis | Yes | Original rating matrix absent |
| Latency benchmark | Yes | Original raw latency samples absent |
| Aggregate tables/figures | Yes | Values transcribed from manuscript |

## Values that must be recovered from the original run

The manuscript reports LoRA rank 16, alpha 32, three epochs, learning rate
`2e-4`, and four A100 80GB GPUs. It does not currently establish:

- exact model repository ID and immutable revision;
- LoRA target modules and dropout;
- optimizer, scheduler, warm-up, and weight decay;
- batch sizes and gradient accumulation;
- random seeds and checkpoint-selection rule;
- prompt template and decoding parameters;
- maximum input/output lengths and truncation policy;
- hidden-state layer, pooling, and normalization used as embeddings;
- software/driver/CUDA versions;
- fine-tuning corpus sizes and patient-level split manifest.

Populate these fields from the original logs. Do not copy illustrative values
from `configs/medgemma_lora.example.json` into the manuscript as facts.

## Minimum governed derived release

If institutional approval allows, release a table with no clinical text and
study-scoped hashes containing:

- record hash and patient-cluster hash;
- split, site, specialty, and language profile;
- condition and model revision;
- per-record feature counts and F1;
- ROUGE-1/2/L;
- unsupported and mandatory-omission counts;
- correct and total code-switched terms;
- raw latency and input/output token counts.

For the reader study, release one row per evaluator-record-condition rating
with study-scoped hashes. This enables the paired and mixed-effects analyses
without publishing clinical text.

## Before tagging a manuscript release

- [ ] Freeze the exact manuscript commit and repository tag.
- [ ] Fill every null in `configs/paper_reported.json` from primary logs.
- [ ] Add hashes for train/tune/test manifests and assert patient non-overlap.
- [ ] Add the approved knowledge-base version and construction protocol.
- [ ] Run real MedGemma and baseline inference from a clean environment.
- [ ] Persist raw predictions before aggregation.
- [ ] Compute term-level code-switch denominators.
- [ ] Run cluster-aware confidence intervals.
- [ ] Run the blinded reader-study analysis from its raw matrix.
- [ ] Persist raw latency measurements and serving configuration.
- [ ] Run `pytest -q` and `bash run_all.sh` in CI.
- [ ] Confirm the manuscript numbers from generated artifacts, not typed CSVs.

Create a release manifest with:

```bash
python scripts/create_manifest.py \
  --output output/release_manifest.json \
  configs/medgemma_lora.local.json \
  data/private/splits/train.jsonl \
  data/private/splits/tune.jsonl \
  data/private/splits/test.jsonl
```
