"""Specialty-aware multilingual EHR summarization and evidence pipeline.

The package implements governed-data adaptation, Persian preprocessing,
patient-level splitting, explicit embeddings, specialty filtering, optional
LoRA training and MedGemma inference, a matched extractive baseline, clinical
error metrics, reader-study analysis, and latency evidence capture.

Original clinical records and model adapters are not included. Legacy
paper-reported aggregates and calibrated synthetic demonstrations are retained
only for arithmetic and software-path checks.
"""

__version__ = "2.0.0"

from . import (  # noqa: F401
    baseline,
    config,
    data_loader,
    deidentification,
    embeddings,
    engine,
    evaluation,
    feature_extraction,
    knowledge_base,
    latency,
    metrics,
    modeling,
    pipeline,
    preprocessing,
    prompting,
    reader_study,
    schemas,
    service,
    specialty_filter,
    splits,
    stats,
    synthetic,
)
