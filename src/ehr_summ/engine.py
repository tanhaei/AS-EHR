"""Reusable inference engine for batch scripts and the REST adapter."""
from __future__ import annotations

from dataclasses import dataclass

from .config import deep_get
from .embeddings import HashingEmbedder, MedGemmaHiddenStateEmbedder, TextEmbedder
from .knowledge_base import build_centroids, load_knowledge_base
from .modeling import DryRunSummarizer, MedGemmaSummarizer
from .pipeline import select_agnostic_features, select_specialty_features
from .preprocessing import normalize_record
from .schemas import ClinicalRecord, SummaryPrediction


@dataclass
class InferenceEngine:
    config: dict
    embedder: TextEmbedder
    summarizer: object
    centroids: dict

    def summarize(self, record: ClinicalRecord, condition: str = "specialty-aware") -> SummaryPrediction:
        record = normalize_record(record)
        k_values = deep_get(self.config, "filter.K_by_specialty", {})
        K = int(k_values.get(record.specialty, deep_get(self.config, "filter.default_K", 12)))
        safety = bool(deep_get(self.config, "filter.safety_layer", True))
        if condition == "specialty-aware":
            if record.specialty not in self.centroids:
                raise ValueError(f"no knowledge-base concepts for {record.specialty}")
            features = select_specialty_features(
                record,
                self.centroids[record.specialty],
                self.embedder,
                tau=float(deep_get(self.config, "filter.tau", 0.75)),
                K=K,
                safety_layer=safety,
            )
        elif condition == "specialty-agnostic":
            features = select_agnostic_features(record, K=K, safety_layer=safety)
        else:
            raise ValueError("condition must be specialty-aware or specialty-agnostic")
        return self.summarizer.summarize(record, features, condition=condition)


def build_engine(config: dict, backend: str = "dry-run") -> InferenceEngine:
    if backend == "dry-run":
        embedder = HashingEmbedder(int(deep_get(config, "embedding.dimension", 256)))
        summarizer = DryRunSummarizer()
    elif backend == "medgemma":
        model_id = deep_get(config, "model.model_id", "google/medgemma-27b-it")
        summarizer = MedGemmaSummarizer(
            model_id,
            adapter_path=deep_get(config, "model.adapter_path"),
            generation=deep_get(config, "generation", {}),
        )
        embedder = MedGemmaHiddenStateEmbedder(
            model_id,
            layer=int(deep_get(config, "embedding.layer", -1)),
            pooling=deep_get(config, "embedding.pooling", "mean"),
            max_length=int(deep_get(config, "embedding.max_length", 512)),
            model=summarizer.model,
            tokenizer=summarizer.tokenizer,
        )
    else:
        raise ValueError("backend must be dry-run or medgemma")
    concepts = load_knowledge_base(deep_get(config, "data.knowledge_base_csv"))
    return InferenceEngine(config, embedder, summarizer, build_centroids(concepts, embedder))

