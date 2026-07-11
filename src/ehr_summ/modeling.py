"""Summarizer backends for deterministic dry-runs and real MedGemma inference."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Sequence

from .prompting import SYSTEM_PROMPT, build_user_prompt
from .schemas import ClinicalFeature, ClinicalRecord, SummaryPrediction


def _extract_json(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("model output does not contain a JSON object")
    return json.loads(text[start : end + 1])


@dataclass
class DryRunSummarizer:
    model_id: str = "dry-run-grounded-summarizer"

    def summarize(self, record: ClinicalRecord, features: Sequence[ClinicalFeature],
                  *, condition: str = "specialty-aware") -> SummaryPrediction:
        started = time.perf_counter()
        ordered = list(features)
        summary = "؛ ".join(feature.text for feature in ordered)
        return SummaryPrediction(
            record_id=record.record_id,
            condition=condition,
            summary=summary,
            included_feature_ids=[feature.feature_id for feature in ordered],
            model_id=self.model_id,
            latency_seconds=time.perf_counter() - started,
        )


class MedGemmaSummarizer:
    def __init__(self, model_id: str = "google/medgemma-27b-it", *, adapter_path: str | None = None,
                 device_map: str = "auto", torch_dtype: str = "bfloat16",
                 generation: dict | None = None):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("install the 'model' optional dependencies") from exc
        dtype = getattr(torch, torch_dtype)
        self._torch = torch
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, device_map=device_map
        )
        if adapter_path:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError("PEFT is required to load a LoRA adapter") from exc
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()
        self.generation = {"max_new_tokens": 512, "do_sample": False}
        self.generation.update(generation or {})

    def summarize(self, record: ClinicalRecord, features: Sequence[ClinicalFeature],
                  *, condition: str = "specialty-aware") -> SummaryPrediction:  # pragma: no cover
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(record, features)},
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        device = self.model.get_input_embeddings().weight.device
        batch = self.tokenizer(text, return_tensors="pt").to(device)
        started = time.perf_counter()
        with self._torch.inference_mode():
            generated = self.model.generate(**batch, **self.generation)
        latency = time.perf_counter() - started
        continuation = generated[0, batch["input_ids"].shape[1] :]
        raw = self.tokenizer.decode(continuation, skip_special_tokens=True)
        parsed = _extract_json(raw)
        return SummaryPrediction(
            record_id=record.record_id,
            condition=condition,
            summary=str(parsed["summary"]),
            included_feature_ids=list(parsed.get("included_feature_ids", [])),
            model_id=self.model_id,
            latency_seconds=latency,
            raw_output=raw,
        )

