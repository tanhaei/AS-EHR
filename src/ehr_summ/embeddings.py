"""Embedding backends for dry-runs, BioBERT, and MedGemma hidden states."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

from .preprocessing import tokenize_clinical


class TextEmbedder(Protocol):
    model_id: str

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


@dataclass
class HashingEmbedder:
    """Deterministic dependency-light encoder for tests and dry-runs only."""

    dimension: int = 256
    model_id: str = "dry-run-hashing-encoder"

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in tokenize_clinical(text):
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
                index = int.from_bytes(digest[:8], "big") % self.dimension
                sign = 1.0 if digest[8] & 1 else -1.0
                matrix[row, index] += sign
            norm = np.linalg.norm(matrix[row])
            if norm:
                matrix[row] /= norm
        return matrix


class TransformerMeanPoolEmbedder:
    """Mean-pooled encoder backend, suitable for an explicit BioBERT baseline."""

    def __init__(self, model_id: str, *, device: str | None = None, max_length: int = 512):
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional heavy dependency
            raise RuntimeError("install the 'model' optional dependencies") from exc
        self._torch = torch
        self.model_id = model_id
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()

    def encode(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover - needs model
        torch = self._torch
        batch = self.tokenizer(list(texts), padding=True, truncation=True,
                               max_length=self.max_length, return_tensors="pt")
        batch = {key: value.to(self.device) for key, value in batch.items()}
        with torch.inference_mode():
            hidden = self.model(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        pooled = torch.nn.functional.normalize(pooled.float(), p=2, dim=1)
        return pooled.cpu().numpy()


class MedGemmaHiddenStateEmbedder:
    """Explicit hidden-state embedding extraction from MedGemma.

    The manuscript's previous ``MedGemma.Embed`` pseudocode was ambiguous.
    This backend makes the layer, pooling operation, and L2 normalization
    explicit and records them in the experiment configuration.
    """

    def __init__(
        self,
        model_id: str = "google/medgemma-27b-it",
        *,
        layer: int = -1,
        pooling: str = "mean",
        max_length: int = 512,
        device_map: str = "auto",
        model=None,
        tokenizer=None,
    ):
        if pooling not in {"mean", "last-token"}:
            raise ValueError("pooling must be 'mean' or 'last-token'")
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("install the 'model' optional dependencies") from exc
        self._torch = torch
        self.model_id = model_id
        self.layer = layer
        self.pooling = pooling
        self.max_length = max_length
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_id)
        self.model = model or AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map=device_map
        )
        self.model.eval()

    def encode(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover
        torch = self._torch
        device = self.model.get_input_embeddings().weight.device
        batch = self.tokenizer(list(texts), padding=True, truncation=True,
                               max_length=self.max_length, return_tensors="pt")
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.inference_mode():
            output = self.model(**batch, output_hidden_states=True, use_cache=False)
        hidden = output.hidden_states[self.layer]
        mask = batch["attention_mask"]
        if self.pooling == "last-token":
            indices = mask.sum(1) - 1
            pooled = hidden[torch.arange(hidden.shape[0], device=device), indices]
        else:
            expanded = mask.unsqueeze(-1)
            pooled = (hidden * expanded).sum(1) / expanded.sum(1).clamp(min=1)
        pooled = torch.nn.functional.normalize(pooled.float(), p=2, dim=1)
        return pooled.cpu().numpy()
