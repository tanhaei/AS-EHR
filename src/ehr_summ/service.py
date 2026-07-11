"""Optional FastAPI adapter matching the EHR JSON request/response boundary."""
from __future__ import annotations

from .engine import InferenceEngine
from .schemas import ClinicalRecord


def create_app(engine: InferenceEngine):  # pragma: no cover - optional service dependency
    try:
        from fastapi import FastAPI, HTTPException
    except ImportError as exc:
        raise RuntimeError("install the 'serve' optional dependencies") from exc

    app = FastAPI(title="Specialty-Aware EHR Summarization", version="2.0.0")

    @app.get("/health")
    def health():
        return {"status": "ok", "model_id": getattr(engine.summarizer, "model_id", "unknown")}

    @app.post("/v1/summaries")
    def summarize(payload: dict):
        try:
            condition = payload.pop("condition", "specialty-aware")
            record = ClinicalRecord.from_dict(payload)
            return engine.summarize(record, condition).__dict__
        except (TypeError, ValueError, KeyError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app

