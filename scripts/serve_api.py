#!/usr/bin/env python3
"""Serve the JSON EHR integration boundary with FastAPI/Uvicorn."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ehr_summ.config import load_config
from ehr_summ.engine import build_engine
from ehr_summ.service import create_app


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--backend", choices=["dry-run", "medgemma"], default="dry-run")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("install the 'serve' optional dependencies") from exc
    app = create_app(build_engine(load_config(args.config), args.backend))
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
