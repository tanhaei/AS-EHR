# Release 2.0.0 verification notes

Verified on 2026-07-11 with Python 3.12 in the dependency-light environment.

- `bash run_all.sh`: passed.
- Fallback test runner: 48 passed, 0 failed.
- All Python source, scripts, and tests compiled successfully.
- Synthetic EHR adaptation, aware/agnostic inference, extractive baseline,
  record-level evaluation, paired reader analysis, latency evidence capture,
  table checks, and figure regeneration completed.
- Improved figures were visually inspected in raster form and are also emitted
  as PDF and SVG.

MedGemma 27B and BioBERT model execution was not performed in this environment
because model weights, governed clinical data, and the required GPU were not
provided. Their code paths are optional and explicitly separated from the
dependency-light dry-run. Successful dry-run execution is not clinical
validation and must not be reported as reproduction of the manuscript results.
