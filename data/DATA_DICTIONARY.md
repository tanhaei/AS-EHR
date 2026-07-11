# Public data dictionary

No clinical text or direct patient identifier is required for the statistical
reproduction layer. Institutions should export only governed, derived fields.

## Clinical-record JSONL

- `record_id`: study-scoped identifier; never a hospital encounter identifier.
- `patient_id_hash`: salted study hash used only to prevent split leakage.
- `specialty`, `site_id`, `language_profile`: stratification variables.
- `note`: present only in private governed datasets; examples are synthetic.
- `features`: normalized source facts with stable `feature_id` values.
- `required_feature_ids`: clinician-adjudicated must-include facts.
- `code_switched_terms`: term-level denominator plus acceptable renderings.
- `reference_summary`: clinician-authored target.

## Public per-record derived score CSV

Recommended non-PHI release columns:

`record_hash, patient_split, specialty, site_id, language_profile, condition,
precision, recall, f1, rouge_1, rouge_2, rouge_l,
unsupported_feature_count, predicted_feature_count,
mandatory_omission_count, mandatory_feature_count,
code_switch_correct, code_switch_total`

Release requires institutional governance review even when clinical text is
absent.

## Threshold-tuning evidence CSV

Required columns are `similarity` and `is_required`. Optional safe stratifiers
include `specialty`, `site_id`, and a study-scoped patient-cluster hash. One row
represents one candidate clinical feature, not one patient record.

## Reader-study CSV

`evaluator_hash, record_id, condition, specialty, relevance, clarity, utility,
time_seconds`

The evaluator and record identifiers must be study-scoped hashes. Each paired
comparison requires both conditions for the same evaluator-record pair.
