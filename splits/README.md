# Data splits

`splits.csv` and `manifest.json` are produced by `notebooks/HireMind_M0_Baseline.ipynb` (section 3) and contain **IDs only, no CV or job text**.

| Column | Meaning |
|---|---|
| `row_uid` | `<original file>_<row index>` in the pinned dataset revision |
| `split` | `train`, `validation` or `test` |
| `cv_id`, `job_id` | first 16 hex characters of SHA-256 of the whitespace-normalised, lower-cased text |
| `job_group` | job ID after merging near-duplicate job descriptions (TF-IDF cosine ≥ 0.95) |
| `label`, `label_id` | No Fit = 0, Potential Fit = 1, Good Fit = 2 |

Policy (fixed before training, seed 42): remove all rows of label-conflicting pairs; keep one copy of exact duplicates and drop a training copy of any pair that also appears in test; drop training rows whose job is a near-duplicate of a test job; keep the official test file as the test set; validation = one fold of `StratifiedGroupKFold(5, shuffle=True, random_state=42)` grouped by job.
