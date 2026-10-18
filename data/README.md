# Dataset

**Name:** resume-job-description-fit
**Source:** https://huggingface.co/datasets/cnamuangtoun/resume-job-description-fit
**Inspected revision:** `08978e21714984bb417547d2c0f9b477f5298163` (inspected 13 October 2026)

## Contents (from our audit)

| Item | Value |
|---|---|
| Columns | `resume_text`, `job_description_text`, `label` |
| Rows | 8,000 (6,241 train / 1,759 test) |
| Labels | No Fit 4,000 · Potential Fit 2,000 · Good Fit 2,000 |
| Unique CVs / jobs | 643 / 351 |
| Conflicting duplicate pairs | 6 pairs (12 rows) — to be removed |
| Train–test overlap | 476 identical CVs shared; no identical job descriptions shared |

Full counts and file hashes: `../results/dataset_audit.json`.

**Status:** Approved by the instructor for this project.

## Known limitations

- **No dataset card or licence** in the inspected revision, so permission to use is not documented.
- **Original source unknown:** the repository does not say where the CVs and job descriptions come from.
- **Label origin unknown:** it is not documented who or what assigned the labels. The exact 50% / 25% / 25%
  class split suggests the pairs were sampled to a fixed ratio; this does not tell us how labels were judged.
- Many CVs appear in both the official train and test files, so the test measures **unseen jobs with
  possibly familiar CVs**, not unseen candidates.
- English text, mainly US job postings.

## Rules for this repository

- **Raw CV text is not committed** while usage permission is unresolved. Keep downloads in `data/raw/`
  (ignored by git).
- Do not relabel or redistribute the data.
