# HireMind DL — Efficient CV–Job Fit Classification through Distillation and Quantization

**CS4085 Deep Learning · Effat University · Fall 2026**
**Team:** Rawan Khojah · Hams Aljohani
**Instructor:** Dr Naila Marir

> **Status:** Proposal and dataset audit prepared; model training has not started.

## Project description

HireMind is our Senior project: an AI-assisted recruitment platform. This Deep Learning project studies
one component that HireMind could later use: a small model that reads a **CV and a job description**
and predicts **No Fit / Potential Fit / Good Fit**.

The project has two connected parts, as required by the course:

| Part | What we investigate |
|---|---|
| **Part 1 — Deep learning** | Does teacher-guided training (knowledge distillation) improve a compact BERT classifier? |
| **Part 2 — Efficiency** | Does post-training INT8 quantization make that model smaller while keeping the improvement? |

**Research question:** Can teacher-guided training improve compact CV–job fit classification, and can
quantization reduce model size while preserving the improvement?

## Planned models

| Model | Definition |
|---|---|
| **Teacher** (support only) | `bert-base-uncased`, fine-tuned on the training split; its logits are cached for distillation |
| **M0** — baseline | BERT-Tiny (`google/bert_uncased_L-2_H-128_A-2`, ≈4.4M parameters) trained with hard-label cross-entropy |
| **M1** — DL change | Same student and initialization, trained with `0.5·CE + 0.5·T²·KL(teacher ‖ student)` |
| **M2** — efficiency | CPU copy of M1 with dynamic INT8 quantization (no extra training) |

**Hypotheses (exploratory targets):**
- **H1:** M1 improves test macro-F1 over M0 by at least 1.0 point.
- **H2:** M2 is at least 5% smaller than M1, loses at most 1.0 macro-F1 point, and keeps at least half of M1's gain over M0.

**Measurements:** macro-F1, accuracy, per-class scores, confusion matrix, parameter count, serialized size,
MACs/FLOPs, CPU median/p95 latency and peak memory, measured on stated hardware under the same settings.

## Repository structure

| Folder / file | Contents |
|---|---|
| `proposal/` | `proposal.tex` (ACL format) and the latest proposal PDF |
| `literature/` | `references.txt` and the research pack with our reading notes on the five papers |
| `scripts/` | `audit_dataset.py` — dataset-checking script (no training, no downloads) |
| `data/` | `README.md` with the dataset link, inspected version and known limitations. **Raw data is not stored here.** |
| `results/` | `dataset_audit.json` — a **data audit**, not model performance |
| `notebooks/` | Training and evaluation notebooks (to be added) |

## Setup

Requirements so far: Python 3 (standard library only for the audit).

```sh
# 1. Download the two CSV files from the dataset page (see data/README.md) into data/raw/ (not committed)
# 2. Run the audit
python scripts/audit_dataset.py data/raw/train.csv data/raw/test.csv --output results/dataset_audit.json
```

Training will use Google Colab with PyTorch and Hugging Face Transformers; exact versions will be pinned after the pilot.

## Progress

| Item | Status |
|---|---|
| Five-paper review and research question | ✅ Done |
| Dataset structure audit | ✅ Done (see `results/dataset_audit.json`) |
| Proposal draft (M0/M1/M2, H1/H2) | ✅ Draft — being revised by the team |
| Dataset provenance, permission and label origin | ⏳ Pending |
| Colab pilot | ⏳ Not started |
| Teacher, M0, M1, M2 training and evaluation | ⏳ Not started |
| Final report, presentation, poster | ⏳ Not started |

No model results are reported yet. Results will be added only after real experiments.

## Links

- Overleaf project: _add link_
- Portfolio website: _add link_
