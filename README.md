# HireMind DL — Efficient CV–Job Fit Classification through Distillation and Quantization

**CS4085 Deep Learning · Effat University · Fall 2026**
**Team:** Rawan Khojah · Hams Aljohani
**Instructor:** Dr Naila Marir

> **Status:** Assignment 1 submitted. Assignment 2: M0 baseline trained and evaluated on **validation only**; report draft in `report/milestones_1_2/`. M1 and M2 not started.

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
| **M2** — efficiency | CPU copy of M1 with dynamic INT8 Linear layers and 8-bit weight-only embeddings (post-training, no extra training) |

**Hypotheses (exploratory targets):**
- **H1:** M1 improves test macro-F1 over M0 by at least 1.0 point.
- **H2:** M2 is at least 60% smaller than M1, loses at most 1.0 macro-F1 point, and keeps at least half of M1's gain over M0.

**Measurements:** macro-F1, accuracy, per-class scores, confusion matrix, parameter count, serialized size,
MACs/FLOPs, CPU median/p95 latency and peak memory, measured on stated hardware under the same settings.

## Repository structure

| Folder / file | Contents |
|---|---|
| `proposal/` | Final proposal PDF, `proposal.tex` (ACL format), `ai_review.tex` (AI review appendix), Prompt C check and the instructor's reviewer prompt |
| `literature/` | `references.txt` and the research pack with our reading notes on the five papers |
| `notebooks/` | `HireMind_M0_Baseline.ipynb` — Assignment 2: dataset audit, grouped split, overfit-one-batch test, M0 training, loss curves, parameters/MACs, CPU latency |
| `scripts/` | `data_prep.py` (audit, cleaning, grouped split), `bench_cpu.py` (CPU latency/memory in a fresh process), `draw_architecture.py` (M0 diagram), `build_notebook.py` (regenerates the notebook), `audit_dataset.py` (Assignment 1 audit) |
| `splits/` | `splits.csv` + `manifest.json` — exact train/validation/test row IDs (hashed, no text); see `splits/README.md` |
| `figures/` | Architecture diagram and figures produced by the notebook |
| `report/milestones_1_2/` | Assignment 2 report: `HireMind_A2_Milestones_1_2.pdf` and its source (`main.tex`, ACL format) |
| `data/` | `README.md` with the dataset link, inspected version and known limitations. **Raw data is not stored here.** |
| `results/` | `dataset_audit.json` (Assignment 1) and, after the notebook runs, `dataset_stats.json`, `m0_baseline.json`, `m0_cpu_bench.json` — **validation results only** |

## Reproduce

**Assignment 2 (M0 baseline).** Open `notebooks/HireMind_M0_Baseline.ipynb` in Google Colab
(`File → Open notebook → GitHub → Hams770/HireMind-DL`), choose `Runtime → Change runtime type → T4 GPU`,
then `Runtime → Run all`. The notebook:

1. downloads the dataset at the pinned revision `08978e21714984bb417547d2c0f9b477f5298163` and checks the SHA-256 hashes;
2. audits classes, duplicates, near-duplicates and train/test overlap, and plots the statistics;
3. builds the job-grouped train/validation split (seed 42) and saves the row IDs;
4. trains simple references and M0 (learning-rate search on validation, then seeds 43 and 44), with an overfit-one-batch test;
5. measures parameters, MACs, saved size, CPU latency and memory;
6. downloads `HireMind_A2_outputs.zip` with figures, JSON results, split IDs and `tex/report_numbers.tex` for the report.

The test set is **not** evaluated in Assignment 2. Raw CV text and checkpoints stay in the notebook's `private/` folder and are never committed.

Local alternative: `pip install -r requirements.txt`, then run the notebook with Jupyter.

**Assignment 1 audit.** `python scripts/audit_dataset.py data/raw/train.csv data/raw/test.csv --output results/dataset_audit.json`

## Progress

| Item | Status |
|---|---|
| Five-paper review and research question | ✅ Done |
| Dataset structure audit | ✅ Done  |
| item 1 proposal (M0/M1/M2, H1/H2) | ✅  |
| Prompt C hypothesis check | ✅ Done  |
| Presentation format | ✅ Technical Report |
| New-chat AI review (proposal appendix) | ✅ Done — 83/100 |
| Dataset provenance, licence and label origin | ⚠️ treated as a limitation |
| item 2 : 9-paper comparison, research gap, dataset analysis, split code | ✅  |
| item 2: M0 architecture diagram, parameters and MACs | ✅ Ready |
| item 2: M0 training, overfit test, loss curves, cost | ✅ Done |
| 90-pair label check | ⏳  |
| Teacher, M1, M2 training and evaluation | ⏳ Not started |
| Final report, presentation, poster | ⏳ Not started |

### M0 baseline (validation split, BERT-Tiny, 256 tokens)

| Model | Macro-F1 (%) | Accuracy (%) |
|---|---|---|
| Majority class | 22.2 | 49.8 |
| TF-IDF + logistic regression | 36.7 | 52.8 |
| M0, seed 42 (lr 3e-4) | 48.2 | 56.2 |
| M0, seeds 42–44 (mean ± s.d.) | 44.6 ± 5.0 | 53.5 ± 2.4 |

Overfit-one-batch test passed (loss 1.150 → 0.0009, 100% accuracy). Cost: 4,386,307 parameters, 134.2 M MACs per pair,
16.74 MiB saved weights, 5.4 ms median CPU latency (1 thread, batch 1; 24.2 ms including tokenisation),
peak memory 864 MiB (771 MiB of which is PyTorch/Transformers after import).
All pairs exceed 256 tokens (median 1,563), so the model sees only the start of each text.
Figures: `figures/`; raw numbers: `results/m0_baseline.json`.

Only results produced by the notebook are reported. Validation is used for all choices; the test set is reserved for the final evaluation.

## Links

Overleaf item 1 (proposal): https://www.overleaf.com/project/6acffdc25099ba40f22e8c46/share#865f4fc4739de1c88a160527c2ef36eae1981d1718b814aa
Overleaf item 2 (Milestones 1–2) :https://www.overleaf.com/project/6adb4cc6c2e32fc022083954/share#4ba0bb552aa2a65c09a740720fe7b8bfbae12b644691937f
