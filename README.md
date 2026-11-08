# HireMind DL — Efficient CV–Job Fit Classification through Distillation and Quantization

**CS4085 Deep Learning · Effat University · Fall 2026**
**Team:** Rawan Khojah (S22107688) · Hams Aljohani (S20106833)
**Instructor:** Dr. Naila Marir

HireMind is our Senior project, an AI-assisted recruitment platform. This Deep Learning project studies one
component it could use: a small model that reads a **CV and a job description** and predicts
**No Fit / Potential Fit / Good Fit**.

**Research question:** Can teacher-guided training (knowledge distillation) improve a compact CV–job fit
classifier, and can post-training quantization reduce its cost while keeping useful performance?

## Deliverables by assignment

| Assignment | Main document | Code and evidence |
|---|---|---|
| **Item 1 — Proposal** | [`proposal/HireMind_DL_Proposal_Final.pdf`](proposal/HireMind_DL_Proposal_Final.pdf) | [`proposal/`](proposal/) (LaTeX source, AI review, Prompt C check) · [`literature/`](literature/) |
| **Item 2 — Milestones 1–2** (related work, data, M0 baseline) | [`report/milestones_1_2/HireMind_A2_Milestones_1_2.pdf`](report/milestones_1_2/HireMind_A2_Milestones_1_2.pdf) | [`notebooks/`](notebooks/) · [`scripts/`](scripts/) · [`splits/`](splits/) · [`results/`](results/) · [`figures/`](figures/) |
| **Item 3 — Preliminary results** (validation only) | [`assignment_3/HireMind_Assignment_3.pdf`](assignment_3/HireMind_Assignment_3.pdf) | [`assignment_3/`](assignment_3/) — notebook, saved evidence, paper source · [README](assignment_3/README.md) |
| **Item 4 — Final research project** (held-out test) | [`assignment_4/HireMind_Final_Report.pdf`](assignment_4/HireMind_Final_Report.pdf) · [slides](assignment_4/HireMind_Presentation.pdf) · [poster](assignment_4/HireMind_A1_Poster.pdf) | [`assignment_4/`](assignment_4/) — evaluation notebook, verification scripts, results · [README](assignment_4/README.md) |

## Models

| Model | Definition |
|---|---|
| **Teacher** (support only) | `bert-base-uncased`, fine-tuned on the training split; its training logits are cached for distillation |
| **M0** — baseline | BERT-Tiny (`google/bert_uncased_L-2_H-128_A-2`, 4,386,307 parameters) trained with cross-entropy |
| **M1** — DL modification | Same student, trained with `0.5·CE + 0.5·T²·KL(teacher ‖ student)`, T = 2 |
| **M2** — efficiency | CPU copy of M1 with dynamic INT8 Linear layers and 8-bit weight-only embeddings, no extra training |

## Final results (held-out test, 1,759 pairs, mean ± sample SD over seeds 42, 43, 44)

| Model | Accuracy (%) | Macro-F1 (%) | Good Fit recall (%) | Saved state (MB) | CPU median (ms) |
|---|---|---|---|---|---|
| M0 | 45.33 ± 1.12 | **33.70 ± 3.92** | 16.16 | 17.559 | 5.502 |
| M1 | 47.19 ± 0.82 | 31.88 ± 5.07 | 12.52 | 17.559 | 5.563 |
| M2 | 47.60 ± 0.61 | 32.39 ± 5.23 | 12.81 | **4.675** | 6.311 |

- **H1 (distillation gains ≥ 1 macro-F1 point): not met** — M1 − M0 = −1.82 points.
- **H2 (M2 ≤ 40% of M1's size, ≤ 1 point loss):** both criteria met (size ratio 0.266, M2 − M1 = +0.51); gain retention is not confirmatory because H1 failed.
- M2 is 73.38% smaller but 13.46% slower on the measured CPU (one thread, batch 1).
- Good Fit recall is low; at seed 44, M2 finds 7 of 458 Good Fit pairs. The model must not be used to reject candidates automatically.

Source files: [`assignment_4/results/`](assignment_4/results/). Validation-stage results (Item 2 and Item 3) are in [`results/`](results/) and [`assignment_3/item3_evidence/`](assignment_3/item3_evidence/).

## Dataset and splits

[`cnamuangtoun/resume-job-description-fit`](https://huggingface.co/datasets/cnamuangtoun/resume-job-description-fit), pinned revision `08978e21714984bb417547d2c0f9b477f5298163`.
After cleaning: 4,993 training, 1,162 validation and 1,759 test pairs. Jobs never appear in two splits; 474 CVs appear in both training and test.
Row IDs only (no text) are in [`splits/splits.csv`](splits/splits.csv); see [`splits/README.md`](splits/README.md) and [`data/README.md`](data/README.md).
Raw CV text and model weights are not stored in this repository.

## How to reproduce

**Check the published tables and figures (no GPU, no weights):**
```sh
pip install pandas numpy matplotlib
cd assignment_4
python code/verify_tables.py        # recomputes all nine test results from the confusion matrices
python code/plot_test.py            # rebuilds the error-analysis figure
python reproduce_item3_figures.py   # rebuilds the validation figures
```

**Retrain the models (Google Colab, Tesla T4 GPU):**
1. **M0 (Item 2):** open [`notebooks/HireMind_M0_Baseline.ipynb`](notebooks/HireMind_M0_Baseline.ipynb) in Colab and run all cells. It downloads the pinned dataset, audits it, builds the split, trains M0 and measures CPU cost.
2. **Teacher, M1 and M2 (Item 3):** use [`assignment_3/notebooks/HireMind_Assignments_2_3_4_SetupFixed.ipynb`](assignment_3/notebooks/HireMind_Assignments_2_3_4_SetupFixed.ipynb) with the versions in [`assignment_3/item3_evidence/reproducibility/`](assignment_3/item3_evidence/reproducibility/).
3. **Held-out evaluation (Item 4):** [`assignment_4/code/HireMind_A4_Final_Evaluation.ipynb`](assignment_4/code/HireMind_A4_Final_Evaluation.ipynb). Configuration and frozen selections: [`assignment_4/results/final_freeze.json`](assignment_4/results/final_freeze.json).

**Hardware:** training on a Tesla T4; CPU timing on an Intel Xeon 2.00 GHz, one PyTorch thread, batch 1, sequence length 256, 30 warm-up and 200 timed calls.

## Repository structure

```
proposal/              Item 1 proposal (PDF + LaTeX)
literature/            reading notes and references
report/milestones_1_2/ Item 2 report (PDF + LaTeX + figures)
notebooks/             Item 2 M0 notebook
scripts/               data audit, cleaning and split, CPU benchmark, architecture figure
splits/                frozen train/validation/test row IDs
results/               Item 2 validation results (M0, dataset audit)
figures/               Item 2 figures
data/                  dataset link and limitations (no raw data)
assignment_3/          Item 3 report, notebook, validation evidence, paper source
assignment_4/          Item 4 report, evaluation code, test results, paper source
```

## AI assistance

ChatGPT and Claude were used for code debugging and language editing. All experiments, results and conclusions were produced and verified by the authors.

## Links

- Overleaf, Item 1 proposal: https://www.overleaf.com/project/6acffdc25099ba40f22e8c46/share#865f4fc4739de1c88a160527c2ef36eae1981d1718b814aa
- Overleaf, Item 2 Milestones 1–2: https://www.overleaf.com/project/6adb4cc6c2e32fc022083954/share#4ba0bb552aa2a65c09a740720fe7b8bfbae12b644691937f
- Team submission fork: https://github.com/RAWANKHOJAH/HireMind-DL
