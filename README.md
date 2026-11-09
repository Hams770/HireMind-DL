# HireMind DL — Efficient CV–Job Fit Classification through Distillation and Quantization

**CS4085 Deep Learning · Effat University · Fall 2026**
**Team:** Rawan Khojah (S22107688) · Hams Aljohani (S20106833)
**Instructor:** Dr. Naila Marir

HireMind is our Senior project, an AI-assisted recruitment platform. This Deep Learning project studies one
component it could use: a small model that reads a **CV and a job description** and predicts
**No Fit / Potential Fit / Good Fit**.

**Research question:** Can teacher-guided training (knowledge distillation) improve a compact CV–job fit
classifier, and can post-training quantization reduce its cost while keeping useful performance?

**What the code does:** downloads the pinned public dataset, rebuilds the job-disjoint split, trains the
BERT-Tiny baseline (M0) and the distilled student (M1), quantizes M1 into M2, scores all three on
validation and test, and measures their size and CPU latency. **Answer:** distillation did not help
(test macro-F1 −1.82 points); quantization made the model 73% smaller with no quality loss, but 13% slower
on our CPU.

**Contents:** [Deliverables](#deliverables-by-assignment) · [Models](#models) · [Final results](#final-results-held-out-test-1759-pairs-mean--sample-sd-over-seeds-42-43-44) · [Dataset](#dataset) · [Installation](#installation) · [How to reproduce](#how-to-reproduce) · [Hardware](#hardware-and-compute) · [Where the results are](#where-the-final-results-are) · [Structure](#repository-structure)

## Deliverables by assignment

Each assignment has its own folder with the submitted document, its source and its evidence.
Code shared by all milestones (the reproducible pipeline) is at the top level.

| Assignment | Folder | Main document |
|---|---|---|
| **Item 1 — Proposal** | [`assignment_1/`](assignment_1/) | [`HireMind_DL_Proposal_Final.pdf`](assignment_1/HireMind_DL_Proposal_Final.pdf) |
| **Item 2 — Milestones 1–2** (related work, data, M0 baseline) | [`assignment_2/`](assignment_2/) | [`HireMind_A2_Milestones_1_2.pdf`](assignment_2/HireMind_A2_Milestones_1_2.pdf) |
| **Item 3 — Preliminary results** (validation only) | [`assignment_3/`](assignment_3/) | [`HireMind_Assignment_3.pdf`](assignment_3/HireMind_Assignment_3.pdf) |
| **Item 4 — Final research project** (held-out test) | [`assignment_4/`](assignment_4/) | [`HireMind_Final_Report.pdf`](assignment_4/HireMind_Final_Report.pdf) · [slides](assignment_4/HireMind_Presentation.pdf) · [poster](assignment_4/HireMind_A1_Poster.pdf) |

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

Source files: [`assignment_4/results/`](assignment_4/results/). Validation-stage results are in [`assignment_2/results/`](assignment_2/results/) (Item 2) and [`assignment_3/item3_evidence/`](assignment_3/item3_evidence/) (Item 3).

## Dataset

- **Name:** [`cnamuangtoun/resume-job-description-fit`](https://huggingface.co/datasets/cnamuangtoun/resume-job-description-fit) on the Hugging Face Hub (public, no account needed).
- **Pinned revision:** `08978e21714984bb417547d2c0f9b477f5298163`. The scripts download `train.csv` and `test.csv` at this revision automatically and stop if their SHA-256 differs from [`configs/experiment.json`](configs/experiment.json).
- **Splits:** 4,993 training, 1,162 validation and 1,759 test pairs after cleaning. Jobs never appear in two splits; 474 CVs appear in both training and test. The split definition (row IDs and labels only, no text) is [`splits/splits.csv`](splits/splits.csv), SHA-256 `2ff3926c…539f9b`; [`splits/README.md`](splits/README.md) and [`data/README.md`](data/README.md) explain it.
- **Not in this repository:** raw CV/job text, tokenized caches, model weights and teacher logits. They are recreated locally in `data/cache/` and `runs/`, which are git-ignored.

## Installation

Python 3.13 was used (3.10 or newer should work). From the repository root:

```sh
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu   # skip on Colab/GPU machines
pip install -r requirements.txt
python scripts/smoke_test.py     # about 1 minute; must print SMOKE TEST PASSED
```

[`requirements.txt`](requirements.txt) pins the exact versions that trained and evaluated the models (torch 2.11.0, transformers 4.51.3, huggingface-hub 0.30.2, numpy 2.1.3, pandas 2.2.3, scikit-learn 1.6.1, matplotlib 3.10.1). On Google Colab, run `pip install -r requirements.txt` in a cell and keep Colab's own GPU build of torch.

## How to reproduce

All settings come from [`configs/experiment.json`](configs/experiment.json): the frozen protocol (seeds 42, 43, 44; batch 16; 3 epochs = 939 updates; AdamW, weight decay 0.01; 10% warm-up; clipping 1.0; validation every 100 updates; 256 tokens), the search grids and the selected values (M0 lr 1e-4; teacher lr 5e-5; M1 lr 5e-5, T = 2, alpha 0.5).

**Option A, check every published table and figure (about 2 minutes, no GPU, no dataset or weights needed):**
```sh
pip install numpy==2.1.3 pandas==2.2.3 matplotlib==3.10.1 tabulate==0.9.0   # or the full requirements
python scripts/reproduce_paper_tables.py
```
It rebuilds Tables 1–5 and the hypothesis checks from the saved result files, asserts that they equal the numbers printed in the papers, and redraws the three figures.

**Option B, retrain everything from the raw data (GPU recommended):**
```sh
python scripts/build_splits.py                # optional: rebuilds splits.csv from raw data and checks it is identical
python scripts/prepare_data.py                # downloads the dataset, checks it, tokenizes -> data/cache/
python scripts/train.py --model M0 --search   # learning-rate search (seed 42)   -> runs/M0_search.json
python scripts/train.py --model M0            # selected M0, seeds 42 43 44
python scripts/train.py --model teacher --search
python scripts/train.py --model teacher       # selected teacher + cached training logits
python scripts/train.py --model M1 --search   # lr x temperature search (seed 42)
python scripts/train.py --model M1            # selected M1, seeds 42 43 44
python scripts/train.py --model M1 --ablation # T = 1 ablation (Table 4)
python scripts/quantize_m2.py --all-variants  # M2 + one-factor quantization ablation (Figure 2)
python scripts/evaluate.py --split validation --benchmark   # Tables 2-3, Figure 1
python scripts/evaluate.py --split test --benchmark         # Table 5 and hypotheses (run once, after selection)
```
Outputs go to `outputs/` (`*_per_seed.csv`, `*_confusions.json`, `*_summary.csv`); the test command also prints the difference from the published per-seed results. The `--search` steps can be skipped because the selected values are already in the config. Exact numbers can differ slightly on other GPUs or library builds; CPU latency depends on the processor.

The original Colab notebooks are kept as the record of the actual runs: [`assignment_2/notebooks/HireMind_M0_Baseline.ipynb`](assignment_2/notebooks/HireMind_M0_Baseline.ipynb) (Item 2), [`assignment_3/notebooks/HireMind_Assignments_2_3_4_SetupFixed.ipynb`](assignment_3/notebooks/HireMind_Assignments_2_3_4_SetupFixed.ipynb) (teacher, M1, M2, validation) and [`assignment_4/code/HireMind_A4_Final_Evaluation.ipynb`](assignment_4/code/HireMind_A4_Final_Evaluation.ipynb) (held-out test). The scripts above use the same functions and settings without the Google Drive and crash-recovery code.

### Where each table and figure comes from (Item 4 paper)

| Paper item | Saved result file | Rebuild from saved file | Regenerate from scratch |
|---|---|---|---|
| Table 1, frozen partitions | `splits/manifest.json` | `reproduce_paper_tables.py` | `build_splits.py` |
| Table 2, validation comparison | `assignment_3/item3_evidence/reports/validation/per_seed.csv` | `reproduce_paper_tables.py` | `evaluate.py --split validation --benchmark` |
| Table 3, individual CPU runs | same file (latency, IQR, peak RSS) | `reproduce_paper_tables.py` | `evaluate.py --split validation --benchmark` |
| Figure 1, accuracy vs latency | same file | `assignment_3/reproduce_item3_figures.py` | `evaluate.py --split validation --benchmark` |
| Figure 2, quantization ablation | same file (`Linear only`, `Embeddings only`, `M2` rows) | `assignment_3/reproduce_item3_figures.py` | `quantize_m2.py --all-variants` |
| Table 4, temperature ablation | `.../validation/temperature_ablation.csv` | `reproduce_paper_tables.py` | `train.py --model M1 --ablation` |
| Table 5, held-out test | `assignment_4/results/per_seed.csv`, `summary_mean_sample_SD.csv` | `reproduce_paper_tables.py`, `assignment_4/code/verify_tables.py` | `evaluate.py --split test` |
| Figure 3, test confusion (M2 seed 44) | `assignment_4/results/confusions.json` | `assignment_4/code/plot_test.py` | `evaluate.py --split test` |
| H1/H2 checks | `assignment_4/results/hypotheses.json`, `paired_changes.csv` | `reproduce_paper_tables.py` | `evaluate.py --split test --benchmark` |

Item 2 figures and results (`assignment_2/paper/figures/`, `assignment_2/results/`) come from the M0 notebook.

### Tested

[`.github/workflows/reproduce.yml`](.github/workflows/reproduce.yml) runs these instructions on a clean GitHub machine on every push: Option A, then the pinned install, the smoke test, `build_splits.py` (rebuilt split must be byte-identical), `prepare_data.py`, and a full M0 seed-42 training run on CPU compared with the published validation score. Results are in the repository's **Actions** tab.

Last tested run (8 November 2026, Ubuntu, Python 3.13, CPU only): all steps passed; the rebuilt split was byte-identical to `splits/splits.csv`. The M0 seed-42 run matched the original GPU run exactly at the first evaluation (validation macro-F1 22.171 at update 100), then drifted because CPU and GPU arithmetic differ; its best validation macro-F1 was 43.75 against 46.54 on the T4. That gap is smaller than the seed-to-seed spread (SD 5.72). Exact numbers need the same GPU and library builds.

## Hardware and compute

| Stage | Hardware | Cost |
|---|---|---|
| Training (teacher, M0, M1) | Google Colab, NVIDIA Tesla T4 (16 GB), CUDA 13.0, Python 3.13.16 | one student run ≈ 4 min, peak ≈ 0.3 GiB GPU; selected teacher run 1,719 s, peak 2.79 GiB; whole study (all searches, seeds and ablation) estimated at 2–3 T4 GPU-hours |
| M2 quantization | CPU (no training) | seconds per seed |
| Latency benchmark | Intel Xeon @ 2.00 GHz, 1 PyTorch thread, batch 1, 256 tokens, 20 fixed validation inputs, 30 warm-up + 200 timed calls | under a minute per model and seed |
| Checks (Option A) | any laptop | about 2 minutes |

## Where the final results are

- **Held-out test (final):** [`assignment_4/results/`](assignment_4/results/): `per_seed.csv`, `summary_mean_sample_SD.csv`, `confusions.json`, `paired_changes.csv`, `hypotheses.json`, `final_freeze.json` (frozen choices made before the test was opened).
- **Validation (Item 3):** [`assignment_3/item3_evidence/`](assignment_3/item3_evidence/): `reports/validation/`, `quantization/`, `temperature_ablation.json`, `protocol.json`.
- **Item 2 M0 baseline:** [`assignment_2/results/`](assignment_2/results/).
- **Your re-runs:** `outputs/` (created by the scripts, not committed).

## Repository structure

```
assignment_1/             Item 1: proposal (PDF + LaTeX), AI review, literature pack
assignment_2/             Item 2: milestones report (PDF + LaTeX + figures), M0 notebook, results, audit/benchmark scripts
assignment_3/             Item 3: preliminary-results paper, notebook, validation evidence (item3_evidence/)
assignment_4/             Item 4: final paper, slides, poster, test evaluation notebook and code, test results
src/hiremind_dl/          shared code: splits.py (cleaning + split), data.py (download, checks, tokenization),
                          models.py (student/teacher, KD loss), training.py, evaluation.py,
                          quantization.py (M2), benchmark.py (size, MACs, CPU latency)
scripts/                  command-line pipeline: build_splits, prepare_data, train, quantize_m2,
                          evaluate, reproduce_paper_tables, smoke_test
configs/experiment.json   frozen protocol, search grids, selected hyperparameters
splits/                   frozen train/validation/test row IDs and manifest (used by Items 2-4)
data/README.md            dataset link and limitations (no raw data)
senior2_512/              Senior Project II extension: retrain M0 with 512-token inputs (Colab notebook)
requirements.txt          pinned package versions
```

## AI assistance

ChatGPT and Claude were used for code debugging and language editing. All experiments, results and conclusions were produced and verified by the authors.

## Links

- Overleaf, Item 1 proposal: https://www.overleaf.com/project/6acffdc25099ba40f22e8c46/share#865f4fc4739de1c88a160527c2ef36eae1981d1718b814aa
- Overleaf, Item 2 Milestones 1–2: https://www.overleaf.com/project/6adb4cc6c2e32fc022083954/share#4ba0bb552aa2a65c09a740720fe7b8bfbae12b644691937f
- **Project portfolio (this repository, linked in the final paper, slides and poster QR):** https://github.com/Hams770/HireMind-DL
- Team fork with the same content (linked in the Item 3 paper): https://github.com/RAWANKHOJAH/HireMind-DL
- HireMind platform (Senior Project II), where this classifier is used: https://github.com/Hams770/Senior-2
