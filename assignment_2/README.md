# Item 2 · Milestones 1–2: related work, data and M0 baseline

[Read the report (PDF)](HireMind_A2_Milestones_1_2.pdf) · [Project overview](../README.md)

This milestone audits the dataset, fixes the job-disjoint train/validation/test split and trains the
BERT-Tiny baseline (M0) on validation only.

| Folder / file | Content |
|---|---|
| `HireMind_A2_Milestones_1_2.pdf` | Submitted report |
| `paper/` | LaTeX source (`main.tex`), generated numbers (`tex/`) and all report figures (`figures/`) |
| `notebooks/HireMind_M0_Baseline.ipynb` | Colab notebook that ran the audit, split, M0 learning-rate search, three seeds and CPU benchmark |
| `results/` | Dataset audit and statistics, M0 validation results, CPU benchmark, tested package versions |
| `scripts/` | `audit_dataset.py` (data audit), `bench_cpu.py` (CPU latency), `draw_architecture.py` (Figure: M0 architecture), `build_notebook.py` (generates the notebook) |

The frozen split produced here is in [`../splits/`](../splits/) because Items 3 and 4 use it too.
The split code is [`../src/hiremind_dl/splits.py`](../src/hiremind_dl/splits.py).

**Reproduce:** open the notebook in Google Colab (GPU runtime) and run all cells, or use the standalone
scripts from the repository root: `python scripts/build_splits.py`, `python scripts/prepare_data.py --no-test`,
`python scripts/train.py --model M0 --search`, `python scripts/train.py --model M0`.
