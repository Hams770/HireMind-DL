# HireMind | CS4085 final research project
Rawan Khojah and Hams Aljohani · Effat University

Report: [HireMind_Final_Report.pdf](HireMind_Final_Report.pdf) · Slides: [PPTX](HireMind_Presentation.pptx) / [PDF](HireMind_Presentation.pdf) · Poster: [HireMind_A1_Poster.pdf](HireMind_A1_Poster.pdf) · Item 3 materials: [../assignment_3](../assignment_3/) · Project overview: [../README.md](../README.md)

The paper compares supervised BERT-Tiny (M0), a distilled student (M1) and its quantized state (M2) for three-way CV–job compatibility. This folder includes the report, slides, A1 poster, evaluation code, editable paper source and saved evidence. The nine held-out evaluations are complete.

## Reproduce the published measurements from saved evidence
From this directory, in a Python environment with pandas, NumPy and matplotlib:
```sh
python -m pip install pandas numpy matplotlib
python code/verify_tables.py
python code/plot_test.py
python reproduce_item3_figures.py
```
These commands require no checkpoints, Drive account or GPU. The verifier checks all nine accuracy, macro-F1 and Good Fit recall values against the 3×3 confusion matrices. It checks the mean/sample-SD table and prints the test comparison. The plots go to paper/figures/test_errors.png and reproduced/figures/. In the matrices, rows are labels and columns predictions; class order is No Fit, Potential Fit, Good Fit. Scores are percentage points. Rows from different seeds describe repeated predictions on the same examples.

| Paper output | Data and reconstruction |
|---|---|
| Partition sizes | Count rows by split in item3_evidence/reproducibility/splits.csv. |
| Validation comparison | item3_evidence/reports/validation/per_seed.csv; select M0, M1, M2; group by model and aggregate mean/sample SD. The validation figure script does this. |
| Accuracy–latency and component ablation plots | reproduce_item3_figures.py reads the same CSV and plots individual seeds with latency quartiles or paired quantization variants. |
| Individual CPU runs | Read latency_ms, iqr_ms and peak_process_MiB by model and seed in the comparison CSV. The benchmark_sessions directory contains individual timings. |
| Temperature table | item3_evidence/reports/validation/temperature_ablation.csv; compare each seed's best validation checkpoint at T=1 vs T=2. |
| Held-out comparison and error figure | code/verify_tables.py and code/plot_test.py. results/per_seed.csv contains all nine records; results/paired_changes.csv and hypotheses.json give direction and thresholds. |
| Recovery and source identity | item3_evidence/recovery_incident plus results/final_freeze.json, with the nine expected file digests in the evaluation code. |

## Training and evaluation implementation
Dataset: [resume-job-description-fit](https://huggingface.co/datasets/cnamuangtoun/resume-job-description-fit). The exact dataset and student revisions, hyperparameters, selection digests and test checksum appear in results/final_freeze.json. Use the pinned revision and frozen row IDs. Private CV text, private review sheets, tokens and weights are excluded.

For the reference implementation and original training entry points, use the corrected SetupFixed notebook in [../assignment_3/notebooks](../assignment_3/notebooks/). For a new, separately authorized reproduction use a fresh work directory, install item3_evidence/reproducibility/requirements.txt, compare tested_versions.json, then use the notebook's ordered setup, data preparation, M0, Teacher/cache, M1, quantization, benchmark and validation-reporting stages. Search uses seed 42; selected student settings repeat at 43 and 44. Training used a Tesla T4. CPU timing used an Intel Xeon at 2.00 GHz with one thread, batch one, sequence length 256, 30 warmups and 200 measured calls. Reproducing identical timings requires comparable hardware and software.

For provenance, code/HireMind_A4_Final_Evaluation.ipynb contains the setup and definitions plus the successful resume implementation in the last cell, disabled by default. code/HireMind_A4_Resume.py preserves the historical Colab cell; it depends on the notebook namespace and the private existing freeze and weights. Do not execute it as a standalone script or rerun this cohort for tuning. The held-out table uses saved test predictions; CPU timings were copied from the matched benchmark, not remeasured using test pairs.

## Oral defense practice
The speaker notes give a full script for each slide and assign both team members roles totaling about 13 minutes 20 seconds before questions. Time one full rehearsal. Both should be able to answer: Why macro-F1 rather than accuracy? How can a smaller model run slower? Why does M1 seed 44 matter for interpreting M2 errors? Why did the missing H1 gain prevent confirming retention? What does cross-split CV overlap allow you to claim?

## AI assistance log
ChatGPT and Claude were used for code debugging and language editing. All experiments, results and conclusions were produced and verified by the authors.
