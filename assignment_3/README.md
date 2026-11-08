# HireMind · Assignment 3
Rawan Khojah and Hams Aljohani · CS4085 Deep Learning

[Read the research report](HireMind_Assignment_3.pdf) · [Open the corrected notebook](notebooks/HireMind_Assignments_2_3_4_SetupFixed.ipynb) · [Inspect the measured validation results](item3_evidence/reports/validation/per_seed.csv)

The notebook contains the SetupFixed cell sources. Publication metadata and output history were stripped. The original execution copy is retained separately. The experiments use the frozen split, 256-token paired inputs and seeds 42–44. The held-out test is reserved for the final milestone.

The evidence folder includes the fixed split IDs, protocol, software versions, CPU bench records, component and temperature ablations, Teacher recovery comparison and summary tables. See the paper for benchmark conditions and units. Private CVs, checkpoints and individual human review sheets stay outside the repository.

To reproduce the two figures and a summary from the saved CSV, run python reproduce_item3_figures.py with pandas, numpy and matplotlib installed. This script reads the CSV without training or evaluating a model. The paper/ folder contains the editable LaTeX source and figures.
