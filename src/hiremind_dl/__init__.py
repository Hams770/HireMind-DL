"""HireMind DL: CV-job fit classification with BERT-Tiny (M0), distillation (M1) and INT8 quantization (M2).

The modules follow the Colab notebooks that produced the paper results
(assignment_3/notebooks/HireMind_Assignments_2_3_4_SetupFixed.ipynb and
assignment_4/code/HireMind_A4_Final_Evaluation.ipynb), without the Drive/recovery code.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LABELS = ["No Fit", "Potential Fit", "Good Fit"]


def load_config(path=None):
    path = Path(path) if path else ROOT / "configs" / "experiment.json"
    return json.loads(path.read_text())
