"""HireMind DL: CV-job fit classification with BERT-Tiny (M0), distillation (M1) and INT8 quantization (M2).

The modules follow the Colab notebooks that produced the paper results
(assignment_3/notebooks/HireMind_Assignments_2_3_4_SetupFixed.ipynb and
assignment_4/code/HireMind_A4_Final_Evaluation.ipynb), without the Drive/recovery code.
"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LABELS = ["No Fit", "Potential Fit", "Good Fit"]


# Senior Project II extension: HIREMIND_MAX_LEN=512 retrains with longer inputs. Everything else stays frozen,
# and its files go to separate folders so the 256-token results of the paper are never overwritten.
MAX_LEN = int(os.getenv("HIREMIND_MAX_LEN", "256"))


def load_config(path=None):
    path = Path(path) if path else ROOT / "configs" / "experiment.json"
    cfg = json.loads(path.read_text())
    cfg["tokenization"]["max_len"] = MAX_LEN
    return cfg
