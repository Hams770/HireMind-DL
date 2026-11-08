"""Backward-compatible import path. The split code now lives in src/hiremind_dl/splits.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hiremind_dl.splits import *  # noqa: F401,F403,E402
from hiremind_dl.splits import LABELS, LABEL2ID, NEAR_DUP_THRESHOLD  # noqa: E402,F401
