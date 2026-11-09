"""Where tokens, weights and reproduced results are stored (all git-ignored except results)."""
import json
from pathlib import Path

import torch

from . import MAX_LEN, ROOT

_SUFFIX = "" if MAX_LEN == 256 else f"_L{MAX_LEN}"
CACHE = ROOT / "data" / f"cache{_SUFFIX}"
RUNS = ROOT / f"runs{_SUFFIX}"
OUT = ROOT / "outputs" / (f"L{MAX_LEN}" if _SUFFIX else "")


def device(name="auto"):
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def save_run(info, state, runs=RUNS):
    d = Path(runs) / info["tag"]
    d.mkdir(parents=True, exist_ok=True)
    torch.save(state, d / "weights.pt")
    (d / "info.json").write_text(json.dumps(info, indent=2))
    return d


def read_info(tag, runs=RUNS):
    return json.loads((Path(runs) / tag / "info.json").read_text())


def load_weights(tag, runs=RUNS):
    return torch.load(Path(runs) / tag / "weights.pt", map_location="cpu", weights_only=True)


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))
