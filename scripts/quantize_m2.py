"""Build M2 from each M1 seed (post-training quantization, no retraining) and score it on validation.

    python scripts/quantize_m2.py                  # M2 = Linear INT8 + 8-bit embeddings
    python scripts/quantize_m2.py --all-variants   # also the Linear-only / embeddings-only ablation
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch  # noqa: E402

from hiremind_dl import load_config  # noqa: E402
from hiremind_dl.benchmark import state_size_bytes  # noqa: E402
from hiremind_dl.data import as_dataset  # noqa: E402
from hiremind_dl.evaluation import evaluate  # noqa: E402
from hiremind_dl.models import build_model  # noqa: E402
from hiremind_dl.quantization import quantize  # noqa: E402
from hiremind_dl.runs import CACHE, OUT, RUNS, dump, load_weights  # noqa: E402
from hiremind_dl.training import tag_for  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-variants", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    torch.set_num_threads(cfg["benchmark"]["threads"])
    va = as_dataset(CACHE / "validation_tokens.pt")
    m1 = cfg["selected"]["M1"]
    variants = ["linear", "embeddings", "both"] if a.all_variants else ["both"]
    rows = []
    for s in cfg["seeds"]:
        src = tag_for("M1", m1["lr"], s, m1["temperature"])
        fp = build_model("M1", cfg, s)
        fp.load_state_dict(load_weights(src), strict=True)
        fp_met, _, _ = evaluate(fp, va, "cpu")
        for v in variants:
            q = quantize(fp, v)
            tag = f"M2_s{s}" if v == "both" else f"Q_{v}_s{s}"
            (RUNS / tag).mkdir(parents=True, exist_ok=True)
            torch.save(q.state_dict(), RUNS / tag / "weights.pt")
            met, _, _ = evaluate(q, va, "cpu")
            rows.append(dict(tag=tag, source=src, variant=v, seed=s,
                             validation_macro_f1=met["macro_f1"], m1_validation_macro_f1=fp_met["macro_f1"],
                             change=met["macro_f1"] - fp_met["macro_f1"],
                             size_MB=state_size_bytes(q) / 1e6, m1_size_MB=state_size_bytes(fp) / 1e6))
            print(rows[-1])
    dump(OUT / "quantization_validation.json", rows)


if __name__ == "__main__":
    main()
