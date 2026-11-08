"""Score M0, M1 and M2 (seeds 42, 43, 44) on validation or the held-out test split.

    python scripts/evaluate.py --split validation
    python scripts/evaluate.py --split test --benchmark      # Item 4 Table + CPU cost

Writes results/reproduced/<split>_per_seed.csv, <split>_confusions.json and
<split>_summary.csv (mean and sample SD over seeds). For the test split it also prints the
difference from the published numbers in assignment_4/results/per_seed.csv.
Test results must not be used to choose anything (see "rule" in configs/experiment.json).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from hiremind_dl import ROOT, load_config  # noqa: E402
from hiremind_dl.benchmark import analytic_macs, cpu_benchmark, state_size_bytes  # noqa: E402
from hiremind_dl.data import as_dataset  # noqa: E402
from hiremind_dl.evaluation import evaluate  # noqa: E402
from hiremind_dl.models import build_model  # noqa: E402
from hiremind_dl.quantization import quantize  # noqa: E402
from hiremind_dl.runs import CACHE, OUT, RUNS, device as pick_device, dump, load_weights  # noqa: E402
from hiremind_dl.training import tag_for  # noqa: E402


def load(name, seed, cfg):
    sel = cfg["selected"]
    if name == "M0":
        m = build_model("M0", cfg, seed)
        m.load_state_dict(load_weights(tag_for("M0", sel["M0"]["lr"], seed)), strict=True)
    else:
        m = build_model("M1", cfg, seed)
        m.load_state_dict(load_weights(tag_for("M1", sel["M1"]["lr"], seed, sel["M1"]["temperature"])), strict=True)
        if name == "M2":
            m = quantize(m, "both")
            m.load_state_dict(torch.load(RUNS / f"M2_s{seed}" / "weights.pt", weights_only=True))
    return m.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["validation", "test"], required=True)
    ap.add_argument("--benchmark", action="store_true", help="also measure CPU latency (1 thread, batch 1)")
    ap.add_argument("--device", default="auto", help="device for M0/M1 scoring; M2 always runs on CPU")
    a = ap.parse_args()
    cfg = load_config()
    ds = as_dataset(CACHE / f"{a.split}_tokens.pt")
    tok = AutoTokenizer.from_pretrained(cfg["student"]["name"], revision=cfg["student"]["revision"])
    bench = json.loads((CACHE / "benchmark_inputs.json").read_text())
    pairs = [(r["resume_text"], r["job_description_text"]) for r in bench]
    rows, confusions = [], {}
    for name in ["M0", "M1", "M2"]:
        for s in cfg["seeds"]:
            m = load(name, s, cfg)
            dev = "cpu" if name == "M2" else pick_device(a.device)
            met, _, _ = evaluate(m.to(dev), ds, dev)
            m = m.cpu()
            row = dict(model=name, seed=s, accuracy=met["accuracy"], macro_f1=met["macro_f1"],
                       good_fit_recall=met["good_fit_recall"], size_MB=state_size_bytes(m) / 1e6,
                       macs_G=analytic_macs(m, cfg["tokenization"]["max_len"]) / 1e9)
            if a.benchmark:
                b = cpu_benchmark(m, tok, pairs, cfg)
                row.update({k: b[k] for k in ["latency_ms", "q1_ms", "q3_ms", "iqr_ms", "inclusive_ms"]})
            rows.append(row)
            confusions[f"{name}_s{s}"] = met["confusion"]
            print({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()})
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"{a.split}_per_seed.csv", index=False)
    dump(OUT / f"{a.split}_confusions.json", confusions)
    summary = df.drop(columns="seed").groupby("model").agg(["mean", "std"])  # std = sample SD (ddof=1)
    summary.to_csv(OUT / f"{a.split}_summary.csv")
    print(summary.round(2).to_string())
    if a.split == "test":
        paper = pd.read_csv(ROOT / "assignment_4/results/per_seed.csv").set_index(["model", "seed"])
        ours = df.set_index(["model", "seed"])
        cols = ["accuracy", "macro_f1", "good_fit_recall"]
        print("\nDifference from the published test results (reproduced - paper, points):")
        print((ours[cols] - paper.loc[ours.index, cols]).round(2).to_string())


if __name__ == "__main__":
    main()
