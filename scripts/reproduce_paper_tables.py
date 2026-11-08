"""Rebuild every table and figure of the Item 3 and Item 4 papers from the saved result files.

No GPU, model weights or dataset download is needed:

    python scripts/reproduce_paper_tables.py

Tables are printed and written to outputs/paper_tables.md; figures go to outputs/figures/
and assignment_3/reproduced/figures/. Each check compares against the numbers in the papers.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
VAL = ROOT / "assignment_3/item3_evidence/reports/validation"
TEST = ROOT / "assignment_4/results"
lines = []


def show(title, df, source):
    text = f"### {title}\nSource: `{source}`\n\n{df.to_markdown()}\n"
    print(text)
    lines.append(text)


def pm(g, col, nd=2):
    return f"{g[col].mean():.{nd}f} ± {g[col].std():.{nd}f}"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)

    # Table 1: frozen partitions
    m = json.loads((ROOT / "splits/manifest.json").read_text())["counts"]
    t1 = pd.DataFrame({k: dict(pairs=v["rows"], CVs=v["cvs"], jobs=v["jobs"], **v["class_counts"])
                       for k, v in m.items()}).T
    assert t1.pairs.tolist() == [4993, 1162, 1759]
    show("Table 1 - Frozen partitions (Item 3 and Item 4)", t1, "splits/manifest.json")

    # Table 2: validation comparison
    v = pd.read_csv(VAL / "per_seed.csv")
    main_v = v[v.model.isin(["M0", "M1", "M2"])]
    t2 = pd.DataFrame({k: {"Accuracy (%)": pm(g, "accuracy"), "Macro-F1 (%)": pm(g, "macro_f1"),
                           "Params (M)": f"{g.params_M.mean():.3f}", "State (MB)": f"{g.size_MB.mean():.3f}",
                           "MACs (G)": f"{g.macs_G.mean():.3f}", "CPU median (ms)": pm(g, "latency_ms", 3)}
                       for k, g in main_v.groupby("model")}).T
    assert t2.loc["M0", "Macro-F1 (%)"] == "40.01 ± 5.72" and t2.loc["M2", "State (MB)"] == "4.675"
    show("Table 2 - Validation comparison (main table)", t2, VAL.relative_to(ROOT) / "per_seed.csv")

    # Table 3: individual CPU runs
    t3 = main_v.set_index(["model", "seed"])[["latency_ms", "iqr_ms", "peak_process_MiB"]].round({"latency_ms": 3, "iqr_ms": 3, "peak_process_MiB": 1})
    assert t3.loc[("M2", 43), "latency_ms"] == 6.364
    show("Table 3 - Individual CPU runs", t3, VAL.relative_to(ROOT) / "per_seed.csv")

    # Table 4: temperature ablation
    t = pd.read_csv(VAL / "temperature_ablation.csv").set_index("seed")
    m1 = main_v[main_v.model == "M1"].set_index("seed")
    t4 = pd.DataFrame({"T=1 macro-F1": t.macro_f1, "T=2 macro-F1": m1.macro_f1}).round(2)
    t4.loc["Mean"] = [round(t.macro_f1.mean(), 2), round(m1.macro_f1.mean(), 2)]
    assert t4.loc["Mean"].tolist() == [37.62, 37.28]
    show("Table 4 - Temperature ablation (validation)", t4, VAL.relative_to(ROOT) / "temperature_ablation.csv")

    # Table 5: held-out test
    s = pd.read_csv(TEST / "per_seed.csv")
    t5 = pd.DataFrame({k: {"Accuracy": pm(g, "accuracy"), "Macro-F1": pm(g, "macro_f1"),
                           "Good Fit recall": f"{g.good_fit_recall.mean():.2f}"}
                       for k, g in s.groupby("model")}).T
    assert t5.loc["M0", "Macro-F1"] == "33.70 ± 3.92" and t5.loc["M2", "Good Fit recall"] == "12.81"
    show("Table 5 - Held-out test results (Item 4)", t5, TEST.relative_to(ROOT) / "per_seed.csv")

    # Hypotheses (Item 4)
    w = s.pivot(index="seed", columns="model")
    gain = (w.macro_f1.M1 - w.macro_f1.M0).mean()
    loss = (w.macro_f1.M2 - w.macro_f1.M1).mean()
    ratio = w.size_MB.M2 / w.size_MB.M1
    slower = (w.latency_ms.M2.mean() / w.latency_ms.M1.mean() - 1) * 100
    smaller = (1 - w.size_MB.M2.mean() / w.size_MB.M1.mean()) * 100
    h = json.loads((TEST / "hypotheses.json").read_text())
    np.testing.assert_allclose([gain, loss], [h["gain_points"], -h["quality_loss_points"]], atol=1e-9)
    hyp = pd.DataFrame({"value": {
        "H1: M1 - M0 macro-F1 (points, need >= +1)": round(gain, 2),
        "H2: M2/M1 size ratio per seed (need <= 0.40)": ", ".join(f"{r:.3f}" for r in ratio),
        "H2: M2 - M1 macro-F1 (points, need >= -1)": round(loss, 2),
        "M2 smaller than M1 (%)": round(smaller, 2),
        "M2 slower than M1 (%)": round(slower, 2)}})
    show("Hypothesis checks (Item 4)", hyp, TEST.relative_to(ROOT) / "per_seed.csv")

    (OUT / "paper_tables.md").write_text("\n".join(lines))

    # Figures
    subprocess.run([sys.executable, str(ROOT / "assignment_3/reproduce_item3_figures.py")], check=True)
    subprocess.run([sys.executable, str(ROOT / "assignment_4/code/verify_tables.py")], check=True,
                   stdout=subprocess.DEVNULL)
    subprocess.run([sys.executable, str(ROOT / "assignment_4/code/plot_test.py"),
                    str(OUT / "figures/test_errors.png")], check=True)
    print("Figures 1-2 (accuracy_latency, quantization_ablation): assignment_3/reproduced/figures/")
    print("Figure 3 (test_errors):", OUT / "figures/test_errors.png")
    print("ALL PAPER TABLES MATCH THE SAVED RESULTS")


if __name__ == "__main__":
    main()
