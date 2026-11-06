"""Check public test tables from aggregate counts; no network or model required."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[1] / "results"
counts = json.loads((root / "confusions.json").read_text())
records = pd.read_csv(root / "per_seed.csv").set_index(["model", "seed"])
expected = {(model, seed) for model in ["M0", "M1", "M2"] for seed in [42, 43, 44]}
assert set(records.index) == expected and len(records) == len(counts) == 9
for (model, seed), row in records.iterrows():
    matrix = np.array(counts[f"{model}_s{seed}"])
    assert list(matrix.sum(axis=1)) == [857, 444, 458]
    diagonal = np.diag(matrix)
    accuracy = 100 * diagonal.sum() / matrix.sum()
    macro_f1 = 100 * (2 * diagonal / (matrix.sum(axis=0) + matrix.sum(axis=1))).mean()
    good_recall = 100 * diagonal[2] / matrix[2].sum()
    np.testing.assert_allclose([accuracy, macro_f1, good_recall],
                               row[["accuracy", "macro_f1", "good_fit_recall"]].astype(float))
metrics = records[["accuracy", "macro_f1", "good_fit_recall", "latency_ms", "size_MB"]]
table = metrics.groupby(level="model").agg(["mean", "std"])
saved = pd.read_csv(root / "summary_mean_sample_SD.csv", header=[0, 1], index_col=0)
np.testing.assert_allclose(table, saved.loc[table.index, table.columns])
print(table.round(2).to_string())
