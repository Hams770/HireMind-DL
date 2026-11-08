"""Rebuild the report's held-out figure from aggregate counts; no inference."""
from pathlib import Path
import json
import sys
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
matrix = json.loads((root / "results/confusions.json").read_text())["M2_s44"]
labels = ["No Fit", "Potential Fit", "Good Fit"]
fig, ax = plt.subplots(figsize=(4.4, 3.35), layout="constrained")
ax.imshow(matrix, cmap="Blues", vmin=0)
ax.set(xticks=range(3), yticks=range(3), xticklabels=labels, yticklabels=labels,
       xlabel="Predicted class", ylabel="True class")
for row, counts in enumerate(matrix):
    for col, count in enumerate(counts):
        ax.text(col, row, str(count), ha="center", va="center",
                color="white" if count > 390 else "#17324d", fontsize=13)
out = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "paper/figures/test_errors.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=220)
print("wrote", out)
