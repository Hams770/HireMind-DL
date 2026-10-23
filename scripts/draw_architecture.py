"""Draws the M0 (BERT-Tiny pair classifier) architecture with per-block parameters and MACs.
Also marks what changes in M1 (distillation loss) and M2 (quantized modules).
Numbers are analytic for google/bert_uncased_L-2_H-128_A-2 at 256 tokens; the notebook
re-measures them with torch.utils.flop_counter.FlopCounterMode.
"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

out = sys.argv[1] if len(sys.argv) > 1 else "m0_architecture"

INK = "#1f2937"; MUTED = "#6b7280"
EMB = "#fde68a"; ENC = "#bfdbfe"; HEAD = "#bbf7d0"; IO = "#f3f4f6"
Q_EDGE = "#7c3aed"; KD = "#fecaca"

fig, ax = plt.subplots(figsize=(11, 13))
ax.set_xlim(-2, 100); ax.set_ylim(0, 132); ax.axis("off")
LINE = 2.9


def box(x, y_top, w, title, lines=(), fc=IO, ec=INK, lw=1.2, ls="-", title_size=11):
    h = 6.0 + LINE * len(lines)
    y = y_top - h
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                fc=fc, ec=ec, lw=lw, ls=ls))
    ax.text(x + w / 2, y_top - 1.6, title, ha="center", va="top", fontsize=title_size, weight="bold", color=INK)
    for i, t in enumerate(lines):
        ax.text(x + w / 2, y_top - 5.2 - i * LINE, t, ha="center", va="top", fontsize=8.6, color=INK)
    return y, h


def arrow(x1, y1, x2, y2, color=INK, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13,
                                 color=color, lw=1.3, ls=ls))


def cost(y_mid, params, macs):
    ax.text(72, y_mid + 1.4, params, ha="left", va="center", fontsize=8.8, color=INK, weight="bold")
    ax.text(72, y_mid - 1.6, macs, ha="left", va="center", fontsize=8.2, color=MUTED)


ax.text(49, 130.5, "M0: BERT-Tiny CV–job fit classifier (L=2, H=128, A=2, 256 tokens)",
        ha="center", fontsize=13, weight="bold", color=INK)
ax.text(72, 124.5, "Parameters\nMACs per pair (256 tokens)", ha="left", va="center", fontsize=8.6, color=MUTED)

X, W, GAP = 16, 52, 3.4
cx = X + W / 2
blocks = [
    dict(title="Input", lines=["CV text  +  job description text"], fc=IO, q=False, cost=None),
    dict(title="WordPiece tokenizer (uncased, 30,522-token vocabulary)",
         lines=["[CLS] CV tokens [SEP] job tokens [SEP] → 256 ids",
                "longest-first truncation · segment ids 0/1 · padding mask"], fc=IO, q=False, cost=("0", "—")),
    dict(title="Embeddings",
         lines=["word 30,522×128 + position 512×128 + segment 2×128",
                "→ LayerNorm, dropout → 256 × 128 vectors"], fc=EMB, q=True,
         cost=("3,972,864  (90.6% of all)", "≈0 (table look-ups)")),
]
for i in (1, 2):
    blocks.append(dict(title=f"Transformer encoder layer {i}",
                       lines=["Multi-head self-attention, 2 heads × 64:",
                              "Q, K, V, O = four Linear 128→128,  softmax(QKᵀ/√64)·V",
                              "Add & LayerNorm",
                              "Feed-forward: Linear 128→512 · GELU · Linear 512→128",
                              "Add & LayerNorm → 256 × 128"], fc=ENC, q=True,
                       cost=("198,272", "67.1 M (projections 16.8 M,\nattention 16.8 M, FFN 33.6 M)")))
blocks += [
    dict(title="Pooler + classifier head",
         lines=["[CLS] vector → Linear 128→128, tanh (pooler)",
                "dropout → Linear 128→3 (new, randomly initialised)"], fc=HEAD, q=True,
         cost=("16,512 + 387", "16.4 K + 0.4 K")),
    dict(title="Output", lines=["softmax over  No Fit · Potential Fit · Good Fit"], fc=IO, q=False, cost=None),
]
y_top = 121
centers = {}
for k, b in enumerate(blocks):
    y, h = box(X, y_top, W, b["title"], b["lines"], fc=b["fc"],
               ec=Q_EDGE if b["q"] else INK, lw=2.0 if b["q"] else 1.2, ls="--" if b["q"] else "-")
    centers[b["title"]] = y + h / 2
    if b["cost"]:
        cost(y + h / 2, *b["cost"])
    if k < len(blocks) - 1:
        arrow(cx, y - 0.4, cx, y - GAP + 0.6)
    y_top = y - GAP

ax.text(72, 4.5, "Total: 4,386,307 parameters\n≈134.2 M MACs per pair", ha="left", va="center",
        fontsize=10.5, weight="bold", color=INK)

# M1 annotation (left)
my = centers["Pooler + classifier head"]
ax.add_patch(FancyBboxPatch((-1.5, my - 4), 14.5, 30, boxstyle="round,pad=0.4,rounding_size=1.2", fc=KD, ec=INK, lw=1.2))
ax.text(5.75, my + 24.5, "M1 change", ha="center", va="top", fontsize=10, weight="bold", color=INK)
for i, t in enumerate(["Teacher", "BERT-base", "(fine-tuned,", "training only)", "gives soft", "targets", "", "loss =", "0.5·CE", "+ 0.5·T²·KL"]):
    ax.text(5.75, my + 20.5 - i * 2.4, t, ha="center", va="top", fontsize=8.4, color=INK)
arrow(13.2, my + 2, X - 0.3, my + 2, color="#b91c1c", ls="--")

# M2 legend (top right)
ax.text(72, 117, "M2 change (post-training):", fontsize=9, weight="bold", color=Q_EDGE)
ax.text(72, 114.5, "dashed purple boxes are quantized\n• all Linear layers → dynamic INT8\n• embeddings → 8-bit weight-only\n• LayerNorm stays FP32",
        fontsize=8.4, va="top", color=Q_EDGE)

plt.tight_layout()
plt.savefig(out + ".png", dpi=220, bbox_inches="tight")
plt.savefig(out + ".pdf", bbox_inches="tight")
print("saved", out)
