"""Draws the M0 (BERT-Tiny pair classifier) architecture with per-block parameters and MACs.
Also marks what changes in M1 (distillation loss) and M2 (quantized modules).
Drawn at its printed size (one ACL column, 3.3 in wide) so the text stays readable.
Numbers are analytic for google/bert_uncased_L-2_H-128_A-2 at 256 tokens; the notebook
re-measures them with torch.utils.flop_counter.FlopCounterMode.
"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

out = sys.argv[1] if len(sys.argv) > 1 else "m0_architecture"

INK = "#1f2937"; MUTED = "#4b5563"
EMB = "#fde68a"; ENC = "#bfdbfe"; HEAD = "#bbf7d0"; IO = "#f3f4f6"
Q_EDGE = "#7c3aed"; KD = "#fecaca"; KD_EDGE = "#b91c1c"

W_IN, H_IN = 3.3, 4.05
fig = plt.figure(figsize=(W_IN, H_IN))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100); ax.set_ylim(0, 100 * H_IN / W_IN); ax.axis("off")
YMAX = 100 * H_IN / W_IN
LINE = 3.1          # line spacing in data units (~7 pt text)
TITLE_FS, LINE_FS, COST_FS = 7.6, 6.5, 6.5


def box(x, y_top, w, title, lines=(), cost=None, fc=IO, ec=INK, lw=0.9, ls="-"):
    n = len(lines) + (1 if cost else 0)
    h = 4.6 + LINE * n
    y = y_top - h
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.0",
                                fc=fc, ec=ec, lw=lw, ls=ls))
    ax.text(x + w / 2, y_top - 0.9, title, ha="center", va="top", fontsize=TITLE_FS, weight="bold", color=INK)
    for i, t in enumerate(lines):
        ax.text(x + w / 2, y_top - 4.3 - i * LINE, t, ha="center", va="top", fontsize=LINE_FS, color=INK)
    if cost:
        ax.text(x + w / 2, y_top - 4.3 - len(lines) * LINE, cost, ha="center", va="top",
                fontsize=COST_FS, color=MUTED, style="italic")
    return y, h


def arrow(x1, y1, x2, y2, color=INK, ls="-", style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=7,
                                 color=color, lw=0.9, ls=ls, shrinkA=0, shrinkB=0))


ax.text(50, YMAX - 1.2, "M0: BERT-Tiny cross-encoder (L=2, H=128, A=2)", ha="center", va="top",
        fontsize=8.2, weight="bold", color=INK)

X, W, GAP = 3, 94, 3.0
cx = X + W / 2
blocks = [
    dict(title="Input", lines=["CV text  +  job description text"], fc=IO, q=False),
    dict(title="WordPiece tokenizer (30,522-word vocabulary)",
         lines=["[CLS] CV [SEP] job [SEP] → 256 token ids",
                "longest-first truncation · segment ids · padding mask"], fc=IO, q=False,
         cost="0 parameters · 0 MACs"),
    dict(title="Embeddings",
         lines=["word 30,522×128 + position 512×128 + segment 2×128",
                "LayerNorm, dropout → 256 × 128 vectors"], fc=EMB, q=True,
         cost="3,972,864 parameters (90.6%) · ≈0 MACs (look-ups)"),
    dict(title="Transformer encoder layer  × 2",
         lines=["self-attention: 2 heads × 64, Q/K/V/O Linear 128→128",
                "feed-forward: Linear 128→512 · GELU · Linear 512→128",
                "residual + LayerNorm after each sub-layer"], fc=ENC, q=True,
         cost="198,272 parameters · 67.1 M MACs per layer"),
    dict(title="Pooler + classifier head",
         lines=["[CLS] → Linear 128→128, tanh → dropout → Linear 128→3",
                "(the 128→3 layer is new, randomly initialised)"], fc=HEAD, q=True,
         cost="16,512 + 387 parameters · 16.9 K MACs"),
    dict(title="Output", lines=["softmax over  No Fit · Potential Fit · Good Fit"], fc=IO, q=False),
]
y_top = YMAX - 7.0
centers = {}
for k, b in enumerate(blocks):
    y, h = box(X, y_top, W, b["title"], b["lines"], b.get("cost"), fc=b["fc"],
               ec=Q_EDGE if b["q"] else INK, lw=1.3 if b["q"] else 0.9, ls="--" if b["q"] else "-")
    centers[b["title"]] = (y, h)
    if k < len(blocks) - 1:
        arrow(cx, y - 0.2, cx, y - GAP + 0.3)
    y_top = y - GAP

ax.text(50, y_top - 0.2, "Total: 4,386,307 parameters · 134.2 M MACs per pair (256 tokens)",
        ha="center", va="top", fontsize=6.6, weight="bold", color=INK)

# M1 change (training only)
y_m1 = y_top - 5.0
y, h = box(X, y_m1, W, "M1 change (training only)",
           ["BERT-base teacher (fine-tuned) gives soft targets for the logits:",
            "loss = 0.5·CE(label) + 0.5·T²·KL(teacher ‖ student)"], fc=KD, ec=KD_EDGE, lw=1.1)

# M2 legend
ax.text(50, y - 2.0, "Dashed purple = quantized in M2 (post-training): Linear → dynamic INT8,",
        ha="center", va="top", fontsize=6.3, color=Q_EDGE)
ax.text(50, y - 2.0 - LINE * 0.95, "embeddings → 8-bit weight-only; LayerNorm stays FP32.",
        ha="center", va="top", fontsize=6.3, color=Q_EDGE)

plt.savefig(out + ".png", dpi=300, bbox_inches="tight", pad_inches=0.02)
plt.savefig(out + ".pdf", bbox_inches="tight", pad_inches=0.02)
print("saved", out)
