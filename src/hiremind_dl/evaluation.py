"""Validation/test metrics: loss, accuracy, macro-F1, per-class report, confusion matrix."""
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from . import LABELS

KEYS = ["input_ids", "attention_mask", "token_type_ids", "labels"]


def batch_args(b, device, labels=True):
    ks = KEYS if labels else KEYS[:3]
    return {k: b[i].to(device) for i, k in enumerate(ks)}


@torch.inference_mode()
def evaluate(model, ds, device="cpu", batch_size=16):
    model.eval()
    ys, pred, logits, total = [], [], [], 0.0
    for b in DataLoader(ds, batch_size=batch_size, shuffle=False):
        out = model(**batch_args(b, device))
        assert torch.isfinite(out.logits).all()
        total += out.loss.item() * len(b[3])
        ys.extend(b[3].tolist())
        pred.extend(out.logits.argmax(-1).cpu().tolist())
        logits.append(out.logits.cpu())
    cm = confusion_matrix(ys, pred, labels=[0, 1, 2])
    metrics = dict(
        loss=total / len(ys),
        accuracy=100 * accuracy_score(ys, pred),
        macro_f1=100 * f1_score(ys, pred, labels=[0, 1, 2], average="macro", zero_division=0),
        good_fit_recall=100 * cm[2, 2] / max(cm[2].sum(), 1),
        per_class=classification_report(
            ys, pred, labels=[0, 1, 2], target_names=LABELS, output_dict=True, zero_division=0
        ),
        confusion=cm.tolist(),
    )
    return metrics, torch.cat(logits), pred
