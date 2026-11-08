"""Model builders (M0/M1 student, teacher), the distillation loss and seeding."""
import random

import numpy as np
import torch
import torch.nn.functional as F

from . import LABELS


def seed_everything(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def build_model(kind, cfg, seed, config_override=None):
    """kind in {"M0", "M1", "teacher"}. The classifier head is initialised after seeding.

    config_override: a transformers BertConfig for offline tests (random weights, no download).
    """
    from transformers import AutoModelForSequenceClassification, BertForSequenceClassification

    seed_everything(seed)
    if config_override is not None:
        config_override.num_labels = 3
        config_override._attn_implementation = "eager"
        return BertForSequenceClassification(config_override)
    src = cfg["teacher"] if kind == "teacher" else cfg["student"]
    return AutoModelForSequenceClassification.from_pretrained(
        src["name"],
        revision=src["revision"],
        num_labels=3,
        attn_implementation="eager",
        id2label=dict(enumerate(LABELS)),
        label2id={c: i for i, c in enumerate(LABELS)},
    )


def kd_loss(student_logits, labels, teacher_logits, temperature, alpha=0.5):
    """M1 objective: alpha * CE + (1 - alpha) * T^2 * KL(p_teacher || p_student)."""
    ce = F.cross_entropy(student_logits, labels)
    kl = (
        F.kl_div(
            F.log_softmax(student_logits / temperature, dim=-1),
            F.softmax(teacher_logits / temperature, dim=-1),
            reduction="batchmean",
        )
        * temperature**2
    )
    return alpha * ce + (1 - alpha) * kl, ce, kl
