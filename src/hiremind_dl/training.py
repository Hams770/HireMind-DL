"""One training run (M0, teacher or M1) with the frozen protocol.

AdamW (weight decay 0.01), linear schedule with 10% warm-up, gradient clipping 1.0,
batch 16 (teacher: 4 micro-batches of 4), 3 epochs = 939 updates, validation every
100 updates and at the last update; the best validation macro-F1 checkpoint is kept.
"""
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .evaluation import batch_args, evaluate
from .models import build_model, kd_loss


def tag_for(kind, lr, seed, temperature=None):
    """Same naming as the notebooks, e.g. M1_s42_lr5e-05_T2."""
    return f"{kind}_s{seed}_lr{lr:g}" + (f"_T{temperature:g}" if temperature else "")


def train_run(kind, lr, seed, cfg, train_ds, val_ds, device, teacher_logits=None,
              temperature=None, max_updates=None, config_override=None, log=print):
    tc = cfg["training"]
    if kind == "M1" and (teacher_logits is None or temperature is None):
        raise ValueError("M1 needs cached teacher logits and a temperature")
    model = build_model(kind, cfg, seed, config_override).to(device)
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(train_ds, batch_size=tc["batch"], shuffle=True, generator=g, num_workers=0)
    total = tc["epochs"] * len(loader)
    if len(train_ds) == cfg["dataset"]["rows"]["train"]:
        assert total == tc["updates"]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=tc["weight_decay"])
    from transformers import get_linear_schedule_with_warmup

    sch = get_linear_schedule_with_warmup(opt, int(tc["warmup_fraction"] * total), total)
    micro = tc["teacher_micro_batch"] if kind == "teacher" else tc["student_micro_batch"]
    alpha = cfg["selected"]["M1"]["alpha"]
    stop = total if max_updates is None else min(total, max_updates)

    step, running, history = 0, [], []
    best, best_state = {"macro_f1": -1.0}, None
    started = time.perf_counter()
    for epoch in range(tc["epochs"]):
        model.train()
        for b in loader:
            opt.zero_grad(set_to_none=True)
            batch_loss = 0.0
            for off in range(0, len(b[3]), micro):
                bb = tuple(t[off : off + micro] for t in b)
                args = batch_args(bb, device)
                out = model(**args)
                if teacher_logits is None:
                    loss = out.loss
                else:
                    loss = kd_loss(out.logits, args["labels"], teacher_logits[bb[4]].to(device),
                                   temperature, alpha)[0]
                fraction = len(bb[3]) / len(b[3])
                (loss * fraction).backward()
                batch_loss += loss.item() * fraction
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc["clip_norm"])
            opt.step()
            sch.step()
            step += 1
            running.append(batch_loss)
            if step % tc["eval_every"] == 0 or step == stop:
                met, _, _ = evaluate(model, val_ds, device, tc["validation_batch"])
                history.append(dict(step=step, epoch=epoch + 1,
                                    train_objective=float(np.mean(running)),
                                    validation_ce=met["loss"],
                                    validation_macro_f1=met["macro_f1"],
                                    validation_accuracy=met["accuracy"]))
                running = []
                if met["macro_f1"] > best["macro_f1"]:  # first best evaluation wins ties
                    best = {**met, "step": step}
                    best_state = {k: x.detach().cpu().clone() for k, x in model.state_dict().items()}
                log(f"{tag_for(kind, lr, seed, temperature)} update {step}/{stop} "
                    f"validation macro-F1 {met['macro_f1']:.3f}")
                model.train()
            if step >= stop:
                break
        if step >= stop:
            break
    info = dict(tag=tag_for(kind, lr, seed, temperature), kind=kind, lr=lr, seed=seed,
                temperature=temperature, updates=step, best=best, history=history,
                seconds=time.perf_counter() - started, micro_batch=micro,
                effective_batch=tc["batch"], device=str(device))
    return info, best_state


def select(runs):
    """Highest best-checkpoint validation macro-F1; ties broken by lower validation loss."""
    return max(runs, key=lambda r: (r["best"]["macro_f1"], -r["best"]["loss"]))
