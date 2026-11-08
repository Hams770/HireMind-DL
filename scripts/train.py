"""Train M0, the teacher, or M1 with the frozen protocol (configs/experiment.json).

    python scripts/train.py --model M0                 # selected lr, seeds 42 43 44
    python scripts/train.py --model teacher            # selected teacher + cached training logits
    python scripts/train.py --model M1                 # selected lr and T, seeds 42 43 44
    python scripts/train.py --model M0 --search        # full learning-rate search at seed 42
    python scripts/train.py --model M1 --search        # lr x T search at seed 42
    python scripts/train.py --model M1 --ablation      # T = 1 ablation, seeds 42 43 44
    python scripts/train.py --model M0 --lr 1e-4 --seed 42   # a single run

Weights and logs go to runs/<tag>/ (git-ignored). A GPU is strongly recommended
(the paper used a Tesla T4); --device cpu works but the teacher is slow.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch  # noqa: E402

from hiremind_dl import load_config  # noqa: E402
from hiremind_dl.data import as_dataset  # noqa: E402
from hiremind_dl.evaluation import evaluate  # noqa: E402
from hiremind_dl.models import build_model  # noqa: E402
from hiremind_dl.runs import CACHE, RUNS, device as pick_device, dump, load_weights, read_info, save_run  # noqa: E402
from hiremind_dl.training import select, tag_for, train_run  # noqa: E402

TEACHER_LOGITS = RUNS / "teacher_train_logits.pt"


def run(kind, lr, seed, cfg, tr, va, dev, temperature=None, max_updates=None):
    tag = tag_for(kind, lr, seed, temperature)
    if (RUNS / tag / "info.json").exists() and max_updates is None:
        print("reuse", tag)
        return read_info(tag)
    teacher = None
    if kind == "M1":
        cache = torch.load(TEACHER_LOGITS, weights_only=True)
        assert cache["training_row_ids"] == json.loads((CACHE / "train_row_ids.json").read_text())
        teacher = cache["logits"]
    info, state = train_run(kind, lr, seed, cfg, tr, va, dev, teacher, temperature, max_updates)
    save_run(info, state)
    print(f"{tag}: best validation macro-F1 {info['best']['macro_f1']:.2f} at update {info['best']['step']}")
    return info


def cache_teacher_logits(cfg, info, tr, dev):
    m = build_model("teacher", cfg, info["seed"])
    m.load_state_dict(load_weights(info["tag"]), strict=True)
    _, logits, _ = evaluate(m.to(dev), tr, dev)
    ids = json.loads((CACHE / "train_row_ids.json").read_text())
    torch.save(dict(logits=logits, training_row_ids=ids, teacher_tag=info["tag"]), TEACHER_LOGITS)
    print("teacher training logits ->", TEACHER_LOGITS, tuple(logits.shape))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["M0", "teacher", "M1"])
    ap.add_argument("--search", action="store_true", help="run the seed-42 hyperparameter search")
    ap.add_argument("--ablation", action="store_true", help="M1 with T = 1 (validation ablation)")
    ap.add_argument("--lr", type=float)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--temperature", type=float)
    ap.add_argument("--max-updates", type=int, help="stop early (quick checks only)")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    cfg = load_config()
    dev = pick_device(a.device)
    tr = as_dataset(CACHE / "train_tokens.pt")
    va = as_dataset(CACHE / "validation_tokens.pt")
    sel, search = cfg["selected"], cfg["search"]
    k = a.model

    if a.lr is not None:  # single run
        run(k, a.lr, a.seed if a.seed is not None else 42, cfg, tr, va, dev, a.temperature, a.max_updates)
        return
    if a.search:
        if k == "M0":
            runs = [run(k, lr, search["seed"], cfg, tr, va, dev) for lr in search["m0_lrs"]]
        elif k == "teacher":
            runs = [run(k, lr, search["seed"], cfg, tr, va, dev) for lr in search["teacher_lrs"]]
        else:
            runs = [run(k, lr, search["seed"], cfg, tr, va, dev, t)
                    for lr in search["m1_lrs"] for t in search["temperatures"]]
        best = select(runs)
        dump(RUNS / f"{k}_search.json", [dict(tag=r["tag"], **{m: r["best"][m] for m in ["macro_f1", "loss", "step"]}) for r in runs])
        print(f"selected {best['tag']} (config says {sel[k]})")
        return
    if k == "teacher":
        info = run(k, sel[k]["lr"], sel[k]["seed"], cfg, tr, va, dev)
        cache_teacher_logits(cfg, info, tr, dev)
        return
    temperature = search["temperature_ablation"] if a.ablation else sel[k].get("temperature")
    seeds = [a.seed] if a.seed is not None else cfg["seeds"]
    for s in seeds:
        run(k, sel[k]["lr"], s, cfg, tr, va, dev, temperature, a.max_updates)


if __name__ == "__main__":
    main()
