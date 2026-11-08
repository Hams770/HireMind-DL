"""Offline engineering test of the whole pipeline (about 1 minute on a laptop CPU).

Uses a randomly initialised BERT-Tiny-shaped model, a toy tokenizer and synthetic text, so it
needs no download and no GPU. It checks that the code runs; its numbers are NOT results.

    python scripts/smoke_test.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from transformers import BertConfig, BertTokenizerFast  # noqa: E402

from hiremind_dl import load_config  # noqa: E402
from hiremind_dl.benchmark import cpu_benchmark, state_size_bytes  # noqa: E402
from hiremind_dl.data import as_dataset, tokenize  # noqa: E402
from hiremind_dl.evaluation import evaluate  # noqa: E402
from hiremind_dl.models import kd_loss  # noqa: E402
from hiremind_dl.quantization import quantize  # noqa: E402
from hiremind_dl.splits import build_splits, text_id  # noqa: E402
from hiremind_dl.training import select, train_run  # noqa: E402

WORDS = "python java sql cloud nurse sales design data manager engineer teacher finance".split()


def synthetic(n, rng):
    cvs = [" ".join(rng.choice(WORDS, 12)) + f" cv{i % 40}" for i in range(n)]
    jobs = [" ".join(rng.choice(WORDS, 15)) + f" job{i % 25}" for i in range(n)]
    labels = rng.choice(["No Fit", "Potential Fit", "Good Fit"], n, p=[0.5, 0.25, 0.25])
    return pd.DataFrame(dict(resume_text=cvs, job_description_text=jobs, label=labels))


def main():
    torch.set_num_threads(1)
    cfg = load_config()
    cfg["training"]["eval_every"] = 5
    rng = np.random.default_rng(0)
    checks = []

    # 1. split pipeline on synthetic frames
    train_df, test_df = synthetic(300, rng), synthetic(80, rng)
    d = build_splits(train_df, test_df, seed=42)
    assert set(d.split) == {"train", "validation", "test"}
    assert not set(d[d.split == "train"].job_group) & set(d[d.split == "validation"].job_group)
    checks.append("split pipeline: job-disjoint train/validation")

    # 2. tokenizer + tokenisation (toy vocabulary, same call as the real pipeline)
    with tempfile.TemporaryDirectory() as tmp:
        vocab = Path(tmp) / "vocab.txt"
        vocab.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + WORDS +
                                   [f"cv{i}" for i in range(40)] + [f"job{i}" for i in range(25)]))
        tok = BertTokenizerFast(vocab_file=str(vocab), do_lower_case=True)
    frames = {k: d[d.split == k].reset_index(drop=True) for k in ["train", "validation"]}
    assert frames["train"].resume_text.map(text_id).tolist() == frames["train"].cv_id.tolist()
    max_len = 64
    tr = as_dataset(tokenize(frames["train"], tok, max_len))
    va = as_dataset(tokenize(frames["validation"], tok, max_len))
    checks.append(f"tokenised {len(tr)} train / {len(va)} validation pairs")

    # 3. KD loss sanity
    s = torch.randn(4, 3, requires_grad=True)
    t, y = torch.randn(4, 3), torch.tensor([0, 1, 2, 0])
    loss, ce, kl = kd_loss(s, y, t, 2.0)
    loss.backward()
    assert torch.isfinite(s.grad).all()
    assert abs(kd_loss(t, y, t, 2.0)[2].item()) < 1e-6
    assert torch.allclose(loss, 0.5 * F.cross_entropy(s, y) + 0.5 * kl)
    checks.append("KD loss: finite gradients, zero KL for identical targets")

    # 4. training loop for M0, teacher (micro-batches of 4) and M1
    def tiny():
        return BertConfig(vocab_size=tok.vocab_size, hidden_size=128, num_hidden_layers=2,
                          num_attention_heads=2, intermediate_size=512, max_position_embeddings=max_len)

    runs = []
    for lr in [5e-5, 1e-4]:
        info, m0_state = train_run("M0", lr, 42, cfg, tr, va, "cpu", max_updates=10,
                                   config_override=tiny(), log=lambda *_: None)
        runs.append(info)
    assert select(runs)["best"]["macro_f1"] >= 0
    tinfo, _ = train_run("teacher", 5e-5, 42, cfg, tr, va, "cpu", max_updates=5,
                         config_override=tiny(), log=lambda *_: None)
    assert tinfo["micro_batch"] == 4
    teacher_logits = torch.randn(len(tr), 3)
    m1info, m1_state = train_run("M1", 5e-5, 42, cfg, tr, va, "cpu", teacher_logits, 2.0,
                                 max_updates=10, config_override=tiny(), log=lambda *_: None)
    assert m1info["updates"] == 10 and len(m1info["history"]) == 2
    checks.append("training loop: M0 search + selection, teacher micro-batching, M1 distillation")

    # 5. M2 quantisation, save/reload, evaluation and CPU benchmark
    from hiremind_dl.models import build_model
    m1 = build_model("M1", cfg, 42, tiny())
    m1.load_state_dict(m1_state, strict=True)
    fp_met, _, _ = evaluate(m1, va, "cpu")
    for variant in ["linear", "embeddings", "both"]:
        q = quantize(m1, variant)
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "q.pt"
            torch.save(q.state_dict(), f)
            r = quantize(m1, variant)
            r.load_state_dict(torch.load(f, weights_only=True))
        a, _, _ = evaluate(q, va, "cpu")
        b, _, _ = evaluate(r, va, "cpu")
        assert a["confusion"] == b["confusion"]
    m2 = quantize(m1, "both")
    ratio = state_size_bytes(m2) / state_size_bytes(m1)
    assert ratio < 0.4, ratio
    checks.append(f"M2: all variants save/reload identically; size ratio {ratio:.3f}")
    bcfg = dict(cfg, benchmark=dict(cfg["benchmark"], warmup=3, timed=10),
                tokenization=dict(cfg["tokenization"], max_len=max_len))
    pairs = list(zip(frames["validation"].resume_text[:5], frames["validation"].job_description_text[:5]))
    bench = cpu_benchmark(m2, tok, pairs, bcfg)
    assert bench["latency_ms"] > 0
    checks.append("CPU benchmark runs (1 thread, batch 1)")

    print("SMOKE TEST PASSED (random weights and synthetic data; not research results)")
    for c in checks:
        print("  -", c)


if __name__ == "__main__":
    main()
