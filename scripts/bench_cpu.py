"""CPU latency and memory benchmark for a saved BERT-Tiny classifier state_dict.

Runs in a fresh process so peak memory is not polluted by training.
Usage:
  python bench_cpu.py --weights m0.pt --example example.json --out bench.json
example.json = {"resume": "...", "job": "..."}  (one validation pair, kept outside the repo)
"""
import argparse
import json
import statistics
import time

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

STUDENT = "google/bert_uncased_L-2_H-128_A-2"


def mem_mib(field):
    """Read VmRSS (current) or VmHWM (peak) of THIS process from /proc/self/status.
    ru_maxrss is not used: on Linux it can carry over the parent notebook's peak after exec."""
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(field + ":"):
                return int(line.split()[1]) / 1024.0  # kB -> MiB
    return float("nan")


def timed(fn, warmup, runs):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return {"median_ms": statistics.median(times), "p95_ms": times[int(0.95 * (len(times) - 1))]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--example", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--runs", type=int, default=200)
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    rss_after_imports = mem_mib("VmRSS")
    tok = AutoTokenizer.from_pretrained(STUDENT)
    model = AutoModelForSequenceClassification.from_pretrained(STUDENT, num_labels=3, attn_implementation="eager")
    model.load_state_dict(torch.load(a.weights, map_location="cpu"))
    model.eval()
    ex = json.load(open(a.example))

    def encode():
        return tok(ex["resume"], ex["job"], truncation="longest_first", max_length=a.max_len,
                   padding="max_length", return_tensors="pt")

    enc = encode()
    with torch.inference_mode():
        model_only = timed(lambda: model(**enc), a.warmup, a.runs)
        with_tok = timed(lambda: model(**encode()), a.warmup, a.runs)
    result = {
        "threads": a.threads, "batch_size": 1, "seq_len": a.max_len,
        "warmup": a.warmup, "runs": a.runs,
        "model_only": model_only, "tokenization_inclusive": with_tok,
        "rss_after_imports_mib": rss_after_imports, "peak_rss_mib": mem_mib("VmHWM"),
        "rss_end_mib": mem_mib("VmRSS"),
        "torch": torch.__version__,
    }
    json.dump(result, open(a.out, "w"), indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
