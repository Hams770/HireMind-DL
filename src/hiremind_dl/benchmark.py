"""CPU cost: saved state size, parameter count, MACs and single-thread latency (batch 1)."""
import io
import time
from pathlib import Path

import numpy as np
import torch


def state_size_bytes(model):
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell()


def analytic_macs(model, seq_len=256):
    c = model.config
    L, H, I = seq_len, c.hidden_size, c.intermediate_size
    return c.num_hidden_layers * (4 * L * H * H + 2 * L * L * H + 2 * L * H * I) + H * H + H * 3


def rss_mib(field="VmRSS"):
    f = Path("/proc/self/status")
    if not f.exists():
        return None
    for line in f.read_text().splitlines():
        if line.startswith(field + ":"):
            return int(line.split()[1]) / 1024
    return None


def timing(fn, examples, warmup=30, timed=200):
    for i in range(warmup):
        fn(examples[i % len(examples)])
    xs = []
    for i in range(timed):
        t = time.perf_counter()
        fn(examples[i % len(examples)])
        xs.append((time.perf_counter() - t) * 1000)
    q1, q3 = np.percentile(xs, [25, 75])
    return dict(median_ms=float(np.median(xs)), q1_ms=float(q1), q3_ms=float(q3),
                iqr_ms=float(q3 - q1), p95_ms=float(np.percentile(xs, 95)))


def cpu_benchmark(model, tokenizer, pairs, cfg):
    """pairs: list of (cv_text, job_text). Model-only latency uses pre-tokenised inputs;
    the inclusive latency also tokenises each pair."""
    b = cfg["benchmark"]
    torch.set_num_threads(b["threads"])
    model = model.cpu().eval()
    max_len = cfg["tokenization"]["max_len"]

    def enc(p):
        return tokenizer(p[0], p[1], truncation="longest_first", max_length=max_len,
                         padding="max_length", return_tensors="pt")

    encoded = [enc(p) for p in pairs]
    with torch.inference_mode():
        model_only = timing(lambda x: model(**x), encoded, b["warmup"], b["timed"])
        inclusive = timing(lambda p: model(**enc(p)), pairs, b["warmup"], b["timed"])
    return dict(
        threads=b["threads"], batch_size=1, seq_len=max_len,
        size_MB=state_size_bytes(model) / 1e6,
        latency_ms=model_only["median_ms"], q1_ms=model_only["q1_ms"], q3_ms=model_only["q3_ms"],
        iqr_ms=model_only["iqr_ms"], p95_ms=model_only["p95_ms"],
        inclusive_ms=inclusive["median_ms"], peak_process_MiB=rss_mib("VmHWM"),
    )
