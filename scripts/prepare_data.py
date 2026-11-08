"""Download the pinned dataset, attach the frozen split and tokenize train/validation/test.

    python scripts/prepare_data.py              # writes data/cache/{train,validation,test}_tokens.pt
    python scripts/prepare_data.py --no-test    # Item 3 stage: do not touch the test file

The cache holds raw CV text and is git-ignored; it is never committed.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from hiremind_dl import load_config  # noqa: E402
from hiremind_dl.data import download_raw, load_manifest, read_raw, split_frame, tokenize  # noqa: E402
from hiremind_dl.runs import CACHE, dump  # noqa: E402
from hiremind_dl.splits import check_splits  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-test", action="store_true", help="skip the held-out test file")
    args = ap.parse_args()
    cfg = load_config()
    manifest = load_manifest(cfg)
    print("split checks:", check_splits(manifest))
    raw = {"train": read_raw(download_raw(cfg, "train"))}
    parts = ["train", "validation"]
    if not args.no_test:
        raw["test"] = read_raw(download_raw(cfg, "test"))
        parts.append("test")
    tok = AutoTokenizer.from_pretrained(cfg["student"]["name"], revision=cfg["student"]["revision"])
    CACHE.mkdir(parents=True, exist_ok=True)
    frames = {}
    for part in parts:
        d = split_frame(raw, manifest, part, cfg["dataset"]["rows"][part])
        frames[part] = d
        t = tokenize(d, tok, cfg["tokenization"]["max_len"])
        torch.save(t, CACHE / f"{part}_tokens.pt")
        dump(CACHE / f"{part}_row_ids.json", d.row_uid.tolist())
        print(f"{part}: {len(d)} pairs -> {CACHE / f'{part}_tokens.pt'}")
    bench = frames["validation"].sample(n=cfg["benchmark"]["examples"], random_state=42)
    dump(CACHE / "benchmark_inputs.json",
         bench[["row_uid", "resume_text", "job_description_text"]].to_dict("records"))
    print("benchmark inputs:", bench.row_uid.tolist())


if __name__ == "__main__":
    main()
