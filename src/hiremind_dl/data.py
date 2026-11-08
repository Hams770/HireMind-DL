"""Download the pinned dataset, attach the frozen split, and tokenize CV-job pairs."""
import hashlib
from pathlib import Path

import pandas as pd

from . import ROOT
from .splits import text_id


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download_raw(cfg, part):
    """Return the local path of train.csv or test.csv at the pinned revision (checksum verified)."""
    from huggingface_hub import hf_hub_download

    ds = cfg["dataset"]
    f = hf_hub_download(ds["name"], f"{part}.csv", repo_type="dataset", revision=ds["revision"])
    expected = ds[f"{part}_csv_sha256"]
    if sha256(f) != expected:
        raise ValueError(f"{part}.csv checksum mismatch: expected {expected}")
    return f


def read_raw(path):
    return pd.read_csv(path, keep_default_na=False)


def load_manifest(cfg):
    f = ROOT / cfg["dataset"]["split_file"]
    if sha256(f) != cfg["dataset"]["split_sha256"]:
        raise ValueError("splits/splits.csv does not match the frozen split SHA-256")
    return pd.read_csv(f)


def split_frame(raw_by_part, manifest, split, expected_rows=None):
    """Rows of one split, in manifest order, with ids checked against the text."""
    rows = manifest[manifest.split == split]
    source = "test" if split == "test" else "train"
    assert (rows.orig_split == source).all()
    d = raw_by_part[source].iloc[rows.orig_index.to_numpy()].copy().reset_index(drop=True)
    assert d.label.tolist() == rows.label.tolist(), "labels differ from the split file"
    for k in ["row_uid", "cv_id", "job_id", "job_group", "label_id"]:
        d[k] = rows[k].to_numpy()
    assert d.resume_text.map(text_id).tolist() == d.cv_id.tolist(), "CV text does not match cv_id"
    assert d.job_description_text.map(text_id).tolist() == d.job_id.tolist(), "job text does not match job_id"
    if expected_rows is not None:
        assert len(d) == expected_rows, f"{split}: {len(d)} rows, expected {expected_rows}"
    return d


def tokenize(frame, tokenizer, max_len=256, chunk=64):
    """Pair encoding exactly as in the notebooks: (CV, job), longest_first, padded to max_len."""
    import torch

    out = {k: [] for k in ["input_ids", "attention_mask", "token_type_ids"]}
    for off in range(0, len(frame), chunk):
        b = frame.iloc[off : off + chunk]
        enc = tokenizer(
            b.resume_text.tolist(),
            b.job_description_text.tolist(),
            truncation="longest_first",
            max_length=max_len,
            padding="max_length",
            return_tensors="pt",
        )
        for k in out:
            out[k].append(enc[k])
    out = {k: torch.cat(x) for k, x in out.items()}
    out["labels"] = torch.tensor(frame.label_id.tolist())
    return out


def as_dataset(tokens):
    """TensorDataset(input_ids, attention_mask, token_type_ids, labels, row_index)."""
    import torch
    from torch.utils.data import TensorDataset

    if isinstance(tokens, (str, Path)):
        tokens = torch.load(tokens, weights_only=True)
    return TensorDataset(
        *(tokens[k] for k in ["input_ids", "attention_mask", "token_type_ids", "labels"]),
        torch.arange(len(tokens["labels"])),
    )
