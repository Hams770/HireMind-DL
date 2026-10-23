"""HireMind DL - dataset audit, cleaning and grouped train/validation/test split.

Pure pandas / scikit-learn (no PyTorch), so it can be tested anywhere.
Used by notebooks/HireMind_M0_Baseline.ipynb.

Split policy (fixed before any model training):
  * IDs: cv_id / job_id = first 16 hex chars of SHA-256 of whitespace-normalised, lower-cased text.
  * Near-duplicate job descriptions (TF-IDF cosine >= NEAR_DUP_THRESHOLD) share one job_group.
  * Every row of a pair that carries conflicting labels is removed (any split).
  * Exact duplicate pairs: one copy is kept; if a pair occurs in both files, the training copy is dropped.
  * Training rows whose job is a near-duplicate of a test job are dropped.
  * The official test file is kept as the test set (only the removals above).
  * Validation = one fold of StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    on the cleaned training file, grouped by job_group, stratified by label.
"""
import hashlib
import json
import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedGroupKFold

LABELS = ["No Fit", "Potential Fit", "Good Fit"]
LABEL2ID = {name: i for i, name in enumerate(LABELS)}
NEAR_DUP_THRESHOLD = 0.95


def norm(text):
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def text_id(text):
    return hashlib.sha256(norm(text).encode("utf-8")).hexdigest()[:16]


def add_ids(train_df, test_df):
    """Return one frame with split, orig_index, cv_id, job_id, pair_id, label_id."""
    frames = []
    for name, df in (("train", train_df), ("test", test_df)):
        d = df[["resume_text", "job_description_text", "label"]].copy()
        d["orig_split"] = name
        d["orig_index"] = np.arange(len(d))
        frames.append(d)
    data = pd.concat(frames, ignore_index=True)
    unknown = set(data["label"]) - set(LABELS)
    if unknown:
        raise ValueError(f"Unexpected labels: {unknown}")
    data["cv_id"] = data["resume_text"].map(text_id)
    data["job_id"] = data["job_description_text"].map(text_id)
    data["pair_id"] = data["cv_id"] + "_" + data["job_id"]
    data["label_id"] = data["label"].map(LABEL2ID)
    data["row_uid"] = data["orig_split"] + "_" + data["orig_index"].astype(str)
    return data


def near_duplicate_groups(texts_by_id, threshold=NEAR_DUP_THRESHOLD):
    """Union-find over TF-IDF cosine similarity. Returns {id: group_id} and the list of near-dup pairs."""
    ids = list(texts_by_id.keys())
    if len(ids) < 2:
        return {i: i for i in ids}, []
    tfidf = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    matrix = tfidf.fit_transform([texts_by_id[i] for i in ids])
    sim = cosine_similarity(matrix)
    np.fill_diagonal(sim, 0.0)
    parent = list(range(len(ids)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    pairs = []
    rows, cols = np.where(np.triu(sim, k=1) >= threshold)
    for r, c in zip(rows, cols):
        pairs.append((ids[r], ids[c], float(sim[r, c])))
        parent[find(r)] = find(c)
    groups = {ids[i]: ids[find(i)] for i in range(len(ids))}
    return groups, pairs


def audit(data):
    """Dataset statistics before cleaning."""
    stats = {"rows": int(len(data))}
    for split in ("train", "test"):
        d = data[data.orig_split == split]
        stats[split] = {
            "rows": int(len(d)),
            "class_counts": {k: int(v) for k, v in Counter(d["label"]).items()},
            "unique_cvs": int(d.cv_id.nunique()),
            "unique_jobs": int(d.job_id.nunique()),
        }
    stats["class_counts_total"] = {k: int(v) for k, v in Counter(data["label"]).items()}
    stats["unique_cvs"] = int(data.cv_id.nunique())
    stats["unique_jobs"] = int(data.job_id.nunique())
    labels_per_pair = data.groupby("pair_id")["label"].nunique()
    conflicting = labels_per_pair[labels_per_pair > 1].index
    stats["unique_pairs"] = int(data.pair_id.nunique())
    stats["duplicate_pair_excess_rows"] = int(len(data) - data.pair_id.nunique())
    stats["conflicting_pairs"] = int(len(conflicting))
    stats["conflicting_rows"] = int(data.pair_id.isin(conflicting).sum())
    tr, te = data[data.orig_split == "train"], data[data.orig_split == "test"]
    stats["overlap_train_test"] = {
        "cvs": int(len(set(tr.cv_id) & set(te.cv_id))),
        "jobs": int(len(set(tr.job_id) & set(te.job_id))),
        "pairs": int(len(set(tr.pair_id) & set(te.pair_id))),
    }
    return stats, set(conflicting)


def clean(data, conflicting_pairs):
    """Apply the removal rules; return cleaned frame and a removal log."""
    log = {}
    before = data.orig_split.value_counts().to_dict()
    d = data[~data.pair_id.isin(conflicting_pairs)].copy()
    log["removed_conflicting_rows"] = {
        s: int(before.get(s, 0) - (d.orig_split == s).sum()) for s in ("train", "test")
    }
    test_pairs = set(d.loc[d.orig_split == "test", "pair_id"])
    mask_cross = (d.orig_split == "train") & d.pair_id.isin(test_pairs)
    log["removed_train_rows_duplicated_in_test"] = int(mask_cross.sum())
    d = d[~mask_cross]
    n0 = d.orig_split.value_counts().to_dict()
    d = d.drop_duplicates(subset=["orig_split", "pair_id"], keep="first")
    log["removed_within_split_duplicates"] = {
        s: int(n0.get(s, 0) - (d.orig_split == s).sum()) for s in ("train", "test")
    }
    return d.reset_index(drop=True), log


def remove_train_rows_sharing_test_job_group(d, job_groups):
    """Drop training rows whose job is a near-duplicate of a test job (prevents job-level leakage)."""
    test_groups = {job_groups[j] for j in d.loc[d.orig_split == "test", "job_id"]}
    mask = (d.orig_split == "train") & d.job_id.map(job_groups).isin(test_groups)
    return d[~mask].reset_index(drop=True), int(mask.sum())


def make_splits(d, job_groups, seed=42, n_splits=5):
    """Grouped, stratified validation split from the cleaned training file."""
    d = d.copy()
    d["job_group"] = d["job_id"].map(job_groups)
    tr = d[d.orig_split == "train"]
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    _, val_idx = next(sgkf.split(tr, tr["label_id"], groups=tr["job_group"]))
    val_rows = set(tr.index[val_idx])
    d["split"] = np.where(d.orig_split == "test", "test",
                          np.where(d.index.isin(val_rows), "validation", "train"))
    return d


def check_splits(d):
    """Assertions that the split policy holds. Returns a dict of checks."""
    s = {k: d[d.split == k] for k in ("train", "validation", "test")}
    checks = {
        "train_val_job_group_overlap": len(set(s["train"].job_group) & set(s["validation"].job_group)),
        "train_test_job_overlap": len(set(s["train"].job_id) & set(s["test"].job_id)),
        "val_test_job_overlap": len(set(s["validation"].job_id) & set(s["test"].job_id)),
        "pair_overlap_any": len(set(s["train"].pair_id) & set(s["validation"].pair_id))
        + len(set(s["train"].pair_id) & set(s["test"].pair_id))
        + len(set(s["validation"].pair_id) & set(s["test"].pair_id)),
        "classes_in_validation": int(s["validation"].label_id.nunique()),
        "cv_overlap_train_test": len(set(s["train"].cv_id) & set(s["test"].cv_id)),
        "cv_overlap_train_val": len(set(s["train"].cv_id) & set(s["validation"].cv_id)),
    }
    assert checks["train_val_job_group_overlap"] == 0, "job group leaks between train and validation"
    assert checks["pair_overlap_any"] == 0, "identical pair in two splits"
    assert checks["classes_in_validation"] == 3, "validation is missing a class"
    return checks


def split_counts(d):
    out = {}
    for k in ("train", "validation", "test"):
        part = d[d.split == k]
        out[k] = {
            "rows": int(len(part)),
            "cvs": int(part.cv_id.nunique()),
            "jobs": int(part.job_id.nunique()),
            "class_counts": {LABELS[i]: int((part.label_id == i).sum()) for i in range(3)},
        }
    return out


def save_split_definition(d, path):
    """Save IDs only (no CV or job text is written)."""
    cols = ["row_uid", "orig_split", "orig_index", "split", "cv_id", "job_id", "job_group", "pair_id", "label", "label_id"]
    d[cols].to_csv(path, index=False)


if __name__ == "__main__":
    print(json.dumps({"labels": LABELS, "near_dup_threshold": NEAR_DUP_THRESHOLD}))
