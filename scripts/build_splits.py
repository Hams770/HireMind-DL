"""Rebuild splits/splits.csv from the raw dataset and check it is byte-identical to the frozen file.

    python scripts/build_splits.py            # downloads the pinned CSVs (about 30 MB)
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hiremind_dl import ROOT, load_config  # noqa: E402
from hiremind_dl.data import download_raw, read_raw, sha256  # noqa: E402
from hiremind_dl.splits import build_splits, save_split_definition, split_counts  # noqa: E402


def main():
    cfg = load_config()
    train = read_raw(download_raw(cfg, "train"))
    test = read_raw(download_raw(cfg, "test"))
    d = build_splits(train, test, seed=cfg["dataset"]["split_seed"])
    counts = {k: v["rows"] for k, v in split_counts(d).items()}
    print("rows per split:", counts)
    assert counts == cfg["dataset"]["rows"], counts
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "splits.csv"
        save_split_definition(d, f)
        got = sha256(f)
    want = cfg["dataset"]["split_sha256"]
    print("rebuilt split SHA-256:", got)
    print("frozen split SHA-256: ", want, "(", ROOT / cfg["dataset"]["split_file"], ")")
    if got != want:
        sys.exit("MISMATCH: the rebuilt split differs from splits/splits.csv")
    print("OK: rebuilt split is identical to splits/splits.csv")


if __name__ == "__main__":
    main()
