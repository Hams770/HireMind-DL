from pathlib import Path
import tempfile
if not all(name in globals() for name in ("WORK", "freeze_final", "setup", "tokens", "error_analysis", "torch")):
    raise RuntimeError("Load S01-S15 in the evaluation notebook first.")
if not (Path(WORK) / "public/final_freeze.json").is_file():
    raise RuntimeError("The frozen evaluation record is required.")
# Temporary runtime storage for derived test tokens; all result paths remain on mounted Drive.
_test_cache_dir = tempfile.TemporaryDirectory(prefix="hiremind_a4_", dir="/content")
TEST_TOKEN_CACHE = Path(_test_cache_dir.name) / "test_tokens.pt"
print("Resume started: checkpoints and freeze will be verified.", flush=True)

def tokens(work, part):
    t = torch.load((TEST_TOKEN_CACHE if part == "test" else Path(work) / "private" / f"{part}_tokens.pt"), weights_only=True)
    return TensorDataset(
        *(t[k] for k in ["input_ids", "attention_mask", "token_type_ids", "labels"]),
        torch.arange(len(t["labels"])),
    )

def final_test(work):
    w, p, v = setup(work)
    freeze = read(p / "final_freeze.json")
    assert freeze["protocol"] == protocol()
    for name, h in freeze["selection_hashes"].items():
        assert sha(p / f"{name}.json") == h, "Selection changed after freeze."
    f = hf_hub_download(DATASET, "test.csv", repo_type="dataset", revision=DATA_REV)
    assert sha(f) == TEST_SHA
    raw = pd.read_csv(f, keep_default_na=False)
    ids = pd.read_csv(ROOT / "splits/splits.csv")
    ids = ids[ids.split == "test"]
    assert (ids.orig_split == "test").all()
    d = raw.iloc[ids.orig_index.to_numpy()].copy().reset_index(drop=True)
    assert len(d) == 1759 and d.label.tolist() == ids.label.tolist()
    for k in ["row_uid", "cv_id", "job_id", "job_group", "label_id"]:
        d[k] = ids[k].to_numpy()
    assert (
        d.resume_text.map(text_id).tolist() == d.cv_id.tolist()
        and d.job_description_text.map(text_id).tolist() == d.job_id.tolist()
    )
    tok = AutoTokenizer.from_pretrained(STUDENT, revision=STUDENT_REV)
    out = {k: [] for k in ["input_ids", "attention_mask", "token_type_ids"]}
    for off in range(0, len(d), 64):
        b = d.iloc[off : off + 64]
        x = tok(
            b.resume_text.tolist(),
            b.job_description_text.tolist(),
            truncation="longest_first",
            max_length=256,
            padding="max_length",
            return_tensors="pt",
        )
        for k in out:
            out[k].append(x[k])
    out = {k: torch.cat(x) for k, x in out.items()}
    out["labels"] = torch.tensor(d.label_id.tolist())
    tf = TEST_TOKEN_CACHE
    if not tf.exists():
        atomic_save(out, tf)
    else:
        assert all(torch.equal(torch.load(tf, weights_only=True)[k], out[k]) for k in out)
    for name, info in selected_models(work):
        print(f"Evaluating or verifying {name}, seed {info['seed']} ...", flush=True)
        dst = p / "final_test" / f"{name}_s{info['seed']}.json"
        if dst.exists():
            r = read(dst)
            assert r["freeze_hash"] == json_hash(freeze)
            assert r["source_sha256"] == info["weights_sha256"]
            assert r["row_ids"] == d.row_uid.tolist()
            assert len(r["predictions"]) == len(d)
            continue
        m = load_trial(work, info)
        if name == "M2":
            row = next(
                x
                for x in read(p / "quantization_results.json")
                if x["seed"] == info["seed"] and x["variant"] == "both"
            )
            assert sha(row["weights"]) == row["weights_sha256"]
            m = quantize(m, "both")
            m.load_state_dict(torch.load(row["weights"], map_location="cpu", weights_only=True))
        met, logits, pred = evaluate(m, tokens(work, "test"), "cpu")
        frozen_write(
            dst,
            dict(
                metrics=met,
                freeze_hash=json_hash(freeze),
                source_sha256=info["weights_sha256"],
                row_ids=d.row_uid.tolist(),
                predictions=pred,
                confidence=torch.softmax(logits, dim=-1).max(dim=-1).values.tolist(),
            ),
        )
        del m
        gc.collect()
    return {"test": d}

EXPECTED_SHA256 = {
    "M0_s42_lr0.0001.pt": "6ac7862edea03cba6c9f7854c45f7c7fc470b0c54518ebb8791f571f08997d71",
    "M0_s43_lr0.0001.pt": "08fd438d0393d724787e95f9c1e72d9e689065cd6944d1fd87e2ad3f57ffe4d7",
    "M0_s44_lr0.0001.pt": "9cce30b993ec289e8f94620405740c1857db91269030284344b38b6fd71f6d2b",
    "M1_s42_lr5e-05_T2_6d4c0c0a.pt": "83e2e432c327538043bd6f897ef488d7d9d5a809866cd7a4379ab6c95d1f73c8",
    "M1_s43_lr5e-05_T2_f277f079.pt": "41304d79a8cdca3ac7c2644e1a94c116fc88448500c7efb985589ce1a210ca63",
    "M1_s44_lr5e-05_T2_af7b8c78.pt": "e3c522152fe77dfded904b3be4c567e70647be3abd3dd0384571a8f9231a6e93",
    "Q_both_s42_66c50970.pt": "60304b9343f3be6f4af1898c8b68cc381903559972fe9de71e78420db8b4c3f0",
    "Q_both_s43_426e2838.pt": "df375db3d44e3a0812b1956e1c51f3584a530f9499ee6df55fcc9c3d6df1303e",
    "Q_both_s44_8ecc0a82.pt": "ddfbc746e14f6edaaa5dcbbd2fc0eef46311e9468fe37fbcc3734575143e2d03"
}
public = Path(WORK) / "public"
verified = set()
for filename in ("M0_selected.json", "M1_selected.json", "quantization_results.json"):
    records = read(public / filename)
    selected = [r for r in records if r.get("variant", "both") == "both"]
    assert sorted(int(r["seed"]) for r in selected) == [42, 43, 44], filename
    for record in selected:
        path = checkpoint_path(WORK, record)
        expected = EXPECTED_SHA256[path.name]
        if sha(path) != expected or record["weights_sha256"] != expected:
            raise ValueError(f"Checkpoint mismatch: {path.name}. Evaluation stopped.")
        verified.add(path.name)
assert verified == set(EXPECTED_SHA256)
print("Verified all nine Assignment 3 checkpoints.")
torch.set_num_threads(1)
setup(WORK)
freeze_final(WORK, "FREEZE AFTER ITEM 3 REVIEW")
with torch.inference_mode():
    test_frames = final_test(WORK)
final_table = comparison_tables(WORK, "test")
error_analysis(WORK, test_frames, "test")
print(final_hypotheses(WORK))
assert len(final_table) == 9
print("A4 COMPLETE | 9 test records | public/reports/test/per_seed.csv")
