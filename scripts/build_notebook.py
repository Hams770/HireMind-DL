"""Builds notebook/HireMind_M0_Baseline.ipynb from the cells below."""
import json
from pathlib import Path

ROOT = Path(__file__).parent
cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n")})


md("""
# HireMind DL — Assignment 2 (Milestones 1 & 2)
**Dataset audit · grouped split · baseline M0 (BERT-Tiny) · cost measurement**

Rawan Khojah · Hams Aljohani — CS4085 Deep Learning, Effat University

**How to run**
1. `Runtime → Change runtime type → T4 GPU` (CPU also works, just slower).
2. `Runtime → Run all`. Expected time on a T4: about 10–20 minutes.
3. The last cell downloads `HireMind_A2_outputs.zip` (figures, results JSON, split IDs, LaTeX numbers for the report).

**Rules this notebook follows**
- The **test set is never evaluated** here. All model choices use the validation set only.
- No CV or job text is written to the outputs zip (split files contain hashed IDs only).
- The student model is **pretrained** (`google/bert_uncased_L-2_H-128_A-2`); only fine-tuning happens here.
""")

md("## 0. Setup")
code("""
# Colab already has torch, transformers, scikit-learn and matplotlib; nothing is upgraded, so no restart is needed.
!pip -q install huggingface_hub
""")

code("%%writefile data_prep.py\n" + (ROOT / "data_prep.py").read_text())
code("%%writefile bench_cpu.py\n" + (ROOT / "bench_cpu.py").read_text())

code(r'''
import os, json, time, random, platform, hashlib, statistics, copy, shutil, subprocess, sys
import numpy as np, pandas as pd, torch, matplotlib.pyplot as plt
import transformers, sklearn
from huggingface_hub import list_repo_files, hf_hub_download
import importlib, data_prep as dp
importlib.reload(dp)

SEED = 42
REPO = "cnamuangtoun/resume-job-description-fit"
REVISION = "08978e21714984bb417547d2c0f9b477f5298163"   # pinned revision from Assignment 1
EXPECTED_SHA = {"train": "b86fa488a21258015de1750633a9ac410f701ff96bf533f68978dbe3adec6d40",
                "test":  "f3c48f4cc3559bf192ebb9c241a30c601b5b8b3eee6f1656dd4d26dfd4b0887a"}
STUDENT = "google/bert_uncased_L-2_H-128_A-2"
MAX_LEN, BATCH, EPOCHS = 256, 16, 3
LRS = [2e-5, 5e-5, 1e-4, 3e-4]          # M0 search space (4 trials, as planned)
EXTRA_SEEDS = [43, 44]
EVAL_EVERY = 100                          # validation evaluation interval (steps)
OUT = "outputs"
for sub in ["figures", "results", "splits", "tex"]:
    os.makedirs(f"{OUT}/{sub}", exist_ok=True)
os.makedirs("private", exist_ok=True)     # never zipped: contains raw text

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def cpu_name():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"

ENV = {"python": platform.python_version(), "torch": torch.__version__,
       "transformers": transformers.__version__, "sklearn": sklearn.__version__,
       "device": DEVICE, "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else "none",
       "cpu": cpu_name(), "cpu_count": os.cpu_count()}
print(json.dumps(ENV, indent=2))
''')

md("## 1. Load the dataset at the pinned revision and verify file hashes")
code(r'''
files = list_repo_files(REPO, repo_type="dataset", revision=REVISION)
print("Files in repo:", files)
paths = {}
for f in files:
    if f.lower().endswith(".csv"):
        key = "train" if "train" in f.lower() else ("test" if "test" in f.lower() else f)
        paths[key] = hf_hub_download(REPO, f, repo_type="dataset", revision=REVISION)
sha = {k: hashlib.sha256(open(p, "rb").read()).hexdigest() for k, p in paths.items()}
for k in ("train", "test"):
    print(k, "hash matches Assignment 1 audit:", sha.get(k) == EXPECTED_SHA[k])
train_df = pd.read_csv(paths["train"])
test_df = pd.read_csv(paths["test"])
print(train_df.shape, test_df.shape, list(train_df.columns))
''')

md("## 2. Milestone 1 — dataset audit (size, classes, duplicates, leakage)")
code(r'''
data = dp.add_ids(train_df, test_df)
stats, conflicting = dp.audit(data)

# Near-duplicate check (TF-IDF cosine >= 0.95) for job descriptions and CVs
job_texts = dict(zip(data.job_id, data.job_description_text))
cv_texts = dict(zip(data.cv_id, data.resume_text))
job_groups, job_pairs = dp.near_duplicate_groups(job_texts)
cv_groups, cv_pairs = dp.near_duplicate_groups(cv_texts)
train_jobs = set(data.loc[data.orig_split == "train", "job_id"]); test_jobs = set(data.loc[data.orig_split == "test", "job_id"])
cross_job_pairs = [p for p in job_pairs if (p[0] in train_jobs and p[1] in test_jobs) or (p[1] in train_jobs and p[0] in test_jobs)]
stats["near_duplicates"] = {
    "threshold": dp.NEAR_DUP_THRESHOLD,
    "job_pairs": len(job_pairs), "job_pairs_across_train_test": len(cross_job_pairs),
    "job_groups": len(set(job_groups.values())),
    "cv_pairs": len(cv_pairs), "cv_groups": len(set(cv_groups.values())),
}
stats["source"] = {"repo": REPO, "revision": REVISION, "files": files, "sha256": sha,
                   "licence": "none stated in repository (no dataset card)",
                   "instructor_approved": True}
print(json.dumps(stats, indent=2))
''')

code(r'''
# Token lengths with the student tokenizer (to see how much 256-token truncation removes)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(STUDENT)
def n_tokens(texts):
    return [len(x) for x in tok(list(texts), add_special_tokens=False)["input_ids"]]
cv_len = dict(zip(cv_texts.keys(), n_tokens(cv_texts.values())))
job_len = dict(zip(job_texts.keys(), n_tokens(job_texts.values())))
data["pair_tokens"] = data.cv_id.map(cv_len) + data.job_id.map(job_len) + 3   # [CLS] cv [SEP] job [SEP]
stats["tokens"] = {
    "cv_median": float(np.median(list(cv_len.values()))), "job_median": float(np.median(list(job_len.values()))),
    "pair_median": float(data.pair_tokens.median()),
    "pair_p90": float(data.pair_tokens.quantile(0.9)),
    "pct_pairs_truncated_at_256": float(100 * (data.pair_tokens > MAX_LEN).mean()),
}
print(stats["tokens"])

fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
counts = data.groupby(["orig_split", "label"]).size().unstack()[dp.LABELS]
counts.plot.bar(ax=ax[0], rot=0); ax[0].set_title("Class counts per original file"); ax[0].set_xlabel(""); ax[0].set_ylabel("pairs")
ax[1].hist(data.pair_tokens, bins=60, color="#4C72B0"); ax[1].axvline(MAX_LEN, color="red", ls="--", label=f"max length {MAX_LEN}")
ax[1].set_title("CV + job length (BERT word-pieces)"); ax[1].set_xlabel("tokens per pair"); ax[1].set_ylabel("pairs"); ax[1].legend()
plt.tight_layout(); plt.savefig(f"{OUT}/figures/dataset_overview.png", dpi=200); plt.show()
''')

md("## 3. Milestone 1 — clean and split (grouped by job, fixed seed, IDs saved)")
code(r'''
clean_df, removal_log = dp.clean(data, conflicting)
clean_df, n_near = dp.remove_train_rows_sharing_test_job_group(clean_df, job_groups)
removal_log["removed_train_rows_near_duplicate_of_test_job"] = n_near
splits = dp.make_splits(clean_df, job_groups, seed=SEED)
checks = dp.check_splits(splits)
counts = dp.split_counts(splits)
dp.save_split_definition(splits, f"{OUT}/splits/splits.csv")
manifest = {"seed": SEED, "policy": dp.__doc__.strip(), "removals": removal_log,
            "checks": checks, "counts": counts, "source": stats["source"]}
json.dump(manifest, open(f"{OUT}/splits/manifest.json", "w"), indent=2)
json.dump(stats, open(f"{OUT}/results/dataset_stats.json", "w"), indent=2)
print(json.dumps({"removals": removal_log, "checks": checks, "counts": counts}, indent=2))

tbl = pd.DataFrame({k: v["class_counts"] for k, v in counts.items()}).T[dp.LABELS]
tbl.plot.bar(stacked=True, rot=0, figsize=(6, 3.4)); plt.title("Final split sizes by class"); plt.ylabel("pairs")
plt.tight_layout(); plt.savefig(f"{OUT}/figures/split_classes.png", dpi=200); plt.show()
''')

md("## 4. Simple references on validation (majority class, TF-IDF + logistic regression)")
code(r'''
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report

tr = splits[splits.split == "train"]; va = splits[splits.split == "validation"]
def scores(y, p):
    return {"macro_f1": 100 * f1_score(y, p, average="macro"), "accuracy": 100 * accuracy_score(y, p)}

majority = int(tr.label_id.mode()[0])
ref = {"majority": scores(va.label_id, np.full(len(va), majority))}
vec = TfidfVectorizer(ngram_range=(1, 2), max_features=50000, sublinear_tf=True)
Xtr = vec.fit_transform(tr.resume_text + " [SEP] " + tr.job_description_text)
Xva = vec.transform(va.resume_text + " [SEP] " + va.job_description_text)
logreg = LogisticRegression(max_iter=3000, random_state=SEED).fit(Xtr, tr.label_id)
ref["tfidf_logreg"] = scores(va.label_id, logreg.predict(Xva))
print(json.dumps(ref, indent=2))
''')

md("## 5. Milestone 2 — M0 architecture: parameters and MACs per block")
code(r'''
from transformers import AutoModelForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(STUDENT)

def new_model(seed, dropout=None):
    set_seed(seed)                       # makes the new classifier head identical for M0 and (later) M1
    kw = {} if dropout is None else {"hidden_dropout_prob": dropout, "attention_probs_dropout_prob": dropout}
    return AutoModelForSequenceClassification.from_pretrained(STUDENT, num_labels=3, attn_implementation="eager", **kw)

import gc
for name in ("Xtr", "Xva", "vec", "logreg", "cv_texts", "job_texts", "cv_len", "job_len"):
    globals().pop(name, None)           # free memory from earlier sections
gc.collect()

def encode(df, chunk=256, max_chars=6000):
    """Tokenise in small chunks to keep RAM low. Texts are cut to 6,000 characters first;
    only about the first 1,300 characters of each text can survive 256-token truncation."""
    ids, am, tt = [], [], []
    cvs = [t[:max_chars] for t in df.resume_text]; jobs = [t[:max_chars] for t in df.job_description_text]
    for i in range(0, len(cvs), chunk):
        enc = tok(cvs[i:i + chunk], jobs[i:i + chunk], truncation="longest_first",
                  max_length=MAX_LEN, padding="max_length", return_tensors="pt")
        ids.append(enc["input_ids"]); am.append(enc["attention_mask"]); tt.append(enc["token_type_ids"])
    return TensorDataset(torch.cat(ids), torch.cat(am), torch.cat(tt),
                         torch.tensor(df.label_id.values, dtype=torch.long))

train_ds, val_ds = encode(tr), encode(va)     # the test set is NOT encoded in this notebook
print(len(train_ds), len(val_ds))

m = new_model(SEED).eval()
blocks = {"embeddings": m.bert.embeddings, "encoder.layer.0": m.bert.encoder.layer[0],
          "encoder.layer.1": m.bert.encoder.layer[1], "pooler": m.bert.pooler, "classifier": m.classifier}
params = {k: sum(p.numel() for p in b.parameters()) for k, b in blocks.items()}
params["total"] = sum(p.numel() for p in m.parameters())
params["word_embeddings"] = m.bert.embeddings.word_embeddings.weight.numel()

# Measured MACs (FLOPs / 2) for one pair of 256 tokens
from torch.utils.flop_counter import FlopCounterMode
x = {"input_ids": train_ds.tensors[0][:1], "attention_mask": train_ds.tensors[1][:1], "token_type_ids": train_ds.tensors[2][:1]}
with FlopCounterMode(display=False) as fc, torch.no_grad():
    m(**x)
flop_counts = fc.get_flop_counts()
def block_macs(suffix):
    for name, ops in flop_counts.items():
        if name.endswith("." + suffix):
            return sum(ops.values()) / 2
    return 0.0
macs = {k: block_macs(k) for k in blocks}
macs["total"] = fc.get_total_flops() / 2

# Analytic check: per layer 4·L·H² (Q,K,V,O) + 2·L²·H (scores, weighted sum) + 2·L·H·I (feed-forward)
L, H, I = MAX_LEN, m.config.hidden_size, m.config.intermediate_size
analytic_layer = 4 * L * H * H + 2 * L * L * H + 2 * L * H * I
macs_analytic = {"per_layer": analytic_layer, "pooler": H * H, "classifier": H * 3,
                 "total": m.config.num_hidden_layers * analytic_layer + H * H + H * 3}
cost_static = {"params": params, "macs_measured": macs, "macs_analytic": macs_analytic,
               "config": {"layers": m.config.num_hidden_layers, "hidden": H, "heads": m.config.num_attention_heads,
                          "intermediate": I, "vocab": m.config.vocab_size, "max_len": MAX_LEN}}
print(json.dumps(cost_static, indent=2))
''')

md("## 6. Milestone 2 — overfit-one-batch test")
code(r'''
set_seed(SEED)
probe = new_model(SEED, dropout=0.0).to(DEVICE).train()      # dropout off only for this sanity check
idx = torch.randperm(len(train_ds), generator=torch.Generator().manual_seed(SEED))[:16]
b = [t[idx].to(DEVICE) for t in train_ds.tensors]
opt = torch.optim.AdamW(probe.parameters(), lr=1e-3)
overfit_losses = []
for step in range(300):
    out = probe(input_ids=b[0], attention_mask=b[1], token_type_ids=b[2], labels=b[3])
    opt.zero_grad(); out.loss.backward(); opt.step()
    overfit_losses.append(out.loss.item())
probe.eval()
with torch.no_grad():
    acc = (probe(input_ids=b[0], attention_mask=b[1], token_type_ids=b[2]).logits.argmax(-1) == b[3]).float().mean().item()
overfit = {"batch_size": 16, "steps": 300, "lr": 1e-3, "first_loss": overfit_losses[0],
           "final_loss": overfit_losses[-1], "final_accuracy": 100 * acc,
           "passed": bool(overfit_losses[-1] < 0.05 and acc == 1.0)}
print(overfit)
plt.figure(figsize=(5, 3)); plt.plot(overfit_losses); plt.yscale("log"); plt.xlabel("step"); plt.ylabel("training loss")
plt.title("Overfit one batch (16 pairs)"); plt.tight_layout(); plt.savefig(f"{OUT}/figures/overfit_one_batch.png", dpi=200); plt.show()
del probe
''')

md("## 7. Milestone 2 — train M0 (learning-rate search on validation, then 2 extra seeds)")
code(r'''
from transformers import get_linear_schedule_with_warmup

def evaluate(model, ds, bs=128):
    model.eval(); total, n, preds, labels = 0.0, 0, [], []
    with torch.no_grad():
        for ids, am, tt, y in DataLoader(ds, batch_size=bs):
            ids, am, tt, y = ids.to(DEVICE), am.to(DEVICE), tt.to(DEVICE), y.to(DEVICE)
            out = model(input_ids=ids, attention_mask=am, token_type_ids=tt, labels=y)
            total += out.loss.item() * len(y); n += len(y)
            preds.append(out.logits.argmax(-1).cpu()); labels.append(y.cpu())
    p, l = torch.cat(preds).numpy(), torch.cat(labels).numpy()
    return {"loss": total / n, **scores(l, p)}, p, l

def train_m0(lr, seed):
    model = new_model(seed).to(DEVICE)
    loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, generator=torch.Generator().manual_seed(seed))
    total_steps = EPOCHS * len(loader)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sch = get_linear_schedule_with_warmup(opt, int(0.1 * total_steps), total_steps)
    hist = {"step": [], "train_loss": [], "val_loss": [], "val_macro_f1": [], "epoch_end_steps": []}
    best, best_state, step, running = {"macro_f1": -1.0}, None, 0, []
    if DEVICE == "cuda": torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for epoch in range(EPOCHS):
        model.train()
        for ids, am, tt, y in loader:
            ids, am, tt, y = ids.to(DEVICE), am.to(DEVICE), tt.to(DEVICE), y.to(DEVICE)
            loss = model(input_ids=ids, attention_mask=am, token_type_ids=tt, labels=y).loss
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); running.append(loss.item()); step += 1
            if step % EVAL_EVERY == 0 or step == total_steps:
                v, _, _ = evaluate(model, val_ds); model.train()
                hist["step"].append(step); hist["train_loss"].append(float(np.mean(running))); running = []
                hist["val_loss"].append(v["loss"]); hist["val_macro_f1"].append(v["macro_f1"])
                if v["macro_f1"] > best["macro_f1"]:
                    best = {**v, "step": step, "epoch": epoch + 1}
                    best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        hist["epoch_end_steps"].append(step)
    info = {"lr": lr, "seed": seed, "best": best, "history": hist, "steps": total_steps,
            "train_time_s": time.time() - t0,
            "peak_gpu_mem_mib": torch.cuda.max_memory_allocated() / 2**20 if DEVICE == "cuda" else None}
    model.load_state_dict(best_state)
    return model, info

runs, models = [], {}
for lr in LRS:
    model, info = train_m0(lr, SEED)
    runs.append(info); models[lr] = model
    print(f"lr={lr:g}: best val macro-F1 {info['best']['macro_f1']:.2f} (acc {info['best']['accuracy']:.2f}, "
          f"step {info['best']['step']}), {info['train_time_s']:.0f}s")
best_run = max(runs, key=lambda r: (r["best"]["macro_f1"], -r["best"]["loss"]))
BEST_LR = best_run["lr"]
m0 = models[BEST_LR]
print("Selected learning rate (validation only):", BEST_LR)
''')

code(r'''
seed_runs = [best_run]
for s in EXTRA_SEEDS:
    _, info = train_m0(BEST_LR, s)
    seed_runs.append(info)
    print(f"seed {s}: val macro-F1 {info['best']['macro_f1']:.2f}")
f1s = [r["best"]["macro_f1"] for r in seed_runs]; accs = [r["best"]["accuracy"] for r in seed_runs]
seed_summary = {"seeds": [r["seed"] for r in seed_runs], "val_macro_f1": f1s, "val_accuracy": accs,
                "macro_f1_mean": float(np.mean(f1s)), "macro_f1_sd": float(np.std(f1s, ddof=1)),
                "accuracy_mean": float(np.mean(accs)), "accuracy_sd": float(np.std(accs, ddof=1))}
print(json.dumps(seed_summary, indent=2))
''')

md("## 8. Loss curves, learning-rate search and validation confusion matrix")
code(r'''
h = best_run["history"]
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
ax[0].plot(h["step"], h["train_loss"], label="training loss"); ax[0].plot(h["step"], h["val_loss"], label="validation loss")
for s in h["epoch_end_steps"][:-1]: ax[0].axvline(s, color="grey", ls=":", lw=0.8)
ax[0].set_xlabel("step"); ax[0].set_ylabel("cross-entropy"); ax[0].set_title(f"M0 loss curves (lr={BEST_LR:g}, seed {SEED})"); ax[0].legend()
ax[1].plot(h["step"], h["val_macro_f1"], color="green"); ax[1].axhline(ref["majority"]["macro_f1"], color="red", ls="--", label="majority class")
ax[1].set_xlabel("step"); ax[1].set_ylabel("validation macro-F1 (%)"); ax[1].set_title("Validation macro-F1"); ax[1].legend()
plt.tight_layout(); plt.savefig(f"{OUT}/figures/m0_curves.png", dpi=200); plt.show()

plt.figure(figsize=(5, 3))
plt.bar([f"{r['lr']:g}" for r in runs], [r["best"]["macro_f1"] for r in runs], color="#4C72B0")
plt.xlabel("learning rate"); plt.ylabel("best val macro-F1 (%)"); plt.title("M0 learning-rate search (seed 42)")
plt.tight_layout(); plt.savefig(f"{OUT}/figures/m0_lr_search.png", dpi=200); plt.show()

val_metrics, val_pred, val_true = evaluate(m0, val_ds)
cm = confusion_matrix(val_true, val_pred)
report = classification_report(val_true, val_pred, target_names=dp.LABELS, output_dict=True, digits=4)
fig, axc = plt.subplots(figsize=(4.4, 3.8)); im = axc.imshow(cm, cmap="Blues")
axc.set_xticks(range(3), dp.LABELS, rotation=20); axc.set_yticks(range(3), dp.LABELS)
for i in range(3):
    for j in range(3):
        axc.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
axc.set_xlabel("predicted"); axc.set_ylabel("true"); axc.set_title("M0 validation confusion matrix")
plt.tight_layout(); plt.savefig(f"{OUT}/figures/m0_confusion_val.png", dpi=200); plt.show()
print(json.dumps(report, indent=2))
''')

md("## 9. Computational cost of M0 on CPU (fresh process, 1 thread, batch 1, 256 tokens)")
code(r'''
torch.save({k: v.cpu() for k, v in m0.state_dict().items()}, "private/m0_best.pt")
size_bytes = os.path.getsize("private/m0_best.pt")
ex = va.iloc[0]
json.dump({"resume": ex.resume_text, "job": ex.job_description_text}, open("private/example.json", "w"))
subprocess.run([sys.executable, "bench_cpu.py", "--weights", "private/m0_best.pt",
                "--example", "private/example.json", "--out", f"{OUT}/results/m0_cpu_bench.json"], check=True)
bench = json.load(open(f"{OUT}/results/m0_cpu_bench.json"))
print("Serialized state_dict:", size_bytes, "bytes")
''')

md("## 10. Save results, LaTeX numbers for the report, and download")
code(r'''
def tex_escape(s):
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"), ("$", r"\$")]:
        s = str(s).replace(a, b)
    return s

results = {"env": ENV, "references": ref, "overfit": overfit, "lr_search": [{k: r[k] for k in ("lr", "best", "train_time_s", "steps", "peak_gpu_mem_mib")} for r in runs],
           "selected_lr": BEST_LR, "seeds": seed_summary, "val_report": report, "val_confusion": cm.tolist(),
           "cost": {**cost_static, "serialized_bytes": size_bytes, "cpu_bench": bench},
           "best_run_history": best_run["history"], "note": "Validation results only; the test set was not evaluated."}
json.dump(results, open(f"{OUT}/results/m0_baseline.json", "w"), indent=2, default=float)

c = counts; S = stats; R = removal_log
macros = {
  "NRowsRaw": S["rows"], "NTrainRaw": S["train"]["rows"], "NTestRaw": S["test"]["rows"],
  "NCVs": S["unique_cvs"], "NJobs": S["unique_jobs"], "NConflictPairs": S["conflicting_pairs"], "NConflictRows": S["conflicting_rows"],
  "NCVOverlap": S["overlap_train_test"]["cvs"], "NJobOverlap": S["overlap_train_test"]["jobs"],
  "NNearDupJobPairs": S["near_duplicates"]["job_pairs"], "NNearDupJobCross": S["near_duplicates"]["job_pairs_across_train_test"],
  "NNearDupCVPairs": S["near_duplicates"]["cv_pairs"],
  "NRemovedNearDup": R["removed_train_rows_near_duplicate_of_test_job"],
  "NTrain": c["train"]["rows"], "NVal": c["validation"]["rows"], "NTest": c["test"]["rows"],
  "NTrainJobs": c["train"]["jobs"], "NValJobs": c["validation"]["jobs"], "NTestJobs": c["test"]["jobs"],
  "NTrainNo": c["train"]["class_counts"]["No Fit"], "NTrainPot": c["train"]["class_counts"]["Potential Fit"], "NTrainGood": c["train"]["class_counts"]["Good Fit"],
  "NValNo": c["validation"]["class_counts"]["No Fit"], "NValPot": c["validation"]["class_counts"]["Potential Fit"], "NValGood": c["validation"]["class_counts"]["Good Fit"],
  "NTestNo": c["test"]["class_counts"]["No Fit"], "NTestPot": c["test"]["class_counts"]["Potential Fit"], "NTestGood": c["test"]["class_counts"]["Good Fit"],
  "TokPairMedian": f'{S["tokens"]["pair_median"]:.0f}', "TokPairNinety": f'{S["tokens"]["pair_p90"]:.0f}',
  "TruncPct": f'{S["tokens"]["pct_pairs_truncated_at_256"]:.1f}',
  "MajFone": f'{ref["majority"]["macro_f1"]:.1f}', "MajAcc": f'{ref["majority"]["accuracy"]:.1f}',
  "TfidfFone": f'{ref["tfidf_logreg"]["macro_f1"]:.1f}', "TfidfAcc": f'{ref["tfidf_logreg"]["accuracy"]:.1f}',
  "OverfitFirst": f'{overfit["first_loss"]:.3f}', "OverfitFinal": f'{overfit["final_loss"]:.4f}', "OverfitAcc": f'{overfit["final_accuracy"]:.0f}',
  "OverfitPassed": "passed" if overfit["passed"] else "did not pass",
  "MzeroLR": f"{BEST_LR:g}", "MzeroFone": f'{best_run["best"]["macro_f1"]:.1f}', "MzeroAcc": f'{best_run["best"]["accuracy"]:.1f}',
  "MzeroBestStep": best_run["best"]["step"], "MzeroSteps": best_run["steps"],
  "MzeroFoneMean": f'{seed_summary["macro_f1_mean"]:.1f}', "MzeroFoneSD": f'{seed_summary["macro_f1_sd"]:.1f}',
  "MzeroAccMean": f'{seed_summary["accuracy_mean"]:.1f}', "MzeroAccSD": f'{seed_summary["accuracy_sd"]:.1f}',
  "MzeroParams": f'{params["total"]:,}', "MzeroParamsM": f'{params["total"]/1e6:.2f}',
  "MzeroEmbParams": f'{params["embeddings"]:,}', "MzeroLayerParams": f'{params["encoder.layer.0"]:,}',
  "MzeroPoolerParams": f'{params["pooler"]:,}', "MzeroClsParams": f'{params["classifier"]:,}',
  "MzeroEmbShare": f'{100*params["embeddings"]/params["total"]:.1f}',
  "MzeroMACsM": f'{macs["total"]/1e6:.1f}', "MzeroLayerMACsM": f'{macs["encoder.layer.0"]/1e6:.1f}',
  "MzeroMACsAnalyticM": f'{macs_analytic["total"]/1e6:.1f}',
  "MzeroSizeMB": f'{size_bytes/2**20:.2f}', "MzeroLatMed": f'{bench["model_only"]["median_ms"]:.2f}',
  "MzeroLatPninetyfive": f'{bench["model_only"]["p95_ms"]:.2f}', "MzeroLatTokMed": f'{bench["tokenization_inclusive"]["median_ms"]:.2f}',
  "MzeroRSS": f'{bench["peak_rss_mib"]:.0f}', "MzeroRSSBase": f'{bench["rss_after_imports_mib"]:.0f}',
  "MzeroRSSDelta": f'{bench["peak_rss_mib"] - bench["rss_after_imports_mib"]:.0f}',
  "MzeroTrainTime": f'{best_run["train_time_s"]:.0f}',
  "MzeroPeakGPU": f'{best_run["peak_gpu_mem_mib"]:.0f}' if best_run["peak_gpu_mem_mib"] else "n/a",
  "EnvGPU": tex_escape(ENV["gpu"]), "EnvCPU": tex_escape(ENV["cpu"]), "EnvTorch": tex_escape(ENV["torch"]),
  "EnvTransformers": tex_escape(ENV["transformers"]),
}
with open(f"{OUT}/tex/report_numbers.tex", "w") as f:
    f.write("% Auto-generated by HireMind_M0_Baseline.ipynb - do not edit by hand\n")
    for k, v in macros.items():
        f.write(f"\\renewcommand{{\\{k}}}{{{v}}}\n")
lr_rows = " ".join(f"{r['lr']:g} & {r['best']['macro_f1']:.1f} & {r['best']['accuracy']:.1f} & {r['best']['loss']:.3f} & {r['best']['step']} \\\\" for r in runs)
with open(f"{OUT}/tex/report_numbers.tex", "a") as f:
    f.write("\\renewcommand{\\LRRows}{" + lr_rows + "}\n")
with open(f"{OUT}/tex/seed_rows.tex", "w") as f:
    for r in seed_runs:
        f.write(f"{r['seed']} & {r['best']['macro_f1']:.1f} & {r['best']['accuracy']:.1f} \\\\\n")
print(open(f"{OUT}/tex/report_numbers.tex").read())
''')

code(r'''
shutil.make_archive("HireMind_A2_outputs", "zip", OUT)
print("Created HireMind_A2_outputs.zip (no raw text inside). Private files stay in ./private.")
try:
    from google.colab import files
    files.download("HireMind_A2_outputs.zip")
except Exception:
    print("Not running in Colab: download HireMind_A2_outputs.zip from the file browser.")
''')

md("""
### What to do with the outputs
- Upload the **contents** of the zip to the GitHub repo (`figures/`, `results/`, `splits/`) and to Overleaf (`figures/` and `tex/`).
- The `private/` folder holds the M0 checkpoint and one example pair used for timing. **Do not upload it.**
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}, "accelerator": "GPU",
                                   "colab": {"provenance": []}},
      "nbformat": 4, "nbformat_minor": 5}
for c in nb["cells"]:
    c["source"] = c["source"].splitlines(keepends=True)
out = ROOT.parent / "notebooks" / "HireMind_M0_Baseline.ipynb"
out.write_text(json.dumps(nb, indent=1))
print("wrote", out, len(cells), "cells")
