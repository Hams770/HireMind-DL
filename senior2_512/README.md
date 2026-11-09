# Senior Project II extension: M0 with 512-token inputs

The DL paper's classifier reads 256 tokens of each CV and job together, while CVs have a median of about
1,563 tokens. The Senior Project II midterm report commits to retraining with 512 tokens. This folder does
that without changing anything else, and without touching the paper's 256-token results.

**How to run (Google Colab, T4 GPU, about 1 hour):**
[open the notebook in Colab](https://colab.research.google.com/github/Hams770/HireMind-DL/blob/main/senior2_512/Retrain_M0_512_Colab.ipynb)
and run the cells in order. The same steps from a terminal:

```sh
export HIREMIND_MAX_LEN=512
python scripts/prepare_data.py --no-test
python scripts/train.py --model M0                              # seeds 42, 43, 44, lr 1e-4
python scripts/evaluate.py --split validation --models M0      # compare with the 256 validation results
# only if 512 is better on validation:
python scripts/prepare_data.py && python scripts/evaluate.py --split test --models M0 --benchmark
```

| | 256 tokens (paper) | 512 tokens |
|---|---|---|
| Weights and logs | `runs/` | `runs_L512/` |
| Token cache | `data/cache/` | `data/cache_L512/` |
| Results | `outputs/` | `outputs/L512/` |

Choosing between 256 and 512 uses validation only; the test split is opened once, after the choice.
Weights are not committed: send `HireMind_M0_L512.zip` (made by the last notebook cell) to the team.
