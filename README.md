# Wine Cultivar Classification — MLOps Pipeline

![CI](https://github.com/<YOUR_USERNAME>/wine-mlops-pipeline/actions/workflows/ci.yml/badge.svg)

A reproducible MLOps pipeline for 3-class wine cultivar classification (`sklearn.datasets.load_wine`,
178 samples, 13 features). It covers Makefile automation, 5-fold CV hyperparameter search over two
tree-based model families, MLflow experiment tracking and model registry, and a GitHub Actions
quality gate that blocks degrading models from reaching `main`.

## Repository structure

```
wine-mlops-pipeline/
├── .github/workflows/ci.yml   # CI: install -> lint -> test (incl. quality gate)
├── data/.gitkeep              # dataset is loaded from sklearn at runtime
├── src/
│   ├── data.py                # load, validate, stratified 80/20 split (seed 42)
│   ├── train.py               # CV search, MLflow tracking, champion registration
│   └── evaluate.py            # load WineClassifier@champion, test-set metrics
├── tests/
│   ├── test_data.py           # data pipeline unit tests
│   └── test_model_gate.py     # F1, latency and output-schema quality gate
├── Makefile
└── requirements.txt
```

## Quick start

Requires Python 3.10 and GNU Make.

```bash
python3.10 -m venv .venv
source .venv/bin/activate        # Windows (Git Bash): source .venv/Scripts/activate
make install
make lint
make test
make train
make evaluate
make mlflow-ui                   # open http://127.0.0.1:5000
```

| Target | Action |
|---|---|
| `install` | Upgrade pip and install pinned dependencies |
| `lint` | flake8 on `src/` and `tests/`, max line length 100 |
| `test` | pytest (verbose) — data tests and model quality gate |
| `train` | Run the CV search, log all runs to MLflow, register the champion |
| `evaluate` | Load `models:/WineClassifier@champion` and score the test split |
| `mlflow-ui` | Launch the MLflow UI on the SQLite backend |
| `clean` | Remove bytecode, caches and temporary files |
| `clean-mlflow` | Delete local MLflow tracking data and registry |

## Pipeline details

**Data.** Stratified 80/20 split with `random_state=42` (142 train / 36 test). Validation checks
reject null values, a feature count other than 13, and unexpected class labels.

**Search space.** Three configurations each for `RandomForestClassifier` and
`GradientBoostingClassifier`, evaluated with 5-fold `StratifiedKFold` (shuffled, seed 42) on the
training split. Train and validation Macro F1, Accuracy and Log Loss (mean and std) are recorded.

**MLflow.** Backend: `sqlite:///mlflow.db`. Experiment: `Wine-Cultivar-Classification`. Each
configuration is a separate run with parameters, metrics, tags, an inferred signature, an input
example and the model logged via `mlflow.sklearn.log_model`. The run with the highest validation
macro F1 (ties broken by lower validation log loss) is registered as `WineClassifier` and given the
`champion` alias.

**Quality gate.** `tests/test_model_gate.py` enforces:

- Validation macro F1 ≥ 0.88
- Batch inference latency ≤ 30 ms (median of 20 runs after a warm-up)
- Predictions are integer class indices in {0, 1, 2}

Because the MLflow store is not committed, CI cannot load the registered model. The gate therefore
re-runs the same seeded selection logic from `src/train.py`, which reproduces the exact champion
that `make train` registers locally.

## Reproducibility

All splits, CV folds and model initialisations use seed 42, and every dependency that affects
behaviour is pinned in `requirements.txt` (including SQLAlchemy, since unpinned 2.1.x breaks
MLflow 2.17's SQLite store).
