"""Automated MLOps Quality Gate.

The MLflow registry (mlflow.db / mlruns) is not committed to Git, so CI cannot
load the registered champion. Instead, the gate re-runs the same deterministic
selection used by src/train.py (seed 42, 5-fold stratified CV) and validates the
resulting champion. Because everything is seeded, this is the same model that
`make train` registers locally.
"""

import numpy as np
import pytest

from src.data import get_train_test_data
from src.evaluate import compute_metrics, measure_batch_latency_ms
from src.train import build_model, cross_validate_model, iter_candidates, select_best

F1_THRESHOLD = 0.88
LATENCY_THRESHOLD_MS = 30.0
VALID_CLASSES = {0, 1, 2}


@pytest.fixture(scope="module")
def champion():
    X_train, X_test, y_train, y_test = get_train_test_data()
    results = []
    for family, config_id, params in iter_candidates():
        metrics = cross_validate_model(build_model(family, params), X_train, y_train)
        results.append({"family": family, "config_id": config_id,
                        "params": params, **metrics})
    best = select_best(results)
    model = build_model(best["family"], best["params"]).fit(X_train, y_train)
    return {"model": model, "best": best, "X_test": X_test, "y_test": y_test}


def test_gate_validation_macro_f1(champion):
    val_f1 = champion["best"]["val_f1_macro"]
    assert val_f1 >= F1_THRESHOLD, (
        f"{champion['best']['config_id']} val macro F1 {val_f1:.4f} < {F1_THRESHOLD}"
    )


def test_gate_test_macro_f1(champion):
    metrics = compute_metrics(champion["model"], champion["X_test"], champion["y_test"])
    assert metrics["test_f1_macro"] >= F1_THRESHOLD


def test_gate_batch_inference_latency(champion):
    latency = measure_batch_latency_ms(champion["model"], champion["X_test"])
    assert latency <= LATENCY_THRESHOLD_MS, (
        f"Batch latency {latency:.2f} ms exceeds {LATENCY_THRESHOLD_MS} ms"
    )


def test_gate_output_schema(champion):
    X_test = champion["X_test"]
    preds = champion["model"].predict(X_test)
    assert len(preds) == len(X_test)
    assert np.issubdtype(preds.dtype, np.integer)
    assert set(np.unique(preds)).issubset(VALID_CLASSES)


def test_gate_probability_schema(champion):
    proba = champion["model"].predict_proba(champion["X_test"])
    assert proba.shape == (len(champion["X_test"]), len(VALID_CLASSES))
    assert np.allclose(proba.sum(axis=1), 1.0)
