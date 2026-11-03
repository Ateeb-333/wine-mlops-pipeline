"""Inference verification: load the champion model from the registry and score the test split.

Usage:
    python -m src.evaluate
"""

import argparse
import time
from typing import Dict

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)

from src.data import get_train_test_data
from src.train import CHAMPION_ALIAS, DEFAULT_TRACKING_URI, REGISTERED_MODEL_NAME


def load_champion(tracking_uri: str = DEFAULT_TRACKING_URI):
    """Load the model version currently holding the champion alias."""
    mlflow.set_tracking_uri(tracking_uri)
    model_uri = f"models:/{REGISTERED_MODEL_NAME}@{CHAMPION_ALIAS}"
    return mlflow.sklearn.load_model(model_uri)


def compute_metrics(model, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    """Macro F1, accuracy and log loss on the given data."""
    preds = model.predict(X)
    proba = model.predict_proba(X)
    return {
        "test_f1_macro": float(f1_score(y, preds, average="macro")),
        "test_accuracy": float(accuracy_score(y, preds)),
        "test_log_loss": float(log_loss(y, proba, labels=[0, 1, 2])),
    }


def measure_batch_latency_ms(model, X: pd.DataFrame, repeats: int = 20) -> float:
    """Median wall-clock time (ms) to predict one batch, after a warm-up call."""
    model.predict(X)  # warm-up
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        model.predict(X)
        timings.append((time.perf_counter() - start) * 1000)
    return float(np.median(timings))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the champion WineClassifier")
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    args = parser.parse_args()

    _, X_test, _, y_test = get_train_test_data()
    model = load_champion(args.tracking_uri)

    metrics = compute_metrics(model, X_test, y_test)
    latency = measure_batch_latency_ms(model, X_test)
    preds = model.predict(X_test)

    print(f"Champion model: {type(model).__name__}")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    print(f"batch_latency_ms ({len(X_test)} rows): {latency:.2f}")
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, preds))
    print("\nClassification report:")
    print(classification_report(y_test, preds, digits=4))


if __name__ == "__main__":
    main()
