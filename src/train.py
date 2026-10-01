"""Train RandomForest and GradientBoosting families with 5-fold CV and MLflow tracking.

Usage:
    python -m src.train
    python -m src.train --tracking-uri sqlite:///mlflow.db --no-register
"""

import argparse
import os
import time
from typing import Any, Dict, List

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.data import RANDOM_STATE, get_train_test_data

EXPERIMENT_NAME = "Wine-Cultivar-Classification"
REGISTERED_MODEL_NAME = "WineClassifier"
CHAMPION_ALIAS = "champion"
DEFAULT_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
CV_FOLDS = 5

MODEL_FAMILIES = {
    "RandomForest": RandomForestClassifier,
    "GradientBoosting": GradientBoostingClassifier,
}

SEARCH_SPACE: Dict[str, List[Dict[str, Any]]] = {
    "RandomForest": [
        {"n_estimators": 60, "max_depth": 3, "min_samples_leaf": 1},
        {"n_estimators": 100, "max_depth": 5, "min_samples_leaf": 2},
        {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 1},
    ],
    "GradientBoosting": [
        {"n_estimators": 50, "learning_rate": 0.1, "max_depth": 2},
        {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 3},
        {"n_estimators": 150, "learning_rate": 0.1, "max_depth": 3, "subsample": 0.8},
    ],
}

SCORING = {
    "f1_macro": "f1_macro",
    "accuracy": "accuracy",
    "log_loss": "neg_log_loss",
}


def build_model(family: str, params: Dict[str, Any]):
    """Instantiate a classifier of the given family with a fixed seed."""
    if family not in MODEL_FAMILIES:
        raise ValueError(f"Unknown model family: {family}")
    return MODEL_FAMILIES[family](random_state=RANDOM_STATE, **params)


def cross_validate_model(model, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    """Run stratified 5-fold CV and return mean/std train & validation metrics."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_validate(
        model, X, y, cv=cv, scoring=SCORING, return_train_score=True, n_jobs=None
    )
    metrics: Dict[str, float] = {}
    for name in SCORING:
        for split, prefix in (("train", "train"), ("test", "val")):
            values = scores[f"{split}_{name}"]
            if name == "log_loss":
                values = -values  # sklearn returns negative log loss
            metrics[f"{prefix}_{name}"] = float(np.mean(values))
            metrics[f"{prefix}_{name}_std"] = float(np.std(values))
    return metrics


def iter_candidates():
    """Yield (family, config_id, params) for every configuration in the search space."""
    for family, configs in SEARCH_SPACE.items():
        for idx, params in enumerate(configs, start=1):
            yield family, f"{family}-cfg{idx}", params


def select_best(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pick best result by val macro F1; ties broken by lower val log loss."""
    return sorted(results, key=lambda r: (-r["val_f1_macro"], r["val_log_loss"]))[0]


def run_search(tracking_uri: str, experiment_name: str) -> List[Dict[str, Any]]:
    """Run the full search, logging one MLflow run per configuration."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    X_train, _, y_train, _ = get_train_test_data()
    batch_id = time.strftime("%Y%m%d-%H%M%S")
    results = []

    for family, config_id, params in iter_candidates():
        with mlflow.start_run(run_name=config_id) as run:
            model = build_model(family, params)
            metrics = cross_validate_model(model, X_train, y_train)

            # Refit on the full training split for artifact logging.
            model.fit(X_train, y_train)

            mlflow.log_params(params)
            mlflow.log_params({"random_state": RANDOM_STATE, "cv_folds": CV_FOLDS})
            mlflow.log_metrics(metrics)
            mlflow.set_tags({
                "model_family": family,
                "config_id": config_id,
                "dataset": "sklearn.load_wine",
                "search_batch": batch_id,
            })

            input_example = X_train.head(5)
            signature = infer_signature(X_train, model.predict(X_train))
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                signature=signature,
                input_example=input_example,
            )

            results.append({
                "run_id": run.info.run_id,
                "family": family,
                "config_id": config_id,
                "params": params,
                **metrics,
            })
            print(f"[{config_id}] val_f1_macro={metrics['val_f1_macro']:.4f} "
                  f"val_acc={metrics['val_accuracy']:.4f} "
                  f"val_log_loss={metrics['val_log_loss']:.4f}")
    return results


def register_champion(run_id: str, tracking_uri: str) -> str:
    """Register the run's model and point the champion alias at the new version."""
    mlflow.set_tracking_uri(tracking_uri)
    version = mlflow.register_model(f"runs:/{run_id}/model", REGISTERED_MODEL_NAME)
    client = MlflowClient()
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS, version.version)
    client.set_model_version_tag(
        REGISTERED_MODEL_NAME, version.version, "selection_metric", "val_f1_macro"
    )
    return version.version


def results_table(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Build a summary table for the report (Table 1)."""
    rows = []
    for r in results:
        rows.append({
            "Config": r["config_id"],
            "Params": ", ".join(f"{k}={v}" for k, v in r["params"].items()),
            "Train F1": round(r["train_f1_macro"], 4),
            "Val F1": round(r["val_f1_macro"], 4),
            "Train Acc": round(r["train_accuracy"], 4),
            "Val Acc": round(r["val_accuracy"], 4),
            "Train LogLoss": round(r["train_log_loss"], 4),
            "Val LogLoss": round(r["val_log_loss"], 4),
        })
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wine classifier training pipeline")
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--experiment-name", default=EXPERIMENT_NAME)
    parser.add_argument("--no-register", action="store_true",
                        help="Skip model registry promotion")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_search(args.tracking_uri, args.experiment_name)

    table = results_table(results)
    print("\nHyperparameter search results:")
    print(table.to_string(index=False))
    table.to_csv("search_results.csv", index=False)

    best = select_best(results)
    print(f"\nBest run: {best['config_id']} (run_id={best['run_id']}, "
          f"val_f1_macro={best['val_f1_macro']:.4f})")

    if not args.no_register:
        version = register_champion(best["run_id"], args.tracking_uri)
        print(f"Registered {REGISTERED_MODEL_NAME} v{version} "
              f"with alias '{CHAMPION_ALIAS}'")


if __name__ == "__main__":
    main()
