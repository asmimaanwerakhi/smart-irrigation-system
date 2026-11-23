"""Metrics and error analysis for the smart irrigation classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.preprocessing import (
    DEFAULT_DATA_PATH,
    PROJECT_ROOT,
    TARGET_COLUMN,
    chronological_split,
    load_dataset,
)

MetricScores = dict[str, dict[str, float]]
ClassificationReports = dict[str, dict[str, Any]]
ConfusionMatrices = dict[str, list[list[int]]]


def evaluate_classifiers(
    models: dict[str, Any],
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[MetricScores, ClassificationReports, ConfusionMatrices]:
    """Return binary metrics, class reports, and confusion matrices."""
    metrics: dict[str, dict[str, float]] = {}
    reports: dict[str, dict[str, Any]] = {}
    matrices: dict[str, list[list[int]]] = {}
    for name, model in models.items():
        predictions = model.predict(x_test)
        metrics[name] = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(
                precision_score(y_test, predictions, pos_label=1, zero_division=0)
            ),
            "recall": float(
                recall_score(y_test, predictions, pos_label=1, zero_division=0)
            ),
            "f1": float(f1_score(y_test, predictions, pos_label=1, zero_division=0)),
            "macro_f1": float(
                f1_score(y_test, predictions, average="macro", zero_division=0)
            ),
        }
        reports[name] = classification_report(
            y_test, predictions, labels=[0, 1], output_dict=True, zero_division=0
        )
        matrices[name] = confusion_matrix(y_test, predictions, labels=[0, 1]).tolist()
    return metrics, reports, matrices


def build_error_analysis(
    test_rows: pd.DataFrame,
    y_test: pd.Series,
    predictions: pd.Series,
) -> pd.DataFrame:
    """Summarize holdout errors per hour using timestamps only for analysis."""
    result = pd.DataFrame(
        {
            "Timestamp": test_rows["Timestamp"].to_numpy(),
            "actual": y_test.to_numpy(),
            "predicted": predictions.to_numpy(),
        }
    )
    result["hour"] = result["Timestamp"].dt.hour
    result["error"] = result["actual"] != result["predicted"]
    return (
        result.groupby("hour", as_index=False)
        .agg(
            observations=("error", "size"),
            errors=("error", "sum"),
            error_rate=("error", "mean"),
        )
    )


def save_evaluation_outputs(
    models: dict[str, Any],
    x_test: pd.DataFrame,
    y_test: pd.Series,
    test_rows: pd.DataFrame,
    output_directory: Path,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, Any]]]:
    """Persist evaluation metrics and per-row holdout predictions."""
    output_directory.mkdir(parents=True, exist_ok=True)
    metrics, reports, matrices = evaluate_classifiers(models, x_test, y_test)
    (output_directory / "metrics.json").write_text(
        json.dumps(
            {
                "target": TARGET_COLUMN,
                "positive_class": 1,
                "test_rows": int(len(y_test)),
                "test_start": test_rows["Timestamp"].min().isoformat(),
                "test_end": test_rows["Timestamp"].max().isoformat(),
                "class_counts": {
                    str(label): int(count)
                    for label, count in y_test.value_counts().sort_index().items()
                },
                "metrics": metrics,
                "confusion_matrices": matrices,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_directory / "classification_report.json").write_text(
        json.dumps(reports, indent=2), encoding="utf-8"
    )

    model_predictions = models["Decision tree"].predict(x_test)
    baseline_predictions = models["Most frequent baseline"].predict(x_test)
    predictions = test_rows[["Timestamp"]].copy()
    predictions["actual"] = y_test.to_numpy()
    predictions["predicted"] = model_predictions
    predictions["baseline_predicted"] = baseline_predictions
    predictions["error"] = predictions["actual"] != predictions["predicted"]
    predictions.to_csv(output_directory / "test_predictions.csv", index=False)
    build_error_analysis(
        test_rows,
        y_test,
        pd.Series(model_predictions, index=y_test.index),
    ).to_csv(output_directory / "error_analysis_by_hour.csv", index=False)
    return metrics, reports


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-evaluate the saved model on the time holdout."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument(
        "--model", type=Path, default=PROJECT_ROOT / "models" / "trained_model.joblib"
    )
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "outputs" / "evaluation"
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    if not args.model.is_file():
        raise FileNotFoundError(f"Saved model not found: {args.model}")
    artifact = joblib.load(args.model)
    dataframe, _ = load_dataset(args.data)
    x_train, x_test, _, y_test, test_rows = chronological_split(
        dataframe, test_size=args.test_size
    )
    if len(x_train) != artifact["training_rows"]:
        raise ValueError(
            "The supplied data/split does not match the saved model's training set "
            f"({len(x_train)} != {artifact['training_rows']} rows)."
        )
    metrics, _ = save_evaluation_outputs(
        {
            "Most frequent baseline": artifact["baseline"],
            "Decision tree": artifact["model"],
        },
        x_test,
        y_test,
        test_rows,
        args.output,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
