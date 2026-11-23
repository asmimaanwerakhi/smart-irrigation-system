"""Load and validate persisted project artifacts for the Streamlit dashboard."""

from __future__ import annotations

import json
import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_DIR = PROJECT_ROOT / "outputs" / "evaluation"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
MODEL_PATH = PROJECT_ROOT / "models" / "trained_model.joblib"

FIGURE_FILES = {
    "Sensor distributions": "sensor_distributions.png",
    "Missing values": "missing_values.png",
    "Correlation heatmap": "correlation_heatmap.png",
    "Soil moisture timeline": "soil_moisture_timeline.png",
    "Environmental hourly profiles": "environmental_hourly_profiles.png",
    "Target distribution": "target_distribution.png",
    "Confusion matrix": "confusion_matrix.png",
    "Model comparison": "model_performance_comparison.png",
    "Prediction timeline": "test_predictions_timeline.png",
    "Hourly error rate": "error_rate_by_hour.png",
}


@dataclass
class DashboardData:
    """Validated saved outputs; missing resources are represented as ``None``."""

    metrics: dict[str, Any] | None = None
    classification_report: dict[str, Any] | None = None
    training_summary: dict[str, Any] | None = None
    profile: dict[str, Any] | None = None
    profile_markdown: str | None = None
    predictions: pd.DataFrame | None = None
    error_by_hour: pd.DataFrame | None = None
    feature_importance: pd.DataFrame | None = None
    model_metadata: dict[str, Any] | None = None
    figures: dict[str, Path] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)


@st.cache_data(show_spinner=False)
def _read_json_cached(path: str, modified_ns: int) -> Any:
    del modified_ns
    return json.loads(Path(path).read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def _read_csv_cached(path: str, modified_ns: int) -> pd.DataFrame:
    del modified_ns
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def _read_text_cached(path: str, modified_ns: int) -> str:
    del modified_ns
    return Path(path).read_text(encoding="utf-8")


@st.cache_data(show_spinner=False)
def _read_model_metadata_cached(path: str, modified_ns: int) -> dict[str, Any]:
    del modified_ns
    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or "model" not in artifact:
        raise ValueError("The saved model artifact does not contain a model pipeline.")

    pipeline = artifact["model"]
    steps = getattr(pipeline, "named_steps", {})
    classifier = steps.get("classifier")
    if classifier is None:
        raise ValueError("The saved pipeline has no 'classifier' step.")

    classes = getattr(classifier, "classes_", [])
    return {
        "estimator_name": type(classifier).__name__,
        "feature_columns": artifact.get("feature_columns", []),
        "target_column": artifact.get("target_column"),
        "training_rows": artifact.get("training_rows"),
        "max_depth": getattr(classifier, "max_depth", None),
        "min_samples_leaf": getattr(classifier, "min_samples_leaf", None),
        "random_state": getattr(classifier, "random_state", None),
        "classes": classes.tolist() if hasattr(classes, "tolist") else list(classes),
    }


def _read_json(
    path: Path,
    label: str,
    issues: list[str],
) -> dict[str, Any] | None:
    if not path.is_file():
        issues.append(f"{label} is missing: {path.relative_to(PROJECT_ROOT)}")
        return None
    try:
        modified_ns = path.stat().st_mtime_ns
        payload = _read_json_cached(str(path), modified_ns)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        issues.append(f"Could not read {label}: {exc}")
        return None
    if not isinstance(payload, dict):
        issues.append(f"{label} must contain a JSON object.")
        return None
    return payload


def _read_csv(
    path: Path,
    label: str,
    required_columns: set[str],
    issues: list[str],
) -> pd.DataFrame | None:
    if not path.is_file():
        issues.append(f"{label} is missing: {path.relative_to(PROJECT_ROOT)}")
        return None
    try:
        modified_ns = path.stat().st_mtime_ns
        dataframe = _read_csv_cached(str(path), modified_ns).copy()
    except (
        OSError,
        UnicodeDecodeError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ) as exc:
        issues.append(f"Could not read {label}: {exc}")
        return None

    missing_columns = sorted(required_columns - set(dataframe.columns))
    if missing_columns:
        issues.append(f"{label} is missing required columns: {missing_columns}")
        return None
    return dataframe


def _read_markdown(
    path: Path,
    label: str,
    issues: list[str],
) -> str | None:
    if not path.is_file():
        issues.append(f"{label} is missing: {path.relative_to(PROJECT_ROOT)}")
        return None
    try:
        return _read_text_cached(str(path), path.stat().st_mtime_ns)
    except (OSError, UnicodeDecodeError) as exc:
        issues.append(f"Could not read {label}: {exc}")
        return None


def _validate_metrics(
    metrics: dict[str, Any] | None,
    issues: list[str],
) -> dict[str, Any] | None:
    if metrics is None:
        return None
    issue_count = len(issues)
    required = {
        "target",
        "positive_class",
        "test_rows",
        "test_start",
        "test_end",
        "class_counts",
        "metrics",
        "confusion_matrices",
    }
    missing = sorted(required - set(metrics))
    if missing:
        issues.append(f"metrics.json is missing required keys: {missing}")
        return None
    if (
        metrics["target"] != "Best Time"
        or metrics["positive_class"] != 1
        or isinstance(metrics["positive_class"], bool)
    ):
        issues.append(
            "metrics.json target or positive-class definition does not match "
            "the documented dashboard target."
        )
        return None
    if not isinstance(metrics["metrics"], dict):
        issues.append("metrics.json 'metrics' must be a JSON object.")
        return None
    if not isinstance(metrics["confusion_matrices"], dict):
        issues.append("metrics.json 'confusion_matrices' must be a JSON object.")
        return None
    if not isinstance(metrics["class_counts"], dict):
        issues.append("metrics.json 'class_counts' must be a JSON object.")
        return None
    if set(metrics["class_counts"]) != {"0", "1"}:
        issues.append("metrics.json class counts must contain classes '0' and '1'.")
        return None
    test_rows = metrics["test_rows"]
    class_values = list(metrics["class_counts"].values())
    if (
        not isinstance(test_rows, int)
        or isinstance(test_rows, bool)
        or any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in class_values
        )
    ):
        issues.append("metrics.json has invalid test-row or class-count values.")
        return None
    class_total = sum(class_values)
    if (
        test_rows < 0
        or any(value < 0 for value in class_values)
        or class_total != test_rows
    ):
        issues.append("metrics.json test-row and class-count totals do not agree.")

    required_scores = {"accuracy", "precision", "recall", "f1", "macro_f1"}
    for model_name in ("Most frequent baseline", "Decision tree"):
        scores = metrics["metrics"].get(model_name)
        if not isinstance(scores, dict):
            issues.append(f"metrics.json has no metrics for {model_name}.")
        elif required_scores - set(scores):
            issues.append(
                f"metrics.json has incomplete metrics for {model_name}: "
                f"{sorted(required_scores - set(scores))}"
            )
        else:
            for key in required_scores:
                try:
                    if isinstance(scores[key], bool) or not isinstance(
                        scores[key], (int, float)
                    ):
                        raise TypeError("score must be numeric")
                    value = float(scores[key])
                except (TypeError, ValueError):
                    issues.append(
                        "metrics.json has a non-numeric "
                        f"{key!r} score for {model_name}."
                    )
                    break
                if not math.isfinite(value) or not 0 <= value <= 1:
                    issues.append(
                        "metrics.json has an out-of-range "
                        f"{key!r} score for {model_name}."
                    )
                    break

        matrix = metrics["confusion_matrices"].get(model_name)
        if not isinstance(matrix, list) or len(matrix) != 2 or any(
            not isinstance(row, list) or len(row) != 2 for row in matrix
        ):
            issues.append(
                f"metrics.json has an invalid confusion matrix for {model_name}."
            )
        elif any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for row in matrix
            for value in row
        ):
            issues.append(
                f"metrics.json has invalid confusion-matrix counts for {model_name}."
            )
        elif sum(sum(row) for row in matrix) != test_rows:
            issues.append(
                f"metrics.json confusion matrix for {model_name} does not match "
                "the test-row count."
            )
        elif any(
            sum(matrix[class_index]) != int(metrics["class_counts"][str(class_index)])
            for class_index in (0, 1)
        ):
            issues.append(
                f"metrics.json confusion matrix for {model_name} does not match "
                "the saved class counts."
            )
    if len(issues) > issue_count:
        return None
    return metrics


def _validate_predictions(
    predictions: pd.DataFrame | None,
    expected_rows: int | None,
    issues: list[str],
) -> pd.DataFrame | None:
    if predictions is None:
        return None
    predictions = predictions.copy()
    predictions["Timestamp"] = pd.to_datetime(predictions["Timestamp"], errors="coerce")
    if predictions["Timestamp"].isna().any():
        issues.append("test_predictions.csv contains invalid timestamps.")
        return None

    for column in ("actual", "predicted", "baseline_predicted"):
        values = pd.to_numeric(predictions[column], errors="coerce")
        if values.isna().any() or not values.isin([0, 1]).all():
            issues.append(
                f"test_predictions.csv has invalid binary values in {column!r}."
            )
            return None
        predictions[column] = values.astype("int64")

    if predictions["error"].dtype != bool:
        error_values = (
            predictions["error"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"true": True, "false": False, "1": True, "0": False})
        )
        if error_values.isna().any():
            issues.append("test_predictions.csv contains invalid error flags.")
            return None
        predictions["error"] = error_values.astype(bool)

    observed_errors = predictions["actual"] != predictions["predicted"]
    if not predictions["error"].equals(observed_errors):
        issues.append(
            "Saved prediction error flags do not match actual/predicted values; "
            "the dashboard uses the actual/predicted comparison."
        )
        predictions["error"] = observed_errors

    if expected_rows is not None and len(predictions) != expected_rows:
        issues.append(
            "Prediction row count does not match metrics.json "
            f"({len(predictions):,} vs {expected_rows:,})."
        )
    return predictions


def _validate_error_by_hour(
    error_by_hour: pd.DataFrame | None,
    issues: list[str],
) -> pd.DataFrame | None:
    if error_by_hour is None:
        return None
    error_by_hour = error_by_hour.copy()
    if error_by_hour.empty:
        issues.append("error_analysis_by_hour.csv contains no hourly records.")
        return None
    for column in ("hour", "observations", "errors", "error_rate"):
        error_by_hour[column] = pd.to_numeric(error_by_hour[column], errors="coerce")
        if error_by_hour[column].isna().any():
            issues.append(f"error_analysis_by_hour.csv has invalid {column!r} values.")
            return None
    if not error_by_hour["hour"].isin(range(24)).all():
        issues.append("error_analysis_by_hour.csv contains an hour outside 0-23.")
        return None
    if error_by_hour["hour"].duplicated().any():
        issues.append("error_analysis_by_hour.csv contains duplicate hours.")
        return None
    if (
        (error_by_hour["observations"] < 0).any()
        or (error_by_hour["errors"] < 0).any()
        or (error_by_hour["errors"] > error_by_hour["observations"]).any()
        or (~error_by_hour["error_rate"].between(0, 1)).any()
        or (
            error_by_hour[["observations", "errors"]]
            .mod(1)
            .ne(0)
            .any()
            .any()
        )
    ):
        issues.append("error_analysis_by_hour.csv contains invalid counts or rates.")
        return None
    expected_rates = error_by_hour["errors"].div(
        error_by_hour["observations"].replace(0, math.nan)
    ).fillna(0)
    if not (
        (error_by_hour["error_rate"] - expected_rates).abs() <= 1e-9
    ).all():
        issues.append(
            "error_analysis_by_hour.csv rates do not match its saved counts."
        )
        return None
    return error_by_hour.sort_values("hour").reset_index(drop=True)


def _validate_profile(
    profile: dict[str, Any] | None,
    issues: list[str],
) -> dict[str, Any] | None:
    if profile is None:
        return None
    shape = profile.get("shape")
    columns = profile.get("columns")
    timestamp = profile.get("timestamp")
    if (
        not isinstance(shape, dict)
        or not isinstance(shape.get("rows"), int)
        or isinstance(shape.get("rows"), bool)
        or not isinstance(shape.get("columns"), int)
        or isinstance(shape.get("columns"), bool)
        or not isinstance(columns, dict)
        or not isinstance(timestamp, dict)
    ):
        issues.append(
            "dataset_profile.json has an invalid shape, columns, or timestamp section."
        )
        return None
    if shape["rows"] < 0 or shape["columns"] < 0:
        issues.append("dataset_profile.json has negative dataset dimensions.")
        return None
    for key in ("sensor_rows_with_any_missing", "duplicate_full_rows"):
        value = profile.get(key, 0)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            issues.append(
                f"dataset_profile.json has an invalid {key.replace('_', ' ')} value."
            )
            return None
    for column, detail in columns.items():
        if not isinstance(detail, dict):
            issues.append(
                f"dataset_profile.json has invalid details for column {column!r}."
            )
            return None
        statistics = detail.get("statistics", {})
        if not isinstance(statistics, dict):
            issues.append(
                f"dataset_profile.json has invalid statistics for {column!r}."
            )
            return None
        numeric_values = [
            detail.get("missing_count"),
            detail.get("missing_percent"),
            *statistics.values(),
        ]
        if any(
            value is not None
            and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            )
            for value in numeric_values
        ):
            issues.append(
                f"dataset_profile.json has non-numeric statistics for {column!r}."
            )
            return None
    return profile


def _validate_training_summary(
    summary: dict[str, Any] | None,
    issues: list[str],
) -> dict[str, Any] | None:
    if summary is None:
        return None
    required_counts = ("dataset_rows", "training_rows", "test_rows")
    if any(
        not isinstance(summary.get(key), int) or isinstance(summary.get(key), bool)
        for key in required_counts
    ):
        issues.append("training_summary.json has missing or invalid record counts.")
        return None
    if any(summary[key] < 0 for key in required_counts):
        issues.append("training_summary.json contains negative record counts.")
        return None
    return summary


def _read_model_metadata(issues: list[str]) -> dict[str, Any] | None:
    if not MODEL_PATH.is_file():
        issues.append(
            f"Trained model is missing: {MODEL_PATH.relative_to(PROJECT_ROOT)}"
        )
        return None
    try:
        metadata = _read_model_metadata_cached(
            str(MODEL_PATH), MODEL_PATH.stat().st_mtime_ns
        )
        if not isinstance(metadata.get("feature_columns"), list) or any(
            not isinstance(feature, str) for feature in metadata["feature_columns"]
        ):
            raise ValueError("The model artifact has invalid feature-column metadata.")
        return metadata
    except (
        OSError,
        EOFError,
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
        ModuleNotFoundError,
        ImportError,
        pickle.UnpicklingError,
    ) as exc:
        issues.append(f"Could not inspect the saved model artifact: {exc}")
        return None


def load_dashboard_data() -> DashboardData:
    """Load only saved outputs; this function never fits or predicts a model."""
    data = DashboardData()
    data.metrics = _validate_metrics(
        _read_json(
            EVALUATION_DIR / "metrics.json", "Evaluation metrics", data.issues
        ),
        data.issues,
    )

    data.classification_report = _read_json(
        EVALUATION_DIR / "classification_report.json",
        "Classification report",
        data.issues,
    )
    data.training_summary = _validate_training_summary(
        _read_json(
            EVALUATION_DIR / "training_summary.json",
            "Training summary",
            data.issues,
        ),
        data.issues,
    )
    if data.training_summary is not None:
        summary = data.training_summary
        counts_match = (
            summary["dataset_rows"]
            == summary["training_rows"] + summary["test_rows"]
        )
        test_count_matches = (
            data.metrics is None
            or summary["test_rows"] == data.metrics["test_rows"]
        )
        if not counts_match or not test_count_matches:
            data.issues.append(
                "training_summary.json record counts do not match the saved split."
            )
            data.training_summary = None
    data.profile = _validate_profile(
        _read_json(
            EVALUATION_DIR / "dataset_profile.json",
            "Dataset profile",
            data.issues,
        ),
        data.issues,
    )
    data.profile_markdown = _read_markdown(
        EVALUATION_DIR / "dataset_profile.md",
        "Dataset profile report",
        data.issues,
    )
    data.model_metadata = _read_model_metadata(data.issues)

    predictions = _read_csv(
        EVALUATION_DIR / "test_predictions.csv",
        "Test predictions",
        {"Timestamp", "actual", "predicted", "baseline_predicted", "error"},
        data.issues,
    )
    expected_rows = None
    if data.metrics is not None:
        expected_rows = data.metrics.get("test_rows")
    data.predictions = _validate_predictions(
        predictions, expected_rows, data.issues
    )
    if data.predictions is not None and data.metrics is not None:
        actual_counts = {
            str(label): int((data.predictions["actual"] == label).sum())
            for label in (0, 1)
        }
        saved_counts = {
            str(label): int(count)
            for label, count in data.metrics["class_counts"].items()
        }
        if actual_counts != saved_counts:
            data.issues.append(
                "Prediction class counts do not match metrics.json "
                f"({actual_counts} vs {saved_counts})."
            )
        matrices = data.metrics["confusion_matrices"]
        if all(
            name in matrices
            for name in ("Most frequent baseline", "Decision tree")
        ):
            tree_matrix = matrices["Decision tree"]
            errors_from_matrix = tree_matrix[0][1] + tree_matrix[1][0]
            if errors_from_matrix != int(data.predictions["error"].sum()):
                data.issues.append(
                    "Decision-tree confusion matrix does not match the saved "
                    "prediction error count."
                )

    data.error_by_hour = _validate_error_by_hour(
        _read_csv(
            EVALUATION_DIR / "error_analysis_by_hour.csv",
            "Hourly error analysis",
            {"hour", "observations", "errors", "error_rate"},
            data.issues,
        ),
        data.issues,
    )
    if (
        data.predictions is not None
        and data.error_by_hour is not None
        and (
            int(data.error_by_hour["errors"].sum())
            != int(data.predictions["error"].sum())
            or int(data.error_by_hour["observations"].sum())
            != len(data.predictions)
        )
    ):
        data.issues.append(
            "Hourly error counts do not match the saved prediction error count."
        )
    data.feature_importance = _read_csv(
        EVALUATION_DIR / "feature_importance.csv",
        "Feature importance",
        {"feature", "importance"},
        data.issues,
    )
    if data.feature_importance is not None:
        data.feature_importance["importance"] = pd.to_numeric(
            data.feature_importance["importance"], errors="coerce"
        )
        if data.feature_importance["importance"].isna().any():
            data.issues.append(
                "feature_importance.csv contains invalid importance values."
            )
            data.feature_importance = None
        elif (
            not data.feature_importance["feature"].map(
                lambda feature: isinstance(feature, str) and bool(feature.strip())
            ).all()
            or not data.feature_importance["importance"].between(0, 1).all()
        ):
            data.issues.append(
                "feature_importance.csv contains invalid feature names or values."
            )
            data.feature_importance = None

    data.figures = {
        title: FIGURES_DIR / filename for title, filename in FIGURE_FILES.items()
    }
    return data
