"""Train, evaluate, and persist the smart irrigation scheduling classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.evaluate import evaluate_classifiers, save_evaluation_outputs
from src.preprocessing import (
    DEFAULT_DATA_PATH,
    FEATURE_COLUMNS,
    PROJECT_ROOT,
    TARGET_COLUMN,
    chronological_split,
    create_dataset_profile,
    load_dataset,
    write_dataset_profile,
)
from src.visualize import create_evaluation_figures, create_exploratory_figures

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "trained_model.joblib"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "outputs"
RANDOM_STATE = 429


def build_training_pipeline(model: BaseEstimator) -> Pipeline:
    """Create a leakage-safe numeric preprocessing and estimator pipeline."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", model),
        ]
    )


def run_training(
    data_path: Path = DEFAULT_DATA_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    test_size: float = 0.2,
) -> dict[str, object]:
    """Profile data, train baseline/tree, evaluate, and write project artifacts."""
    data_path = Path(data_path)
    model_path = Path(model_path)
    output_path = Path(output_path)
    dataframe, notes = load_dataset(data_path)

    profile = create_dataset_profile(dataframe, notes, data_path)
    evaluation_directory = output_path / "evaluation"
    figures_directory = output_path / "figures"
    write_dataset_profile(profile, evaluation_directory)
    create_exploratory_figures(dataframe, figures_directory)

    x_train, x_test, y_train, y_test, test_rows = chronological_split(
        dataframe, test_size=test_size
    )
    baseline = build_training_pipeline(
        DummyClassifier(strategy="most_frequent")
    )
    decision_tree = build_training_pipeline(
        DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=100,
            random_state=RANDOM_STATE,
        )
    )

    baseline.fit(x_train, y_train)
    decision_tree.fit(x_train, y_train)
    models = {
        "Most frequent baseline": baseline,
        "Decision tree": decision_tree,
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": decision_tree,
        "baseline": baseline,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "training_rows": len(x_train),
        "test_size": test_size,
        "random_state": RANDOM_STATE,
        "description": (
            "Sensor-only classifier for the workbook's documented Best Time "
            "schedule-window label; not a soil-water-need or pump-control model."
        ),
    }
    joblib.dump(artifact, model_path)

    loaded_artifact = joblib.load(model_path)
    sample_predictions = loaded_artifact["model"].predict(x_test.head(5))
    if len(sample_predictions) != min(5, len(x_test)):
        raise RuntimeError(
            "Reloaded model did not return the expected number of predictions."
        )

    metrics, reports, _ = evaluate_classifiers(models, x_test, y_test)
    save_evaluation_outputs(
        models, x_test, y_test, test_rows, evaluation_directory
    )
    create_evaluation_figures(
        test_rows,
        x_test,
        y_test,
        models,
        metrics,
        figures_directory,
    )

    classifier = decision_tree.named_steps["classifier"]
    feature_importance = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": classifier.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    feature_importance.to_csv(
        evaluation_directory / "feature_importance.csv", index=False
    )

    summary = {
        "dataset_rows": len(dataframe),
        "training_rows": len(x_train),
        "test_rows": len(x_test),
        "test_start": test_rows["Timestamp"].min().isoformat(),
        "test_end": test_rows["Timestamp"].max().isoformat(),
        "target_counts_in_test": {
            str(label): int(count)
            for label, count in y_test.value_counts().sort_index().items()
        },
        "metrics": metrics,
        "saved_model": str(model_path),
        "outputs": str(output_path),
        "reloaded_model_sample_predictions": sample_predictions.tolist(),
        "classification_report": reports,
    }
    (evaluation_directory / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the documented Best Time classifier."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()
    print(
        json.dumps(
            run_training(args.data, args.model, args.output, args.test_size),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
