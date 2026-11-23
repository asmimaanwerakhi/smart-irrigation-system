"""Readable, reproducible figures for exploratory and model evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

from src.preprocessing import FEATURE_COLUMNS, TARGET_COLUMN

sns.set_theme(style="whitegrid", context="notebook")


def _save(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def create_exploratory_figures(
    dataframe: pd.DataFrame,
    figures_directory: Path,
) -> None:
    """Create sensor distributions, missingness, correlation, and time profiles."""
    figures_directory = Path(figures_directory)

    figure, axes = plt.subplots(3, 3, figsize=(15, 11))
    for axis, column in zip(axes.flat, FEATURE_COLUMNS):
        values = dataframe[column].dropna()
        lower, upper = values.quantile([0.01, 0.99])
        sns.histplot(
            values.clip(lower, upper),
            bins=35,
            kde=True,
            ax=axis,
            color="#2878B5",
        )
        axis.set_title(column)
        axis.set_xlabel("Reading (display clipped to 1st–99th percentiles)")
        axis.set_ylabel("Records")
    for axis in axes.flat[len(FEATURE_COLUMNS):]:
        axis.set_visible(False)
    figure.suptitle("Sensor reading distributions", fontsize=16)
    _save(figure, figures_directory / "sensor_distributions.png")

    figure, axes = plt.subplots(
        1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1, 2]}
    )
    missing_percent = dataframe.isna().mean().mul(100).sort_values(ascending=False)
    sns.barplot(
        x=missing_percent.values,
        y=missing_percent.index,
        ax=axes[0],
        color="#D97941",
    )
    axes[0].set_title("Missing values by column")
    axes[0].set_xlabel("Missing records (%)")
    axes[0].set_ylabel("Column")
    sample = dataframe.iloc[::120].isna().T
    sns.heatmap(
        sample,
        cbar=False,
        cmap=sns.color_palette(["#4C9F70", "#D97941"], as_cmap=True),
        ax=axes[1],
    )
    axes[1].set_title("Missingness pattern (one row sampled every 120 minutes)")
    axes[1].set_xlabel("Sampled observation")
    axes[1].set_ylabel("Column")
    _save(figure, figures_directory / "missing_values.png")

    numeric = dataframe[FEATURE_COLUMNS + ["Keladi", "Jasmine", TARGET_COLUMN]].corr()
    figure, axis = plt.subplots(figsize=(11, 8))
    sns.heatmap(
        numeric,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        center=0,
        square=True,
        ax=axis,
    )
    axis.set_title("Sensor and label correlation (timestamp excluded)")
    _save(figure, figures_directory / "correlation_heatmap.png")

    start = dataframe["Timestamp"].min()
    moisture_window = dataframe.loc[
        dataframe["Timestamp"] < start + pd.Timedelta(days=3),
        ["Timestamp", "Soil Moisture 1", "Soil Moisture 2"],
    ]
    figure, axis = plt.subplots(figsize=(14, 5))
    for column in ("Soil Moisture 1", "Soil Moisture 2"):
        axis.plot(
            moisture_window["Timestamp"],
            moisture_window[column],
            label=column,
            linewidth=1,
        )
    axis.set_title("Soil moisture readings during the first three recorded days")
    axis.set_xlabel("Timestamp")
    axis.set_ylabel("Soil moisture reading (%)")
    axis.legend()
    _save(figure, figures_directory / "soil_moisture_timeline.png")

    hourly = dataframe.assign(hour=dataframe["Timestamp"].dt.hour).groupby("hour")[
        ["Env Temp", "Humidity", "LDR"]
    ].mean()
    figure, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    for axis, column, color, units in zip(
        axes,
        ["Env Temp", "Humidity", "LDR"],
        ["#C44E52", "#4C72B0", "#55A868"],
        ["Temperature (°C)", "Humidity (%)", "LDR reading"],
    ):
        axis.plot(hourly.index, hourly[column], marker="o", color=color)
        axis.set_title(f"Mean {column} by hour of day")
        axis.set_ylabel(units)
    axes[-1].set_xlabel("Hour of day")
    figure.suptitle("Environmental sensor patterns by collection hour", fontsize=16)
    _save(figure, figures_directory / "environmental_hourly_profiles.png")

    target_counts = dataframe[TARGET_COLUMN].value_counts().sort_index()
    figure, axis = plt.subplots(figsize=(7, 5))
    sns.barplot(
        x=target_counts.index.astype(str),
        y=target_counts.values,
        ax=axis,
        color="#2878B5",
    )
    axis.set_title("Best Time target distribution")
    axis.set_xlabel("Best Time (0 = pause; 1 = allowed schedule window)")
    axis.set_ylabel("Records")
    for index, count in enumerate(target_counts.values):
        axis.text(index, count, f"{count:,}", ha="center", va="bottom")
    _save(figure, figures_directory / "target_distribution.png")


def create_evaluation_figures(
    test_rows: pd.DataFrame,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    models: dict[str, Any],
    metrics: dict[str, dict[str, float]],
    figures_directory: Path,
) -> None:
    """Create test-set confusion, comparison, and error-analysis charts."""
    figures_directory = Path(figures_directory)
    predictions = models["Decision tree"].predict(x_test)

    figure, axis = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        labels=[0, 1],
        display_labels=["Pause (0)", "Allowed (1)"],
        cmap="Blues",
        values_format="d",
        ax=axis,
    )
    axis.set_title("Decision tree confusion matrix (chronological holdout)")
    _save(figure, figures_directory / "confusion_matrix.png")

    metric_names = ["accuracy", "precision", "recall", "f1", "macro_f1"]
    comparison = pd.DataFrame(metrics).reindex(metric_names)
    figure, axis = plt.subplots(figsize=(10, 6))
    comparison.plot(kind="bar", ax=axis, color=["#9A9A9A", "#2878B5"])
    axis.set_title("Baseline versus decision tree on the chronological holdout")
    axis.set_xlabel("Metric (precision, recall, and F1 use class 1 as positive)")
    axis.set_ylabel("Score")
    axis.set_ylim(0, 1)
    axis.tick_params(axis="x", rotation=0)
    axis.legend(title="Model")
    _save(figure, figures_directory / "model_performance_comparison.png")

    test_predictions = test_rows[["Timestamp"]].copy()
    test_predictions["actual"] = y_test.to_numpy()
    test_predictions["predicted"] = predictions
    test_predictions["error"] = (
        test_predictions["actual"] != test_predictions["predicted"]
    )
    display_end = test_predictions["Timestamp"].min() + pd.Timedelta(days=2)
    timeline = test_predictions.loc[test_predictions["Timestamp"] < display_end]
    figure, axis = plt.subplots(figsize=(14, 5))
    axis.step(
        timeline["Timestamp"],
        timeline["actual"],
        where="post",
        label="Actual",
        alpha=0.8,
    )
    axis.step(
        timeline["Timestamp"],
        timeline["predicted"],
        where="post",
        label="Predicted",
        alpha=0.7,
    )
    axis.set_title("Actual and predicted Best Time labels (first two holdout days)")
    axis.set_xlabel("Timestamp")
    axis.set_ylabel("Class (0 = pause, 1 = allowed schedule window)")
    axis.set_yticks([0, 1])
    axis.legend()
    _save(figure, figures_directory / "test_predictions_timeline.png")

    hourly = (
        test_predictions.assign(hour=test_predictions["Timestamp"].dt.hour)
        .groupby("hour", as_index=False)
        .agg(
            observations=("error", "size"),
            errors=("error", "sum"),
            error_rate=("error", "mean"),
        )
    )
    figure, axis = plt.subplots(figsize=(10, 5))
    sns.barplot(data=hourly, x="hour", y="error_rate", color="#C44E52", ax=axis)
    axis.set_title("Decision tree error rate by hour in the holdout")
    axis.set_xlabel("Hour of day")
    axis.set_ylabel("Misclassification rate")
    axis.set_ylim(0, min(1.0, max(0.1, float(hourly["error_rate"].max()) * 1.15)))
    _save(figure, figures_directory / "error_rate_by_hour.png")
