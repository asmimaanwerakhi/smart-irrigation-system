"""Page renderers for the saved-results Streamlit dashboard."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
import streamlit as st

from src.dashboard_components import (
    class_distribution_chart,
    comparison_chart,
    confusion_matrix_chart,
    correctness_chart,
    feature_importance_chart,
    hourly_error_chart,
    prediction_class_chart,
    render_metric_row,
    render_saved_figure,
)
from src.dashboard_data import DashboardData
from src.preprocessing import FEATURE_COLUMNS


def _show_artifact_error(label: str) -> None:
    st.error(f"{label} could not be loaded. Check the saved files under `outputs/`.")


def _score_label(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "N/A"


def _model_display_name(metadata: dict[str, Any] | None) -> str:
    if metadata is None:
        return "Saved model unavailable"
    name = str(metadata.get("estimator_name", "Classifier"))
    if name == "DecisionTreeClassifier":
        return "Decision Tree Classifier"
    return name


def _metrics_for(data: DashboardData) -> dict[str, dict[str, Any]] | None:
    if data.metrics is None:
        _show_artifact_error("Evaluation metrics")
        return None
    result = data.metrics.get("metrics")
    if not isinstance(result, dict):
        _show_artifact_error("Evaluation metrics")
        return None
    return result


def render_overview(data: DashboardData) -> None:
    """Render the project overview and saved headline metrics."""
    st.title("Smart Irrigation System")
    st.caption(
        "IoT 429 | Machine-learning results dashboard | Saved chronological holdout"
    )
    st.write(
        "Explore the saved evaluation for a sensor-based classifier of the "
        "dataset's documented `Best Time` schedule-window label."
    )

    metrics = _metrics_for(data)
    if metrics is None:
        return
    tree = metrics.get("Decision tree", {})
    baseline = metrics.get("Most frequent baseline", {})
    summary = data.training_summary or {}
    profile = data.profile or {}
    shape = profile.get("shape", {})
    dataset_rows = summary.get("dataset_rows", shape.get("rows"))
    training_rows = summary.get("training_rows")
    test_rows = data.metrics.get("test_rows")
    features = (
        data.model_metadata.get("feature_columns")
        if data.model_metadata
        else FEATURE_COLUMNS
    )
    model_name = _model_display_name(data.model_metadata)

    render_metric_row(
        [
            (
                "Dataset records",
                f"{dataset_rows:,}" if dataset_rows is not None else "N/A",
                None,
            ),
            ("Sensor features", f"{len(features):,}", None),
            (
                "Train / test records",
                (
                    f"{training_rows:,} / {test_rows:,}"
                    if training_rows is not None and test_rows is not None
                    else "N/A"
                ),
                "Chronological split; no shuffling.",
            ),
            (
                "Model",
                model_name,
                "Loaded from the saved model artifact; not retrained.",
            ),
        ]
    )
    render_metric_row(
        [
            ("Accuracy", _score_label(tree.get("accuracy")), None),
            ("Macro-F1", _score_label(tree.get("macro_f1")), None),
            ("Precision - class 1", _score_label(tree.get("precision")), None),
            ("Recall - class 1", _score_label(tree.get("recall")), None),
        ]
    )

    if data.metrics:
        st.caption(
            "Test period: "
            f"{data.metrics.get('test_start', 'N/A')} to "
            f"{data.metrics.get('test_end', 'N/A')}"
        )

    left, right = st.columns([1.2, 1])
    with left:
        if baseline and tree:
            st.plotly_chart(
                comparison_chart(
                    {
                        "Most frequent baseline": baseline,
                        "Decision tree": tree,
                    }
                ),
                use_container_width=True,
                key="overview-model-comparison",
            )
        else:
            st.info("Saved baseline/model comparison metrics are unavailable.")
    with right:
        class_counts = data.metrics.get("class_counts", {})
        if class_counts:
            st.plotly_chart(
                class_distribution_chart(class_counts),
                use_container_width=True,
                key="overview-class-distribution",
            )
        else:
            st.info("Saved class counts are unavailable.")

    st.info(
        "**What the model predicts:** class 0 is the documented 10:00-15:59 "
        "irrigation-pause window; class 1 is the other schedule windows. "
        "This is not a prediction of plant water need or pump status."
    )


def render_performance(data: DashboardData) -> None:
    """Render saved classification metrics, report, and confusion matrices."""
    st.title("Model Performance")
    metrics = _metrics_for(data)
    if metrics is None:
        return
    tree = metrics.get("Decision tree")
    if not isinstance(tree, dict):
        _show_artifact_error("Decision-tree metrics")
        return

    render_metric_row(
        [
            ("Accuracy", _score_label(tree.get("accuracy")), None),
            ("Precision - class 1", _score_label(tree.get("precision")), None),
            ("Recall - class 1", _score_label(tree.get("recall")), None),
            ("F1 - class 1", _score_label(tree.get("f1")), None),
            ("Macro-F1", _score_label(tree.get("macro_f1")), None),
        ]
    )

    st.subheader("Classification report")
    report = data.classification_report
    if not isinstance(report, dict):
        _show_artifact_error("Classification report")
    else:
        report_rows = []
        for model_name, model_report in report.items():
            if not isinstance(model_report, dict):
                continue
            for class_key, class_name in (
                ("0", "Class 0 - irrigation pause"),
                ("1", "Class 1 - other windows"),
                ("macro avg", "Macro average"),
                ("weighted avg", "Weighted average"),
            ):
                values = model_report.get(class_key)
                if isinstance(values, dict):
                    report_rows.append(
                        {
                            "Model": model_name,
                            "Class": class_name,
                            "Precision": values.get("precision"),
                            "Recall": values.get("recall"),
                            "F1-score": values.get("f1-score"),
                            "Support": values.get("support"),
                        }
                    )
        if report_rows:
            st.dataframe(
                pd.DataFrame(report_rows).round(4),
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.info("The saved classification report has no displayable class rows.")

    left, right = st.columns([1, 1.15])
    with left:
        if "Most frequent baseline" in metrics and "Decision tree" in metrics:
            st.plotly_chart(
                comparison_chart(
                    {
                        "Most frequent baseline": metrics["Most frequent baseline"],
                        "Decision tree": metrics["Decision tree"],
                    }
                ),
                use_container_width=True,
                key="performance-comparison",
            )
        else:
            st.info("Saved model-comparison metrics are unavailable.")
    with right:
        matrices = data.metrics.get("confusion_matrices", {})
        model_options = [name for name in metrics if name in matrices]
        if model_options:
            selected = st.selectbox(
                "Confusion matrix model",
                model_options,
                index=model_options.index("Decision tree")
                if "Decision tree" in model_options
                else 0,
            )
            st.plotly_chart(
                confusion_matrix_chart(matrices[selected]),
                use_container_width=True,
                key=f"confusion-{selected}",
            )
        else:
            st.info("Saved confusion matrices are unavailable.")

    st.caption(
        "Class 0 = irrigation pause (10:00-15:59). "
        "Class 1 = other schedule windows. Class 1 is treated as positive "
        "for precision, recall, and F1."
    )


def render_predictions(data: DashboardData) -> None:
    """Explore the saved test predictions with filters and pagination."""
    st.title("Prediction Analysis")
    predictions = data.predictions
    if predictions is None:
        _show_artifact_error("Test predictions")
        return

    error_count = int(predictions["error"].sum())
    correct_count = len(predictions) - error_count
    error_rate = error_count / len(predictions) if len(predictions) else 0.0
    render_metric_row(
        [
            ("Holdout predictions", f"{len(predictions):,}", None),
            ("Correct", f"{correct_count:,}", None),
            ("Incorrect", f"{error_count:,}", None),
            ("Prediction error", f"{error_rate:.2%}", None),
        ]
    )

    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            prediction_class_chart(predictions),
            use_container_width=True,
            key="prediction-class-counts",
        )
    with right:
        st.plotly_chart(
            correctness_chart(predictions),
            use_container_width=True,
            key="prediction-correctness",
        )

    st.subheader("Filter saved test predictions")
    filter_columns = st.columns(3)
    actual_classes = filter_columns[0].multiselect(
        "Actual class",
        options=[0, 1],
        default=[0, 1],
        format_func=lambda value: (
            "0 - irrigation pause" if value == 0 else "1 - other windows"
        ),
    )
    predicted_classes = filter_columns[1].multiselect(
        "Predicted class",
        options=[0, 1],
        default=[0, 1],
        format_func=lambda value: (
            "0 - irrigation pause" if value == 0 else "1 - other windows"
        ),
    )
    correctness = filter_columns[2].selectbox(
        "Correctness",
        ["All predictions", "Correct only", "Incorrect only"],
    )
    search = st.text_input("Search timestamp", placeholder="e.g. 2026-08-01")

    filtered = predictions.loc[
        predictions["actual"].isin(actual_classes)
        & predictions["predicted"].isin(predicted_classes)
    ].copy()
    if correctness == "Correct only":
        filtered = filtered.loc[~filtered["error"]]
    elif correctness == "Incorrect only":
        filtered = filtered.loc[filtered["error"]]
    if search.strip():
        filtered = filtered.loc[
            filtered["Timestamp"]
            .astype(str)
            .str.contains(search.strip(), case=False, regex=False, na=False)
        ]

    page_size = st.selectbox("Rows per page", [25, 50, 100, 250], index=1)
    page_count = max(1, math.ceil(len(filtered) / page_size))
    page_number = st.number_input(
        "Page",
        min_value=1,
        max_value=page_count,
        value=1,
        step=1,
    )
    start = (int(page_number) - 1) * page_size
    page = filtered.iloc[start : start + page_size].copy()
    page["correct"] = ~page["error"]
    st.caption(
        f"{len(filtered):,} matching rows | page {page_number} of {page_count}"
    )
    if page.empty:
        st.info("No predictions match the selected filters.")
    else:
        st.dataframe(
            page[
                [
                    "Timestamp",
                    "actual",
                    "predicted",
                    "baseline_predicted",
                    "correct",
                ]
            ],
            hide_index=True,
            use_container_width=True,
            column_config={
                "actual": st.column_config.NumberColumn("Actual class", format="%d"),
                "predicted": st.column_config.NumberColumn(
                    "Predicted class", format="%d"
                ),
                "baseline_predicted": st.column_config.NumberColumn(
                    "Baseline class", format="%d"
                ),
                "correct": st.column_config.CheckboxColumn("Correct"),
            },
        )


def _sensor_statistics(profile: dict[str, Any]) -> pd.DataFrame:
    """Extract the saved profile's statistics for the seven model features."""
    columns = profile.get("columns", {})
    rows = []
    for feature in FEATURE_COLUMNS:
        detail = columns.get(feature, {})
        stats = detail.get("statistics", {})
        rows.append(
            {
                "Sensor feature": feature,
                "Missing": detail.get("missing_count"),
                "Missing %": detail.get("missing_percent"),
                "Mean": stats.get("mean"),
                "Std. dev.": stats.get("std"),
                "Min": stats.get("min"),
                "Median": stats.get("50%"),
                "Max": stats.get("max"),
            }
        )
    return pd.DataFrame(rows)


def render_sensor_analysis(data: DashboardData) -> None:
    """Show saved feature importance, profile statistics, and sensor figures."""
    st.title("Sensor & Feature Analysis")

    if data.feature_importance is None:
        _show_artifact_error("Feature importance")
    else:
        left, right = st.columns([1.1, 1])
        with left:
            st.plotly_chart(
                feature_importance_chart(data.feature_importance),
                use_container_width=True,
                key="feature-importance",
            )
        with right:
            st.markdown(
                "Feature importance describes how the saved decision tree used "
                "each feature for its splits. It indicates model association, "
                "not causation or agronomic importance."
            )
            st.dataframe(
                data.feature_importance.assign(
                    importance_percent=data.feature_importance["importance"] * 100
                ).rename(columns={"importance": "importance_fraction"}),
                hide_index=True,
                use_container_width=True,
            )

    profile = data.profile
    if not isinstance(profile, dict) or not isinstance(profile.get("columns"), dict):
        _show_artifact_error("Dataset profile")
    else:
        st.subheader("Sensor statistics from the saved dataset profile")
        st.dataframe(
            _sensor_statistics(profile).round(3),
            hide_index=True,
            use_container_width=True,
        )

    st.subheader("Saved sensor visualizations")
    figure_rows = [
        ("Sensor distributions", "Missing values"),
        ("Correlation heatmap", "Soil moisture timeline"),
        ("Environmental hourly profiles", "Target distribution"),
    ]
    for first_title, second_title in figure_rows:
        left, right = st.columns(2)
        with left:
            render_saved_figure(first_title, data.figures[first_title])
        with right:
            render_saved_figure(second_title, data.figures[second_title])


def render_error_analysis(data: DashboardData) -> None:
    """Render the persisted hourly holdout error analysis."""
    st.title("Error Analysis")
    hourly = data.error_by_hour
    if hourly is None:
        _show_artifact_error("Hourly error analysis")
        return

    total_observations = int(hourly["observations"].sum())
    total_errors = int(hourly["errors"].sum())
    overall_rate = total_errors / total_observations if total_observations else 0.0
    peak_row = hourly.loc[hourly["error_rate"].idxmax()]
    figure, top_quartile_cut = hourly_error_chart(hourly)
    render_metric_row(
        [
            ("Holdout observations", f"{total_observations:,}", None),
            ("Errors", f"{total_errors:,}", None),
            ("Overall error rate", f"{overall_rate:.2%}", None),
            (
                "Highest-error hour",
                f"{int(peak_row['hour']):02d}:00 - {peak_row['error_rate']:.1%}",
                "Highest rate among the saved hourly groups.",
            ),
        ]
    )
    st.caption(
        "Red bars mark the top quartile of the observed hourly error rates "
        f"(rate >= {top_quartile_cut:.1%}); this is a visual highlight only."
    )
    st.plotly_chart(
        figure,
        use_container_width=True,
        key="hourly-error-analysis",
    )
    st.dataframe(
        hourly.assign(error_rate_percent=hourly["error_rate"] * 100).round(3),
        hide_index=True,
        use_container_width=True,
    )
    st.info(
        "Hourly error rates describe where mistakes occurred in this holdout. "
        "They do not establish why they occurred or prove an agronomic cause."
    )


def render_methodology(data: DashboardData) -> None:
    """Show dataset facts, preprocessing, target semantics, and limitations."""
    st.title("Dataset & Methodology")
    profile = data.profile
    if isinstance(profile, dict):
        shape = profile.get("shape", {})
        timestamp = profile.get("timestamp", {})
        rows = shape.get("rows")
        cols = shape.get("columns")
        render_metric_row(
            [
                ("Records", f"{rows:,}" if rows is not None else "N/A", None),
                ("Columns", f"{cols:,}" if cols is not None else "N/A", None),
                (
                    "Rows with missing sensor values",
                    f"{profile.get('sensor_rows_with_any_missing', 0):,}",
                    None,
                ),
                (
                    "Duplicate rows",
                    f"{profile.get('duplicate_full_rows', 0):,}",
                    None,
                ),
            ]
        )
        st.caption(
            f"Workbook: {profile.get('dataset_file', 'N/A')} | "
            f"Time range: {timestamp.get('minimum', 'N/A')} to "
            f"{timestamp.get('maximum', 'N/A')} | "
            f"Cadence: {timestamp.get('most_common_interval', 'N/A')}"
        )
    else:
        _show_artifact_error("Dataset profile")

    if isinstance(profile, dict) and isinstance(profile.get("columns"), dict):
        st.subheader("Missing values by column")
        missing_rows = [
            {
                "Column": name,
                "Missing records": details.get("missing_count"),
                "Missing (%)": details.get("missing_percent"),
            }
            for name, details in profile["columns"].items()
        ]
        st.dataframe(
            pd.DataFrame(missing_rows).round(3),
            hide_index=True,
            use_container_width=True,
        )

    st.subheader("Sensor features")
    descriptions = {
        "Soil Moisture 1": "Soil-moisture sensor channel 1.",
        "Soil Moisture 2": "Soil-moisture sensor channel 2.",
        "Soil Temp1": "Soil temperature sensor channel 1.",
        "Soil Temp2": "Soil temperature sensor channel 2.",
        "Env Temp": "Environmental temperature reading.",
        "Humidity": "Environmental humidity reading.",
        "LDR": "Light-dependent-resistor sensor reading.",
    }
    st.dataframe(
        pd.DataFrame(
            [
                {"Feature": name, "Description": descriptions[name]}
                for name in FEATURE_COLUMNS
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.subheader("Target labels and academic limitations")
    st.markdown(
        "- **Class 0:** 10:00-15:59; the workbook documents an "
        "irrigation-pause window.\n"
        "- **Class 1:** other schedule windows; this includes periods that are "
        "allowed but not necessarily described as the best time to water.\n"
        "- The target is based on collection time. It is not ground-truth plant "
        "water need, irrigation effectiveness, or pump status.\n"
        "- The workbook describes the data as **semi-synthetic**: soil moisture "
        "and irrigation targets are simulated, and data-quality issues were injected.\n"
        "- LDR can act as a time-of-day proxy. Feature importance is association, "
        "not causation.\n"
        "- This project does not control hardware and is not a production-ready "
        "autonomous irrigation controller."
    )

    if data.model_metadata:
        st.subheader("Saved model and preprocessing")
        metadata = data.model_metadata
        st.markdown(
            f"- Estimator: **{_model_display_name(metadata)}**.\n"
            f"- Maximum tree depth: **{metadata.get('max_depth', 'N/A')}**; "
            f"minimum leaf size: **{metadata.get('min_samples_leaf', 'N/A')}**; "
            f"random state: **{metadata.get('random_state', 'N/A')}**.\n"
            "- Missing sensor values and documented reset/failure codes are "
            "handled with median imputation inside the saved scikit-learn pipeline; "
            "the imputer was fitted on training data only.\n"
            "- The chronological 80/20 split uses the first 80% for training "
            "and the final 20% for testing, without shuffling. Timestamp is "
            "excluded from model inputs."
        )
    else:
        _show_artifact_error("Saved model configuration")

    if data.profile_markdown:
        with st.expander("Full saved dataset inspection report"):
            st.markdown(data.profile_markdown)
