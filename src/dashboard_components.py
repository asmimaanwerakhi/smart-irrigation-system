"""Reusable Plotly charts and small Streamlit layout helpers."""

from __future__ import annotations

from typing import Sequence

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

NAVY = "#0B1220"
PANEL = "#121E2E"
GREEN = "#39D98A"
MINT = "#9BE7C4"
BLUE = "#60A5FA"
AMBER = "#F5B942"
RED = "#F87171"
TEXT = "#E5EDF5"
GRID = "rgba(148, 163, 184, 0.18)"


def render_metric_row(items: Sequence[tuple[str, str, str | None]]) -> None:
    """Render a row of consistent metric cards."""
    for column, (label, value, help_text) in zip(st.columns(len(items)), items):
        column.metric(label, value, help=help_text)


def _apply_dark_layout(figure: go.Figure, title: str | None = None) -> go.Figure:
    figure.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        font={"color": TEXT, "family": "Inter, Segoe UI, sans-serif"},
        margin={"l": 24, "r": 24, "t": 58, "b": 24},
        legend={"bgcolor": "rgba(0,0,0,0)"},
    )
    figure.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    figure.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return figure


def comparison_chart(metrics: dict) -> go.Figure:
    """Build the baseline/decision-tree comparison from saved metric values."""
    metric_labels = {
        "accuracy": "Accuracy",
        "precision": "Precision (class 1)",
        "recall": "Recall (class 1)",
        "f1": "F1 (class 1)",
        "macro_f1": "Macro-F1",
    }
    figure = go.Figure()
    colors = {"Most frequent baseline": BLUE, "Decision tree": GREEN}
    for model_name, scores in metrics.items():
        figure.add_trace(
            go.Bar(
                name=model_name,
                x=[metric_labels[key] for key in metric_labels],
                y=[scores.get(key) for key in metric_labels],
                marker_color=colors.get(model_name, MINT),
                hovertemplate="%{x}<br>%{y:.3f}<extra>" + model_name + "</extra>",
            )
        )
    figure.update_layout(
        barmode="group",
        yaxis={"title": "Score", "range": [0, 1]},
        xaxis_title="Saved evaluation metric",
    )
    return _apply_dark_layout(figure, "Baseline vs Decision Tree")


def class_distribution_chart(class_counts: dict) -> go.Figure:
    """Plot saved actual class counts."""
    names = {"0": "Class 0 - pause", "1": "Class 1 - other windows"}
    labels = [names.get(str(label), f"Class {label}") for label in class_counts]
    values = [class_counts[label] for label in class_counts]
    colors = [AMBER if str(label) == "0" else GREEN for label in class_counts]
    figure = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker_color=colors,
            text=[f"{value:,}" for value in values],
            textposition="outside",
            hovertemplate="%{x}<br>%{y:,} records<extra></extra>",
        )
    )
    figure.update_layout(yaxis_title="Records", showlegend=False)
    return _apply_dark_layout(figure, "Actual holdout class distribution")


def confusion_matrix_chart(matrix: list[list[int]]) -> go.Figure:
    """Build an interactive raw-count confusion matrix."""
    labels = ["0 - Irrigation pause", "1 - Other schedule windows"]
    figure = go.Figure(
        go.Heatmap(
            z=matrix,
            x=labels,
            y=labels,
            colorscale=[[0, NAVY], [0.55, "#176B58"], [1, GREEN]],
            text=matrix,
            texttemplate="%{text:,}",
            textfont={"size": 18, "color": TEXT},
            hovertemplate=(
                "Actual: %{y}<br>Predicted: %{x}<br>Records: %{z:,}<extra></extra>"
            ),
            showscale=True,
            colorbar={"title": "Records"},
        )
    )
    figure.update_layout(
        xaxis_title="Predicted class",
        yaxis_title="Actual class",
        yaxis={"autorange": "reversed"},
    )
    return _apply_dark_layout(figure, "Confusion matrix - saved holdout results")


def prediction_class_chart(predictions: pd.DataFrame) -> go.Figure:
    """Compare actual and saved predicted class counts."""
    actual_counts = predictions["actual"].value_counts().reindex([0, 1], fill_value=0)
    predicted_counts = (
        predictions["predicted"].value_counts().reindex([0, 1], fill_value=0)
    )
    labels = ["Class 0 - pause", "Class 1 - other windows"]
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            name="Actual",
            x=labels,
            y=actual_counts.tolist(),
            marker_color=BLUE,
            hovertemplate="%{x}<br>%{y:,}<extra>Actual</extra>",
        )
    )
    figure.add_trace(
        go.Bar(
            name="Predicted",
            x=labels,
            y=predicted_counts.tolist(),
            marker_color=GREEN,
            hovertemplate="%{x}<br>%{y:,}<extra>Predicted</extra>",
        )
    )
    figure.update_layout(barmode="group", yaxis_title="Records")
    return _apply_dark_layout(figure, "Actual vs predicted class distribution")


def correctness_chart(predictions: pd.DataFrame) -> go.Figure:
    """Show correct and incorrect saved holdout predictions."""
    error_count = int(predictions["error"].sum())
    correct_count = len(predictions) - error_count
    figure = go.Figure(
        go.Pie(
            labels=["Correct", "Incorrect"],
            values=[correct_count, error_count],
            hole=0.68,
            marker={"colors": [GREEN, RED]},
            textinfo="label+percent",
            hovertemplate="%{label}<br>%{value:,} records<extra></extra>",
        )
    )
    return _apply_dark_layout(figure, "Prediction correctness")


def feature_importance_chart(feature_importance: pd.DataFrame) -> go.Figure:
    """Plot saved impurity-based feature importance."""
    ordered = feature_importance.sort_values("importance", ascending=True)
    figure = go.Figure(
        go.Bar(
            x=ordered["importance"],
            y=ordered["feature"],
            orientation="h",
            marker_color=GREEN,
            hovertemplate="%{y}<br>Importance: %{x:.3f}<extra></extra>",
        )
    )
    figure.update_layout(xaxis_title="Decision-tree feature importance", yaxis_title="")
    return _apply_dark_layout(figure, "Saved decision-tree feature importance")


def hourly_error_chart(error_by_hour: pd.DataFrame) -> tuple[go.Figure, float]:
    """Plot saved hourly error rates and counts; return the top-quartile cut."""
    cutoff = float(error_by_hour["error_rate"].quantile(0.75))
    colors = [
        RED if value >= cutoff and value > 0 else GREEN
        for value in error_by_hour["error_rate"]
    ]
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_trace(
        go.Bar(
            x=error_by_hour["hour"],
            y=error_by_hour["error_rate"],
            name="Error rate",
            marker_color=colors,
            customdata=error_by_hour[["errors", "observations"]],
            hovertemplate=(
                "Hour %{x}:00<br>Error rate: %{y:.1%}<br>"
                "Errors: %{customdata[0]:,}<br>"
                "Observations: %{customdata[1]:,}<extra></extra>"
            ),
        ),
        secondary_y=False,
    )
    figure.add_trace(
        go.Scatter(
            x=error_by_hour["hour"],
            y=error_by_hour["observations"],
            name="Observations",
            mode="lines+markers",
            line={"color": BLUE, "width": 2},
            hovertemplate="Hour %{x}:00<br>Observations: %{y:,}<extra></extra>",
        ),
        secondary_y=True,
    )
    figure.update_xaxes(title_text="Collection hour", tickmode="linear", dtick=1)
    figure.update_yaxes(
        title_text="Misclassification rate",
        tickformat=".0%",
        secondary_y=False,
    )
    figure.update_yaxes(title_text="Observations", secondary_y=True, showgrid=False)
    return _apply_dark_layout(figure, "Saved holdout error rate by hour"), cutoff


def render_saved_figure(title: str, path: Any) -> None:
    """Render an existing PNG or show a clear empty state if it is unavailable."""
    if path.is_file():
        st.image(str(path), caption=title, use_container_width=True)
    else:
        st.info(f"Saved figure is unavailable: `{path.name}`")
