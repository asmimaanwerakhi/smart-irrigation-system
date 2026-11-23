"""Streamlit entry point for the saved Smart Irrigation ML results dashboard."""

from __future__ import annotations

import streamlit as st

from src.dashboard_data import load_dashboard_data
from src.dashboard_pages import (
    render_error_analysis,
    render_methodology,
    render_overview,
    render_performance,
    render_predictions,
    render_sensor_analysis,
)

PAGES = {
    "Overview": render_overview,
    "Model Performance": render_performance,
    "Prediction Analysis": render_predictions,
    "Sensor & Feature Analysis": render_sensor_analysis,
    "Error Analysis": render_error_analysis,
    "Dataset & Methodology": render_methodology,
}


def main() -> None:
    """Render a page using existing reports; never retrain or predict."""
    st.set_page_config(
        page_title="Smart Irrigation - ML Results",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1480px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }
        [data-testid="stSidebar"] {
            border-right: 1px solid rgba(148, 163, 184, 0.18);
        }
        [data-testid="stMetric"] {
            background: #121E2E;
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 12px;
            padding: 14px 16px;
        }
        h1, h2, h3 {
            letter-spacing: -0.02em;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.title("Irrigation Lab")
    st.sidebar.caption("IoT 429 | Saved ML evaluation")
    selected_page = st.sidebar.radio("Dashboard pages", list(PAGES))
    st.sidebar.divider()
    st.sidebar.caption(
        "Artifacts are read from this project; model training is not run."
    )

    data = load_dashboard_data()
    if data.issues:
        with st.sidebar.expander(
            f"Artifact notices ({len(data.issues)})",
            expanded=True,
        ):
            for issue in data.issues:
                st.warning(issue)

    st.warning(
        "**Academic limitation:** the workbook is semi-synthetic and `Best Time` "
        "is a collection-time schedule-window label, not actual plant water need. "
        "The model may use LDR as a time-of-day proxy. This dashboard does not "
        "control a pump or provide validated agronomic recommendations."
    )

    PAGES[selected_page](data)


if __name__ == "__main__":
    main()
