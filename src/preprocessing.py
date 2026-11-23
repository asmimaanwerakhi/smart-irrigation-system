"""Dataset loading, profiling, and leakage-aware preprocessing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "dataset" / "dataset.xlsx"
DATA_SHEET = "Data"
NOTES_SHEET = "Notes"
TIMESTAMP_COLUMN = "Timestamp"
TARGET_COLUMN = "Best Time"
FEATURE_COLUMNS = [
    "Soil Moisture 1",
    "Soil Moisture 2",
    "Soil Temp1",
    "Soil Temp2",
    "Env Temp",
    "Humidity",
    "LDR",
]
REQUIRED_COLUMNS = [
    TIMESTAMP_COLUMN,
    *FEATURE_COLUMNS,
    "Keladi",
    "Jasmine",
    TARGET_COLUMN,
]


def load_dataset(path: Path = DEFAULT_DATA_PATH) -> tuple[pd.DataFrame, dict[str, str]]:
    """Load the documented data sheet and return the workbook's notes."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset workbook not found: {path}")

    workbook = pd.ExcelFile(path)
    if DATA_SHEET not in workbook.sheet_names:
        raise ValueError(
            f"Expected a worksheet named {DATA_SHEET!r}; "
            f"found {workbook.sheet_names!r} in {path.name}."
        )

    dataframe = pd.read_excel(workbook, sheet_name=DATA_SHEET)
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(dataframe.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    dataframe[TIMESTAMP_COLUMN] = pd.to_datetime(
        dataframe[TIMESTAMP_COLUMN], errors="coerce"
    )
    if dataframe[TIMESTAMP_COLUMN].isna().any():
        bad_count = int(dataframe[TIMESTAMP_COLUMN].isna().sum())
        raise ValueError(f"{bad_count} timestamp values could not be parsed.")

    notes: dict[str, str] = {}
    if NOTES_SHEET in workbook.sheet_names:
        notes_frame = pd.read_excel(
            workbook, sheet_name=NOTES_SHEET, header=None, usecols=[0, 1]
        )
        for heading, description in notes_frame.itertuples(index=False, name=None):
            if pd.notna(heading) and pd.notna(description):
                notes[str(heading).strip()] = str(description).strip()
    return dataframe, notes


def _json_value(value: Any) -> Any:
    """Convert NumPy and pandas values to JSON-compatible values."""
    if pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def create_dataset_profile(
    dataframe: pd.DataFrame,
    notes: dict[str, str],
    workbook_path: Path,
) -> dict[str, Any]:
    """Summarize schema, distributions, data quality, and documented label meaning."""
    timestamp = dataframe[TIMESTAMP_COLUMN]
    target_values = dataframe[TARGET_COLUMN]
    sensor_columns = FEATURE_COLUMNS
    numeric_columns = dataframe.select_dtypes(include="number").columns

    column_details: dict[str, Any] = {}
    for column in dataframe.columns:
        series = dataframe[column]
        detail: dict[str, Any] = {
            "dtype": str(series.dtype),
            "missing_count": int(series.isna().sum()),
            "missing_percent": round(float(series.isna().mean() * 100), 4),
            "unique_count_including_missing": int(series.nunique(dropna=False)),
        }
        if column in numeric_columns:
            detail["statistics"] = {
                key: _json_value(value)
                for key, value in series.describe(
                    percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]
                ).items()
            }
        distinct_values = series.dropna().unique()
        if column != TIMESTAMP_COLUMN and len(distinct_values) <= 200:
            if pd.api.types.is_numeric_dtype(series):
                ordered_values = sorted(distinct_values)
            else:
                ordered_values = sorted(distinct_values, key=lambda value: str(value))
            detail["unique_values"] = [
                _json_value(value)
                for value in ordered_values
            ]
        elif column == TIMESTAMP_COLUMN:
            detail["unique_values_note"] = (
                "High-cardinality datetime; see the timestamp range and cadence."
            )
        if column in {"Keladi", "Jasmine", TARGET_COLUMN}:
            detail["value_counts"] = {
                str(_json_value(value)): int(count)
                for value, count in series.value_counts(dropna=False).items()
            }
        column_details[column] = detail

    sorted_timestamps = timestamp.sort_values()
    intervals = sorted_timestamps.diff().dropna()
    best_time_from_hour = (~timestamp.dt.hour.between(10, 15)).astype("int64")
    known_faults = {
        "Soil Temp1 (-127 or 85 reset codes)": int(
            dataframe["Soil Temp1"].isin([-127, 85]).sum()
        ),
        "Soil Temp2 (-127 or 85 reset codes)": int(
            dataframe["Soil Temp2"].isin([-127, 85]).sum()
        ),
        "Env Temp below 1 C": int((dataframe["Env Temp"] < 1).sum()),
        "Humidity above 100 percent": int((dataframe["Humidity"] > 100).sum()),
        "Soil Moisture 1 at 0 or 100": int(
            dataframe["Soil Moisture 1"].isin([0, 100]).sum()
        ),
        "Soil Moisture 2 at 0 or 100": int(
            dataframe["Soil Moisture 2"].isin([0, 100]).sum()
        ),
        "LDR at 0 or 100": int(dataframe["LDR"].isin([0, 100]).sum()),
    }
    iqr_outlier_candidates: dict[str, dict[str, Any]] = {}
    for column in FEATURE_COLUMNS:
        values = dataframe[column].dropna()
        first_quartile, third_quartile = values.quantile([0.25, 0.75])
        spread = third_quartile - first_quartile
        lower_fence = first_quartile - 1.5 * spread
        upper_fence = third_quartile + 1.5 * spread
        below = int((values < lower_fence).sum())
        above = int((values > upper_fence).sum())
        iqr_outlier_candidates[column] = {
            "lower_fence": _json_value(lower_fence),
            "upper_fence": _json_value(upper_fence),
            "below_fence": below,
            "above_fence": above,
            "candidate_count": below + above,
        }

    return {
        "dataset_file": Path(workbook_path).name,
        "format": "Excel workbook (.xlsx)",
        "sheet": DATA_SHEET,
        "shape": {"rows": int(dataframe.shape[0]), "columns": int(dataframe.shape[1])},
        "columns": column_details,
        "sample_records": [
            {
                column: _json_value(value)
                for column, value in row.items()
            }
            for row in dataframe.head(5).to_dict(orient="records")
        ],
        "duplicate_full_rows": int(dataframe.duplicated().sum()),
        "duplicate_timestamps": int(timestamp.duplicated().sum()),
        "timestamp": {
            "minimum": _json_value(timestamp.min()),
            "maximum": _json_value(timestamp.max()),
            "monotonic_increasing": bool(timestamp.is_monotonic_increasing),
            "most_common_interval": (
                str(intervals.mode().iloc[0]) if not intervals.empty else None
            ),
            "non_one_minute_intervals": int(
                (intervals != pd.Timedelta(minutes=1)).sum()
            ),
        },
        "sensor_rows_with_any_missing": int(
            dataframe[sensor_columns].isna().any(axis=1).sum()
        ),
        "sensor_rows_with_all_missing": int(
            dataframe[sensor_columns].isna().all(axis=1).sum()
        ),
        "target_counts": {
            str(_json_value(value)): int(count)
            for value, count in target_values.value_counts(dropna=False).items()
        },
        "best_time_hour_rule_mismatches": int(
            (target_values.astype("int64") != best_time_from_hour).sum()
        ),
        "documented_fault_marker_counts": known_faults,
        "iqr_outlier_candidates": iqr_outlier_candidates,
        "documented_notes": notes,
    }


def write_dataset_profile(profile: dict[str, Any], output_directory: Path) -> None:
    """Write a machine-readable profile and a human-readable data audit."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    json_path = output_directory / "dataset_profile.json"
    json_path.write_text(
        json.dumps(profile, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    row_count = profile["shape"]["rows"]
    missing_rows = profile["sensor_rows_with_any_missing"]
    target_counts = profile["target_counts"]
    marker_lines = "\n".join(
        f"- {name}: {count:,}"
        for name, count in profile["documented_fault_marker_counts"].items()
    )
    iqr_lines = "\n".join(
        f"- {name}: {result['candidate_count']:,} outside "
        f"[{result['lower_fence']:.3g}, {result['upper_fence']:.3g}]"
        for name, result in profile["iqr_outlier_candidates"].items()
    )
    missing_lines = "\n".join(
        f"- **{name}**: {detail['missing_count']:,} "
        f"({detail['missing_percent']:.2f}%)"
        for name, detail in profile["columns"].items()
        if detail["missing_count"]
    )
    descriptions = profile["documented_notes"]
    target_note = descriptions.get(
        "Best Time (target)", "No workbook definition was found."
    )
    event_note = descriptions.get(
        "Keladi, Jasmine (targets)", "No workbook definition was found."
    )
    dataset_note = descriptions.get("About", "No workbook overview was found.")
    quality_note = descriptions.get(
        "Injected data-quality issues", "No injected-quality note was found."
    )
    target_mismatch = profile["best_time_hour_rule_mismatches"]
    column_rows = [
        (
            f"| `{name}` | `{detail['dtype']}` | "
            f"{detail['missing_count']:,} ({detail['missing_percent']:.2f}%) | "
            f"{detail['unique_count_including_missing']:,} |"
        )
        for name, detail in profile["columns"].items()
    ]
    report = f"""# Dataset inspection report

## Source and scope

- Workbook: `{profile['dataset_file']}`; worksheet: `{profile['sheet']}`.
- Shape: {row_count:,} rows × {profile['shape']['columns']} columns.
- Timestamp range: {profile['timestamp']['minimum']} to
  {profile['timestamp']['maximum']}.
- Most common cadence: {profile['timestamp']['most_common_interval']};
  non-one-minute intervals: {profile['timestamp']['non_one_minute_intervals']:,}.
- Exact duplicate rows: {profile['duplicate_full_rows']:,};
  duplicate timestamps: {profile['duplicate_timestamps']:,}.
- Rows with at least one missing sensor input: {missing_rows:,};
  rows with every sensor input missing:
  {profile['sensor_rows_with_all_missing']:,}.

## Workbook documentation and target decision

> {dataset_note}

> **Best Time:** {target_note}

> **Keladi/Jasmine:** {event_note}

The workbook explicitly documents `Best Time` as a binary time-window label, so it
is used as the classification target. Its name is broader than its meaning:
class 1 includes the stated best window and other allowed windows, while class 0
is the midday pause window. `Keladi` and `Jasmine` are separate irrigation-event
indicators and are excluded from model inputs. The target equals the documented
time-of-day rule in **{row_count - target_mismatch:,}/{row_count:,} rows**
(mismatches: {target_mismatch:,}). The timestamp is retained for chronological
splitting and error analysis but excluded from model features. This is not a
prediction of soil water need or pump operation.

## Schema and missingness

| Column | Type | Missing | Unique values |
|---|---:|---:|---:|
{chr(10).join(column_rows)}

{missing_lines if missing_lines else 'No missing values were found.'}

## Distributions and quality observations

- `Best Time` counts: {target_counts}.
- Per-column numerical summaries, sample records, and low-cardinality target
  counts are in [`dataset_profile.json`](./dataset_profile.json).
- The profile includes every distinct value for non-datetime columns with at
  most 200 distinct observations; high-cardinality timestamps are summarized by
  range and cadence.
- The workbook documents synthetic noise, random missing cells, sensor
  faults/outliers, and label flips as follows:

> {quality_note}

Observed documented fault-marker counts (some markers may be legitimate sensor
boundary readings and are **not** indiscriminately discarded):

{marker_lines}

The univariate 1.5×IQR fence flags the following candidate counts (diagnostic
only; these are **not** automatically removed because time trends and real
sensor boundary values can also lie outside an IQR fence):

{iqr_lines}

No whole rows are dropped. Known impossible/reset values are converted to
missing for model inputs, then the training-only median imputer handles missing
features. Sensor boundary values and undocumented transient spikes are retained
because their validity cannot be established row-by-row from the workbook.
Target values are never imputed or generated.

## Modeling implications

This is a minute-level chronological series with strong serial dependence.
Evaluation therefore uses the final 20% of observations as a time-ordered
holdout, with no shuffle. A most-frequent-class dummy model is compared against
a shallow decision tree. The positive class is `Best Time = 1`; the classes are
imbalanced (75%/25% in the supplied workbook), so per-class precision, recall,
F1, and the confusion matrix are reported alongside accuracy.

The label is explicitly derived from collection time and not from irrigation
need. Sensor readings can act as proxies for time-of-day; because timestamp is
not supplied to the model, this avoids direct target leakage but does not
establish a causal or agronomic relationship. The workbook describes the data
as semi-synthetic: environmental data are based on real logs, while soil
moisture and irrigation targets are simulated. Results should not be
generalized to real hardware without validation.
"""
    (output_directory / "dataset_profile.md").write_text(report, encoding="utf-8")


def prepare_model_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return sensor-only model inputs with documented impossible readings nulled."""
    features = (
        dataframe.loc[:, FEATURE_COLUMNS]
        .apply(pd.to_numeric, errors="coerce")
        .copy()
    )

    # These are explicit reset/failure codes described in the workbook notes.
    for column in ("Soil Temp1", "Soil Temp2"):
        features.loc[features[column].isin([-127, 85]), column] = np.nan
    features.loc[features["Env Temp"] < 1, "Env Temp"] = np.nan
    features.loc[features["Env Temp"] == 114, "Env Temp"] = np.nan
    features.loc[features["Humidity"] > 100, "Humidity"] = np.nan
    failed_dht = (dataframe["Env Temp"] < 1) & (dataframe["Humidity"] == 0)
    features.loc[failed_dht, "Humidity"] = np.nan

    return features


def chronological_split(
    dataframe: pd.DataFrame,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """Split chronologically without shuffling and exclude time from model inputs."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be greater than 0 and less than 1.")
    if not dataframe[TIMESTAMP_COLUMN].is_monotonic_increasing:
        raise ValueError(
            "Rows must be ordered by Timestamp before chronological splitting."
        )
    if dataframe[TIMESTAMP_COLUMN].duplicated().any():
        raise ValueError(
            "Duplicate timestamps prevent an unambiguous chronological split."
        )

    target = pd.to_numeric(dataframe[TARGET_COLUMN], errors="coerce")
    if target.isna().any():
        raise ValueError("Best Time contains missing or non-numeric target values.")
    labels = set(target.unique())
    if not labels.issubset({0, 1}):
        raise ValueError(
            f"Expected binary Best Time labels 0 and 1; found {sorted(labels)}."
        )

    boundary = int(len(dataframe) * (1 - test_size))
    if boundary < 1 or boundary >= len(dataframe):
        raise ValueError(
            "The requested test_size leaves an empty training or test partition."
        )
    model_inputs = prepare_model_features(dataframe)
    return (
        model_inputs.iloc[:boundary].copy(),
        model_inputs.iloc[boundary:].copy(),
        target.iloc[:boundary].astype("int64"),
        target.iloc[boundary:].astype("int64"),
        dataframe.iloc[boundary:].copy(),
    )
