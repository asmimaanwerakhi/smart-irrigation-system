# Smart Irrigation System Using Machine Learning

An academically honest, reproducible IoT 429 project that uses the supplied sensor dataset to classify its documented `Best Time` schedule-window label. It profiles the workbook, audits data quality, compares a majority-class baseline with a shallow decision tree, evaluates on a later chronological holdout, saves figures and predictions, and serializes the fitted scikit-learn pipeline.

> **Important:** The workbook's `Notes` sheet describes the data as semi-synthetic: environmental readings are based on real logs, while soil moisture and irrigation targets are simulated, with injected missing values, sensor faults/outliers, and label flips. `Best Time` is based on time of collection, not measured plant water need. This model is a coursework demonstration; it is not validated irrigation advice and it does not operate a physical pump.

## Objectives

- Inspect the supplied workbook and its accompanying data dictionary before modeling.
- Predict the documented binary `Best Time` window from seven sensor readings.
- Avoid using timestamp as a model feature; use it only for chronological splitting and error analysis.
- Compare a most-frequent-class baseline to a reproducible, shallow `DecisionTreeClassifier`.
- Report class-sensitive metrics, confusion matrix, predictions, and saved plots.

## Dataset and findings

- **File:** `dataset/dataset.xlsx`, Excel workbook with `Data` and `Notes` sheets.
- **Size:** 139,680 minute-level records; one row per minute from 2026-05-12 through 2026-08-16.
- **Columns:** `Timestamp`, seven sensor readings (`Soil Moisture 1/2`, `Soil Temp1/2`, `Env Temp`, `Humidity`, `LDR`), `Keladi`, `Jasmine`, and `Best Time`.
- **Missingness:** each of the seven sensor columns has 2,514 missing cells; `Keladi` and `Jasmine` each have 698; the target has none. There are no duplicate rows or timestamps, and the timestamps have one-minute cadence.
- **Target meaning:** the workbook says `Best Time = 0` from 10:00–15:59 (irrigation paused) and `1` in the other collection-time windows; observed labels match this rule for all 139,680 records. The column name is broader than class 1's meaning: it includes both the stated best window and other allowed windows. It is not a demonstrated plant-water-need signal.
- **Other label meanings:** `Keladi` and `Jasmine` identify separate irrigation events. They are excluded from model inputs and are not used as target labels.
- **Sensor quality:** workbook notes identify soil-temperature reset values (-127 and 85 °C), impossible air-temperature/humidity readings, endpoint values, and transient spikes. The preprocessing marks documented reset/failure codes missing. Sensor endpoints and transient spikes whose individual validity cannot be established are retained and explicitly reported rather than silently discarded.
- A detailed, generated schema/profile/quality report is written to [`outputs/evaluation/dataset_profile.md`](outputs/evaluation/dataset_profile.md) and `.json` when the workflow runs.

## Methodology

1. Parse the Excel `Data` sheet, validate required columns and timestamps, and summarize the `Notes` sheet.
2. Keep the seven numeric sensor readings as inputs. Exclude `Timestamp`, `Keladi`, and `Jasmine` from model features.
3. Use a chronological 80% train / 20% test split with no shuffling. At the default settings, the test period is the final 27,936 minutes.
4. Apply documented fault-code handling and median imputation in a scikit-learn `Pipeline`; the imputer is fitted on training data only. Trees do not need feature scaling.
5. Compare a most-frequent-class `DummyClassifier` against a `DecisionTreeClassifier` limited to depth 5 and 100 observations per leaf (random seed 429).
6. Report accuracy, class-1 precision/recall/F1, macro-F1, classification report, confusion matrices, actual-versus-predicted rows, and error rates by hour.

Since `Best Time` is determined directly by collection time, sensor patterns may serve as proxies for time-of-day. Excluding timestamp prevents direct input leakage, but does **not** show that the learned associations are causal, agronomically meaningful, or useful for deciding whether a plant needs water. The chronological holdout reduces minute-level time-series leakage compared with random splitting.

## Technologies

Python 3.10+, pandas, NumPy, scikit-learn, Matplotlib, Seaborn, openpyxl, joblib, Jupyter, Streamlit, and Plotly.

## Project structure

```text
.
├── dataset/
│   └── dataset.xlsx
├── notebooks/
│   └── smart_irrigation_analysis.ipynb
├── models/
│   └── trained_model.joblib            # generated
├── outputs/
│   ├── evaluation/                     # generated reports and predictions
│   └── figures/                        # generated charts
├── .streamlit/
│   └── config.toml                      # dashboard theme
├── dashboard.py                        # saved-results dashboard entry point
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   ├── visualize.py
│   ├── dashboard_data.py
│   ├── dashboard_components.py
│   └── dashboard_pages.py
├── result-analysis.md
├── requirements.txt
└── README.md
```

The original workbook is kept in its existing `dataset/` location; it is not duplicated or moved.

## Installation

From the project root:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the project

Train, profile, evaluate, save charts, and create `models/trained_model.joblib`:

```powershell
python -m src.train
```

Re-evaluate the saved model against the same chronological holdout:

```powershell
python -m src.evaluate
```

To run the notebook, open `notebooks/smart_irrigation_analysis.ipynb` in VS Code/Jupyter and run all cells from top to bottom. The notebook executes the same training workflow and verifies the saved model's seven-feature input contract.

### Launch the results dashboard

Install the requirements above, then start the interactive dashboard from the project root:

```powershell
streamlit run dashboard.py
```

The dashboard reads the existing model metadata, JSON reports, prediction/error CSVs, feature-importance CSV, and saved figures. It does **not** retrain the model or generate new predictions. Missing or malformed artifacts are reported in the sidebar and affected sections show an empty state.

Optional arguments:

```powershell
python -m src.train --data dataset\dataset.xlsx --test-size 0.2
python -m src.evaluate --data dataset\dataset.xlsx --model models\trained_model.joblib
```

## Evaluation results

Results are generated from the actual workbook and written to `outputs/evaluation/metrics.json`, `classification_report.json`, `test_predictions.csv`, `error_analysis_by_hour.csv`, and `training_summary.json`. The plots are written under `outputs/figures/`.

| Model | Accuracy | Precision (class 1) | Recall (class 1) | F1 (class 1) | Macro-F1 |
|---|---:|---:|---:|---:|---:|
| Most-frequent baseline | 0.7517 | 0.7517 | 1.0000 | 0.8583 | 0.4291 |
| Decision tree | 0.8101 | 0.8779 | 0.8681 | 0.8730 | 0.7484 |

These are actual results from the final 27,936 observations (2026-07-28 14:24 through 2026-08-16 23:59); the class counts are 6,936 class-0 and 21,000 class-1 rows. The baseline predicts only the majority class, so its class-0 recall is zero. The decision tree's class-0 precision/recall/F1 are 0.6138/0.6344/0.6239. Its confusion matrix (rows = actual 0/1, columns = predicted 0/1) is `[[4400, 2536], [2769, 18231]]`. The tree improves holdout accuracy by about 5.84 percentage points, but the result is for schedule-window classification only. Read it alongside the per-class report, error analysis, target definition, and semi-synthetic dataset caveat; do not interpret it as irrigation efficacy.

## IoT integration concept

1. Soil and environmental sensors collect readings.
2. The device formats and validates the seven expected feature values.
3. The same trained preprocessing pipeline imputes documented missing/fault values and accepts the sensor feature columns.
4. The model emits a `Best Time` schedule-window prediction.
5. A separately designed irrigation controller could combine that schedule signal with agronomic policy, safety limits, manual overrides, and verified plant-water status before deciding what action to take.

Only steps 2–4 are demonstrated in software here. There is no physical hardware, pump-control code, safety interlock, or cloud deployment in this project.

## Limitations and future work

- Semi-synthetic data and simulated irrigation targets limit external validity.
- The target is a deterministic collection-time policy, not a ground-truth irrigation requirement; a clock-based rule may be more transparent than ML for this label.
- Sensor variables and the target may be correlated through time-of-day; sensor-only classification does not establish causal value.
- Workbook notes report injected noise, outliers, and label errors. Some transient spikes and boundary readings are retained because a reliable row-by-row distinction from valid readings is not available in the workbook.
- A 20% chronological holdout from one synthetic series is not a substitute for testing on new seasons, locations, crops, devices, and independently labeled irrigation outcomes.
- Future work should obtain the actual target-generation/plant-water policy, retain real ground-truth soil-water or irrigation decisions, validate sensor calibration, test on independent farms/seasons, and only then consider a fail-safe hardware integration.

## Reproducibility

The chronological split, decision-tree random seed, feature list, model parameters, and training-row count are saved with the model artifact. All generated evaluation outputs are reproducible by rerunning `python -m src.train` with the same workbook and dependencies.