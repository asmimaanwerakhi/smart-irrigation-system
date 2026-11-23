# Result Analysis — Smart Irrigation System Using Machine Learning

## 1. Purpose and interpretation

This experiment predicts the dataset's documented `Best Time` class from seven
sensor measurements. The workbook defines class `0` as the irrigation-pause
period from 10:00–15:59 and class `1` as the other schedule windows. Its name
can be misleading: class `1` includes both the stated best window and other
allowed periods that the workbook says are not necessarily the best time to
water.

This is therefore **schedule-window classification**, not prediction of plant
water need, irrigation efficacy, or pump status. The workbook's `Notes` sheet
describes the dataset as semi-synthetic: environmental readings are based on
real logs, but soil-moisture readings and irrigation targets are simulated.
Consequently, these results are for a coursework demonstration and should not
be treated as field validation.

## 2. Evaluation design

- Dataset: 139,680 minute-level observations.
- Split: first 111,744 rows for training and final 27,936 for testing; no
  shuffling.
- Test period: 2026-07-28 14:24 through 2026-08-16 23:59.
- Model inputs: `Soil Moisture 1`, `Soil Moisture 2`, `Soil Temp1`,
  `Soil Temp2`, `Env Temp`, `Humidity`, and `LDR`.
- `Timestamp` was excluded from the model inputs and used only to preserve
  chronological order and analyze errors. `Keladi` and `Jasmine` were excluded
  because the workbook describes them as separate irrigation-event labels.
- The pipeline uses training-set median imputation and a shallow decision tree
  (maximum depth 5, minimum leaf size 100, random seed 429). The comparison
  baseline predicts only the most frequent training class.
- Test support: 6,936 class-0 observations and 21,000 class-1 observations.
  Class 1 makes up about 75.2% of this holdout, so accuracy alone would hide
  performance on the less frequent class.

## 3. Baseline versus decision tree

Precision, recall, and F1 in this table use class `1` as the positive class.
Macro-F1 gives equal weight to the two classes.

| Model | Accuracy | Precision (1) | Recall (1) | F1 (1) | Macro-F1 |
|---|---:|---:|---:|---:|---:|
| Most-frequent baseline | 0.7517 | 0.7517 | 1.0000 | 0.8583 | 0.4291 |
| Decision tree | 0.8101 | 0.8779 | 0.8681 | 0.8730 | 0.7484 |

The decision tree improves holdout accuracy by **5.84 percentage points** over
the baseline and substantially improves macro-F1. The baseline's perfect
class-1 recall is not evidence of a useful classifier: it predicts class 1 for
every record and has zero precision, recall, and F1 for class 0.

### Per-class decision-tree report

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| 0 — irrigation pause | 0.6138 | 0.6344 | 0.6239 | 6,936 |
| 1 — other schedule window | 0.8779 | 0.8681 | 0.8730 | 21,000 |
| Macro average | 0.7458 | 0.7513 | 0.7484 | 27,936 |
| Weighted average | 0.8123 | 0.8101 | 0.8111 | 27,936 |

Although the tree performs better than the baseline, its class-0 recall of
0.6344 means it misses about 36.6% of the actual pause-window observations.
That error would matter in a real scheduling system and should not be hidden by
the overall accuracy.

## 4. Confusion matrix and errors

Rows are actual classes; columns are predicted classes.

| Actual class | Predicted 0 | Predicted 1 |
|---|---:|---:|
| 0 — irrigation pause | 4,400 | 2,536 |
| 1 — other schedule window | 2,769 | 18,231 |

The tree makes 5,305 errors out of 27,936 holdout records (18.99%). With class
1 treated as positive, it produces 2,536 false positives (actual pause class 0
predicted as 1) and 2,769 false negatives (actual class 1 predicted as 0).

### Error rate by collection hour

| Hour | Observations | Errors | Error rate |
|---:|---:|---:|---:|
| 00 | 1,140 | 19 | 1.67% |
| 01 | 1,140 | 13 | 1.14% |
| 02 | 1,140 | 14 | 1.23% |
| 03 | 1,140 | 10 | 0.88% |
| 04 | 1,140 | 9 | 0.79% |
| 05 | 1,140 | 2 | 0.18% |
| 06 | 1,140 | 1 | 0.09% |
| 07 | 1,140 | 9 | 0.79% |
| 08 | 1,140 | 378 | 33.16% |
| 09 | 1,140 | 538 | 47.19% |
| 10 | 1,140 | 576 | 50.53% |
| 11 | 1,140 | 541 | 47.46% |
| 12 | 1,140 | 409 | 35.88% |
| 13 | 1,140 | 355 | 31.14% |
| 14 | 1,176 | 318 | 27.04% |
| 15 | 1,200 | 337 | 28.08% |
| 16 | 1,200 | 711 | 59.25% |
| 17 | 1,200 | 418 | 34.83% |
| 18 | 1,200 | 424 | 35.33% |
| 19 | 1,200 | 219 | 18.25% |
| 20 | 1,200 | 0 | 0.00% |
| 21 | 1,200 | 2 | 0.17% |
| 22 | 1,200 | 2 | 0.17% |
| 23 | 1,200 | 0 | 0.00% |

Errors are concentrated between 08:00 and 19:00, with the highest hourly rate
at 16:00 (59.25%), followed by 10:00 (50.53%). The very low rates overnight
contrast with the much higher rates around daytime schedule transitions. These
are descriptive patterns in this holdout, not proof of why the errors occur.

## 5. Feature importance

| Feature | Decision-tree importance |
|---|---:|
| LDR | 81.61% |
| Soil Temp1 | 6.08% |
| Humidity | 5.82% |
| Soil Temp2 | 3.59% |
| Soil Moisture 1 | 1.86% |
| Soil Moisture 2 | 1.03% |
| Env Temp | 0.00% |

LDR dominates this particular tree's impurity-based importance. Since the
target is defined by collection time and light readings can vary with time of
day, LDR may act as a proxy for the schedule rule. Tree feature importance is
not causal evidence and does not show that LDR or the other sensors measure
plant water need.

## 6. Overall conclusion and limitations

On the supplied chronological holdout, the decision tree predicts the
documented schedule-window class better than a majority-class baseline. The
less frequent pause class remains harder to identify, and errors are especially
common during several daytime hours. The measured result supports only the
limited claim that sensor readings in this dataset contain information
associated with its `Best Time` labels.

The label is explicitly based on time of collection, not independent
agronomic ground truth. The dataset is semi-synthetic, and sensor values may
encode time-of-day. Excluding `Timestamp` prevents direct use of the label's
hour rule as a model feature, but it does not remove indirect time-related
associations. No physical controller, safety mechanism, or pump operation was
implemented. Real-world use would require independently collected field data,
verified irrigation-need labels, validation across seasons and devices, and
controller-level safety checks.

## 7. Reproducibility and supporting outputs

The metrics and confusion matrices are in
[`outputs/evaluation/metrics.json`](outputs/evaluation/metrics.json); class
reports are in
[`outputs/evaluation/classification_report.json`](outputs/evaluation/classification_report.json).
Per-record predictions and hourly errors are in
[`outputs/evaluation/test_predictions.csv`](outputs/evaluation/test_predictions.csv)
and
[`outputs/evaluation/error_analysis_by_hour.csv`](outputs/evaluation/error_analysis_by_hour.csv).
The dataset audit is in
[`outputs/evaluation/dataset_profile.md`](outputs/evaluation/dataset_profile.md),
and generated charts are in [`outputs/figures/`](outputs/figures/).

The reported values were produced by the project workflow using
`python -m src.train` and re-evaluated with `python -m src.evaluate`.