# Dataset inspection report

## Source and scope

- Workbook: `dataset.xlsx`; worksheet: `Data`.
- Shape: 139,680 rows × 11 columns.
- Timestamp range: 2026-05-12T00:00:00 to
  2026-08-16T23:59:00.
- Most common cadence: 0 days 00:01:00;
  non-one-minute intervals: 0.
- Exact duplicate rows: 0;
  duplicate timestamps: 0.
- Rows with at least one missing sensor input: 16,621;
  rows with every sensor input missing:
  0.

## Workbook documentation and target decision

> Main sheet "Data": one row per minute, continuous dates and times. This is a semi-synthetic dataset: environmental columns are based on the real logs, soil moisture and the irrigation targets are simulated, and realistic sensor noise, missing values, outliers and label errors were added on purpose (see "Injected data-quality issues"). Describe it as semi-synthetic in any publication. The clean version without these issues is irrigation_continuous_corrected_v5.xlsx.

> **Best Time:** Target based on the time of collection. 1 = best time (20:00 to 04:59) or another allowed period (05:00 to 09:59 and 16:00 to 19:59, when the farmer is told it is not the best time to water). 0 = irrigation halted (10:00 to 15:59), the worst time because temperatures are highest. No blanks or errors in this column. 104760 rows = 1, 34920 rows = 0.

> **Keladi/Jasmine:** 1 = an irrigation event is in progress during that minute; 0 = no irrigation. Decisions are made on the noisy sensor reading, as the Arduino would. An event normally starts when the reading falls below a threshold that varies per event around 40 (37-43) and ends when it passes a stop level of 48-55 (4% of events stop early at 44-48), or when soil temperature <= 20 C, humidity >= 90 or air temperature > 35 C. About 30% of events start 1-20 minutes late and 3% of triggers are ignored for 30-120 minutes (pump fault, power cut). During an event the pump runs 3-second pulses with a 60 s soak, as in sketch_may3a.ino. A soil-moisture glitch can start or stop an event, as on the real device.

The workbook explicitly documents `Best Time` as a binary time-window label, so it
is used as the classification target. Its name is broader than its meaning:
class 1 includes the stated best window and other allowed windows, while class 0
is the midday pause window. `Keladi` and `Jasmine` are separate irrigation-event
indicators and are excluded from model inputs. The target equals the documented
time-of-day rule in **139,680/139,680 rows**
(mismatches: 0). The timestamp is retained for chronological
splitting and error analysis but excluded from model features. This is not a
prediction of soil water need or pump operation.

## Schema and missingness

| Column | Type | Missing | Unique values |
|---|---:|---:|---:|
| `Timestamp` | `datetime64[ns]` | 0 (0.00%) | 139,680 |
| `Soil Moisture 1` | `float64` | 2,514 (1.80%) | 65 |
| `Soil Moisture 2` | `float64` | 2,514 (1.80%) | 68 |
| `Soil Temp1` | `float64` | 2,514 (1.80%) | 153 |
| `Soil Temp2` | `float64` | 2,514 (1.80%) | 161 |
| `Env Temp` | `float64` | 2,514 (1.80%) | 144 |
| `Humidity` | `float64` | 2,514 (1.80%) | 73 |
| `LDR` | `float64` | 2,514 (1.80%) | 102 |
| `Keladi` | `float64` | 698 (0.50%) | 3 |
| `Jasmine` | `float64` | 698 (0.50%) | 3 |
| `Best Time` | `int64` | 0 (0.00%) | 2 |

- **Soil Moisture 1**: 2,514 (1.80%)
- **Soil Moisture 2**: 2,514 (1.80%)
- **Soil Temp1**: 2,514 (1.80%)
- **Soil Temp2**: 2,514 (1.80%)
- **Env Temp**: 2,514 (1.80%)
- **Humidity**: 2,514 (1.80%)
- **LDR**: 2,514 (1.80%)
- **Keladi**: 698 (0.50%)
- **Jasmine**: 698 (0.50%)

## Distributions and quality observations

- `Best Time` counts: {'1': 104760, '0': 34920}.
- Per-column numerical summaries, sample records, and low-cardinality target
  counts are in [`dataset_profile.json`](./dataset_profile.json).
- The profile includes every distinct value for non-datetime columns with at
  most 200 distinct observations; high-cardinality timestamps are summarized by
  range and cadence.
- The workbook documents synthetic noise, random missing cells, sensor
  faults/outliers, and label flips as follows:

> Noise: soil moisture +/-1.1 points (slowly wandering), soil temperature +/-0.08 C in 0.0625 C steps (DS18B20), air temperature +/-0.25 C, humidity +/-1.2% as whole numbers (DHT11), LDR +/-1.5. Missing values (blank cells): single cells removed at random rows, independently in each column (no blocks of consecutive rows). Missing per column: Soil Moisture 1 2514 (1.8%), Soil Moisture 2 2514 (1.8%), Soil Temp1 2514 (1.8%), Soil Temp2 2514 (1.8%), Env Temp 2514 (1.8%), Humidity 2514 (1.8%), LDR 2514 (1.8%), Keladi 698 (0.5%), Jasmine 698 (0.5%). Outliers: soil moisture readings stuck at 0 or 100 or jumping 15-30 points for one minute (227 and 176); soil temperature -127 (sensor disconnected), 85 (DS18B20 reset) or 3-8 C jumps (128 and 156); DHT11 failed reads (air temperature below 1 C with humidity 0) and jumps (air temperature 114, humidity 115); LDR 0/100 or 30-50 point jumps (210). Label errors: 1397 Keladi and 1397 Jasmine values (1%) flipped at single random rows, as from relay or logging glitches. Outliers are also single cells at random rows.

Observed documented fault-marker counts (some markers may be legitimate sensor
boundary readings and are **not** indiscriminately discarded):

- Soil Temp1 (-127 or 85 reset codes): 65
- Soil Temp2 (-127 or 85 reset codes): 81
- Env Temp below 1 C: 47
- Humidity above 100 percent: 0
- Soil Moisture 1 at 0 or 100: 116
- Soil Moisture 2 at 0 or 100: 102
- LDR at 0 or 100: 3,316

The univariate 1.5×IQR fence flags the following candidate counts (diagnostic
only; these are **not** automatically removed because time trends and real
sensor boundary values can also lie outside an IQR fence):

- Soil Moisture 1: 874 outside [36.5, 64.5]
- Soil Moisture 2: 2,306 outside [36.5, 64.5]
- Soil Temp1: 288 outside [24.1, 29.9]
- Soil Temp2: 490 outside [24.2, 29.8]
- Env Temp: 3,198 outside [27.2, 32.5]
- Humidity: 931 outside [42, 74]
- LDR: 1,081 outside [-47, 97]

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
