# Model Evaluation — ML-Based Traffic Speed Prediction

This document describes the evaluation of the trained models published in this repository, accompanying the paper "Cloud-Fog-Edge Deployment of ML-Based Traffic Speed Prediction: An Empirical Analysis of Performance, Cost, and Scalability".

## Models

One multilayer perceptron per sensor direction (37 sensor directions; layer sizes 96-256-256-96-256, ReLU with dropout, ~142k parameters, RMSprop optimiser, MSE loss, float32). Each model ships with its fitted input (x) and output (y) RobustScaler as `.pkl` files. Two model sets are published:

- `trained_models_full42y/` - the headline models evaluated in this document, trained on the full data span (1 December 2020 to 1 February 2025, ~4.2 years) with the final 20% held out; these correspond to the aggregate accuracy reported in the paper (mean MAPE ~12%).
- `trained_models_base3y/` - the three-year base models (trained to 1 December 2023) from which all warm-start re-training scenarios in the paper are initialised; included so the re-training experiments are reproducible.

All models use 15 engineered features (cyclic time encodings and event flags) from five-minute average-speed records of Greater Manchester drakewell sensors.

## Evaluation protocol

Each sensor's data is split chronologically 80/20; the final 20% of the full span serves as the held-out test set - approximately April 2024 to February 2025, with the exact boundary varying per sensor between late January and late April 2024 depending on data completeness. Metrics: mean absolute error (MAE, mph), root mean squared error (RMSE, mph), and mean absolute percentage error (MAPE, %). For context, each row also reports: the coefficient of variation of speed over the full modelled span, 1 December 2020 to 1 February 2025 (Speed CoV, dimensionless) - i.e., the variability of exactly the data the model was trained and evaluated on; the within-time-slot CoV (standard deviation of speed after removing the sensor's time-of-week mean profile, divided by mean speed, computed over calendar year 2023 - a stable year lying entirely inside the training portion); and the congestion frequency (share of five-minute intervals below 50% of the sensor's 85th-percentile free-flow speed, over the full span). A complementary common-test-set evaluation (identical held-out periods for all training scenarios) is summarised in the last section.

## Per-sensor results — 4.2-year training regime

All rows in this table come from the **full-span (4.2-year) training regime**; these are the published models in `trained_models_full42y/` and the source of the paper's ~12% aggregate MAPE.

| # | Sensor | MAE (mph) | RMSE (mph) | MAPE (%) | Speed CoV | Within-slot CoV | Congestion (%) | Test samples |
|---|---|---|---|---|---|---|---|---|
| 1 | 1020 nw | 1.91 | 2.88 | 7.78 | 0.128 | 0.099 | 0.63 | 85,096 |
| 2 | 1020 se | 1.73 | 2.50 | 6.96 | 0.139 | 0.092 | 0.41 | 84,978 |
| 3 | 1056 e | 1.28 | 1.81 | 5.62 | 0.086 | 0.078 | 0.11 | 83,327 |
| 4 | 1056 w | 1.33 | 1.97 | 5.98 | 0.099 | 0.087 | 0.17 | 84,435 |
| 5 | 1063 e | 1.48 | 2.40 | 5.55 | 0.097 | 0.082 | 0.43 | 84,066 |
| 6 | 1063 w | 1.56 | 2.33 | 5.23 | 0.110 | 0.076 | 0.52 | 84,086 |
| 7 | 1071 ne | 2.22 | 4.03 | 12.45 | 0.128 | 0.097 | 1.72 | 86,622 |
| 8 | 1071 sw | 1.40 | 2.19 | 5.01 | 0.082 | 0.060 | 0.17 | 86,872 |
| 9 | 1163 nw | 2.80 | 4.16 | 12.39 | 0.162 | 0.105 | 1.31 | 84,322 |
| 10 | 1163 se | 3.05 | 4.72 | 17.06 | 0.190 | 0.140 | 5.41 | 84,329 |
| 11 | 1165 se | 2.29 | 3.49 | 9.89 | 0.130 | 0.082 | 1.93 | 85,811 |
| 12 | 1173 n | 1.69 | 2.40 | 6.14 | 0.437 | 0.091 | 21.36 | 79,013 |
| 13 | 1273 n | 2.04 | 3.29 | 8.50 | 0.123 | 0.095 | 0.83 | 80,007 |
| 14 | 1276 e | 1.58 | 2.30 | 7.16 | 0.154 | 0.091 | 1.04 | 82,264 |
| 15 | 1276 w | 3.12 | 4.51 | 22.28 | 0.271 | 0.197 | 12.92 | 82,137 |
| 16 | 1385 n | 3.70 | 4.54 | 15.49 | 0.190 | 0.149 | 6.18 | 80,483 |
| 17 | 1404 e | 5.01 | 8.41 | 41.19 | 0.201 | 0.136 | 6.28 | 83,741 |
| 18 | 1404 w | 5.39 | 10.26 | 87.58 | 0.216 | 0.120 | 6.53 | 81,784 |
| 19 | 1407 n | 1.85 | 2.79 | 8.03 | 0.129 | 0.101 | 1.15 | 87,024 |
| 20 | 1407 s | 2.10 | 3.21 | 10.15 | 0.171 | 0.109 | 3.80 | 86,909 |
| 21 | 1413 ne | 1.51 | 2.15 | 5.96 | 0.102 | 0.076 | 0.34 | 85,531 |
| 22 | 1413 sw | 1.75 | 2.44 | 7.02 | 0.100 | 0.076 | 0.11 | 85,402 |
| 23 | 1414 ne | 1.74 | 2.53 | 7.66 | 0.124 | 0.081 | 0.39 | 84,851 |
| 24 | 1414 sw | 1.60 | 2.33 | 6.81 | 0.114 | 0.072 | 0.29 | 84,871 |
| 25 | 1417 s | 1.43 | 1.99 | 6.90 | 0.111 | 0.091 | 0.20 | 86,699 |
| 26 | 1418 e | 1.95 | 3.41 | 9.95 | 0.118 | 0.096 | 1.44 | 79,797 |
| 27 | 1418 w | 1.69 | 2.77 | 6.89 | 0.093 | 0.090 | 0.57 | 80,111 |
| 28 | 1420 nw | 1.95 | 3.33 | 11.19 | 0.158 | 0.109 | 3.37 | 86,446 |
| 29 | 1420 se | 1.95 | 2.92 | 9.18 | 0.128 | 0.106 | 0.92 | 86,213 |
| 30 | 1425 e | 2.04 | 3.54 | 4.52 | 0.066 | 0.048 | 0.14 | 86,582 |
| 31 | 1425 w | 2.52 | 5.11 | 8.13 | 0.080 | 0.074 | 0.51 | 86,589 |
| 32 | 1426 e | 2.34 | 4.13 | 6.10 | 0.075 | 0.066 | 0.17 | 86,018 |
| 33 | 1426 w | 4.11 | 7.70 | 19.89 | 0.171 | 0.121 | 3.26 | 87,203 |
| 34 | 1428 e | 3.83 | 6.65 | 12.40 | 0.164 | 0.113 | 5.04 | 85,481 |
| 35 | 1428 w | 2.24 | 3.43 | 4.39 | 0.066 | 0.060 | 0.06 | 85,345 |
| 36 | 1429 e | 1.90 | 2.61 | 6.94 | 0.108 | 0.088 | 0.07 | 82,425 |
| 37 | 1429 w | 2.01 | 2.96 | 7.62 | 0.108 | 0.095 | 0.25 | 81,458 |

Aggregate: mean MAE 2.27 mph (median 1.95); mean MAPE 11.95% (median 7.66%). The highest-error sensor directions are 1404 west (MAPE 87.6%), 1404 east (41.2%), and 1276 west (22.3%).

Per-sensor error is well explained by the variability columns of the table, and the figures below carry the argument. Against the Speed CoV and congestion frequency columns, MAPE correlates at Spearman rho = 0.79 and 0.80 respectively; against the within-slot CoV the correlation rises to rho = 0.88 (0.86 with the two 1404 directions excluded), visible in the right panel of the figure below. The left panel ranks all 37 sensors by within-slot CoV: 1276 west stands first, matching the highest error among *stable* sites, while 1173 north - the sensor with the highest Speed CoV (0.437) - sits mid-pack on both. 1173's discrepancy between its two CoV columns is a historical level shift (yearly mean speed 8.4, 14.3, 47.4, 46.8 km/h for 2020-2023, entirely inside the training portion), which inflates the Speed CoV but is absorbed by the model's year feature; in its stable regime the road is highly predictable (MAPE 6.1%). The two 1404 directions are the opposite case: only moderate intrinsic variability (within-slot CoV 0.120/0.136), but extreme error - they sit far above the trend in the scatter for a reason external to the variability columns, explained below.

![Left: within-slot CoV (2023) per sensor, sorted; highlighted bars are the sensors discussed in the text. Right: within-slot CoV against held-out MAPE under the 4.2-year regime (Spearman rho = 0.88). The two 1404 directions lie far above the trend: their error is driven by a site change during the test period, not by intrinsic variability.](within_slot_cov.png)

The figure below shows the variability mechanism directly. With all weeks of 2023 folded onto one Monday-Sunday axis, 1276 west's percentile band balloons through every daytime: the same clock time spans near-standstill to free flow depending on the week, so a calendar-feature model - which must predict a single value per slot - is bounded by that band width (a time-of-week average alone still leaves 22% MAPE there). On 1425 east, the narrow ribbon explains its 4.5% MAPE; 1173 north's 2023 band (middle panel) is nearly as narrow, confirming its predictability despite the Speed CoV. In short, for stable sites, prediction difficulty is driven almost entirely by day-to-day spread at the same time-of-week, which is exactly what the within-slot CoV column measures.

![The within-slot spread seen directly: all weeks of 2023 folded onto one Monday-Sunday axis at 5-minute resolution. At each slot, the line is the mean across the ~52 weeks and the bands the 25th-75th and 10th-90th percentiles of those same weeks. On 1276 west (top) the same clock time spans near-standstill to free flow depending on the week. 1173 north (middle) shows why the Speed CoV misleads: within 2023 its band is a narrow ribbon despite the corpus-highest Speed CoV, which stems from pre-2022 level shifts. 1425 east (bottom, lowest within-slot CoV) barely fluctuates at all. A calendar-feature model predicts one value per slot, so the band width bounds its achievable accuracy.](sensor_1276w_profile.png)

The 1404 outliers have a different, site-level cause. Their held-out test period (April 2024 - February 2025) contains a months-long degradation of the road itself: both directions slowed progressively from ~57 km/h (August 2024) to ~33 km/h (January 2025) with the monthly standard deviation (SD, annotated on the chart) rising from ~5 to 17-23 km/h - most likely due to long-term roadworks or road alteration at the site - so models trained on the earlier road are evaluated against a changed one. Recurrent December dips are visible in earlier years too (2021-2023) but recovered each January; the 2024-2025 episode differs in onset (September, months before the seasonal dip), depth (10th-percentile speeds near standstill), and persistence (no January recovery by the end of the data). Excluding the two 1404 directions, the aggregate MAPE of the remaining 35 sensors is 8.95% (median 7.62%).

![Platform 1404, both directions: monthly mean speed with 25th-75th and 10th-90th percentile bands over the full span (Dec 2020 - Feb 2025). Earlier December dips recover each January; the shaded September 2024 - January 2025 decline is deeper, starts months earlier, and shows no recovery - a site-level regime change rather than seasonality, falling inside this regime's held-out test period.](sensor_1404_regime.png)

## Common-test-set scenario comparison

All training scenarios of the paper (3-year base, 4.2-year cold start, and progressive warm-start re-training windows PW3/PW6/PW12) were additionally evaluated on identical held-out periods. Mean MAE (mph) per evaluation window:

| Scenario | Dec 2024 - Feb 2025 | Oct 2024 - Feb 2025 | May 2024 - Feb 2025 |
|---|---|---|---|
| base (3y) | 3.09 | 2.83 | 2.44 |
| cold start (4.2y) | 3.02 | 2.71 | 2.32 |
| re-train PW3 | 2.91 | 2.68 | 2.35 |
| re-train PW6 | 2.89 | 2.64 | 2.31 |
| re-train PW12 | 2.90 | 2.61 | excluded* |

*PW12 trains up to September 2024, overlapping the nine-month window.

Warm-start re-training improves on both cold-start baselines in every window; prediction accuracy is invariant to the training platform (across repeated runs and platforms, the standard deviation of aggregate MAPE is below 0.3 percentage points in over 90% of measured cells).
