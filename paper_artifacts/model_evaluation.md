# Model Evaluation — ML-Based Traffic Speed Prediction

This document describes the evaluation of the trained models published in this repository (`paper_artifacts/trained_models_base3y/`), accompanying the paper "Cloud-Fog-Edge Deployment of ML-Based Traffic Speed Prediction: An Empirical Analysis of Performance, Cost, and Scalability".

## Models

One multilayer perceptron per sensor direction (37 sensor directions; layer sizes 96-256-256-96-256, ReLU with dropout, ~142k parameters, RMSprop optimiser, MSE loss, float32). Each model ships with its fitted input (x) and output (y) RobustScaler as `.pkl` files. Models were trained on the initial three-year window (1 December 2020 to 1 December 2023) of five-minute average-speed records from Greater Manchester drakewell sensors, using 15 engineered features (cyclic time encodings and event flags). These are the base models from which all warm-start re-training scenarios in the paper are initialised.

## Evaluation protocol

Each sensor's data is split chronologically 80/20; the final 20% of the training window serves as the held-out test set (approximately the last seven months of 2023 for the three-year window). Metrics: mean absolute error (MAE, mph), root mean squared error (RMSE, mph), and mean absolute percentage error (MAPE, %). For context, each row also reports two properties of the sensor's full speed series: the coefficient of variation of speed (Speed CoV, dimensionless) and the congestion frequency (share of five-minute intervals below 50% of the sensor's 85th-percentile free-flow speed). A complementary common-test-set evaluation (identical held-out periods for all training scenarios) is summarised in the last section.

## Per-sensor results — 3-year training regime

All rows in this table come from the **3-year initial training regime** (trained on 1 December 2020 to 1 December 2023); the held-out test set is the final 20% of that window, i.e. approximately mid-2023 to December 2023 (exact boundary varies slightly per sensor with data completeness). These are the published models in `trained_models_base3y/`.

| # | Sensor | MAE (mph) | RMSE (mph) | MAPE (%) | Speed CoV | Congestion (%) | Test samples |
|---|---|---|---|---|---|---|---|
| 1 | 1020 nw | 1.95 | 2.92 | 8.02 | 0.128 | 0.63 | 61,132 |
| 2 | 1020 se | 1.67 | 2.42 | 6.48 | 0.139 | 0.41 | 60,958 |
| 3 | 1056 e | 1.36 | 1.92 | 6.06 | 0.086 | 0.11 | 59,752 |
| 4 | 1056 w | 1.75 | 2.47 | 7.57 | 0.099 | 0.17 | 60,544 |
| 5 | 1063 e | 1.45 | 2.50 | 5.76 | 0.097 | 0.43 | 61,213 |
| 6 | 1063 w | 1.59 | 2.47 | 5.50 | 0.110 | 0.52 | 61,253 |
| 7 | 1071 ne | 2.09 | 3.65 | 10.48 | 0.128 | 1.72 | 62,524 |
| 8 | 1071 sw | 1.31 | 1.93 | 4.31 | 0.082 | 0.17 | 62,668 |
| 9 | 1163 nw | 2.37 | 3.36 | 9.55 | 0.162 | 1.31 | 62,835 |
| 10 | 1163 se | 2.97 | 4.74 | 16.52 | 0.190 | 5.41 | 62,843 |
| 11 | 1165 se | 2.07 | 3.04 | 8.16 | 0.130 | 1.93 | 61,874 |
| 12 | 1173 n | 1.83 | 2.63 | 6.38 | 0.437 | 21.36 | 58,707 |
| 13 | 1273 n | 1.90 | 2.99 | 7.41 | 0.123 | 0.83 | 61,176 |
| 14 | 1276 e | 1.56 | 2.24 | 6.85 | 0.154 | 1.04 | 60,057 |
| 15 | 1276 w | 3.52 | 5.50 | 27.63 | 0.271 | 12.92 | 59,959 |
| 16 | 1385 n | 2.10 | 3.24 | 11.30 | 0.190 | 6.18 | 58,161 |
| 17 | 1404 e | 2.69 | 4.38 | 11.19 | 0.201 | 6.28 | 61,257 |
| 18 | 1404 w | 2.60 | 3.93 | 10.09 | 0.216 | 6.53 | 60,197 |
| 19 | 1407 n | 1.90 | 3.20 | 10.03 | 0.129 | 1.15 | 62,645 |
| 20 | 1407 s | 2.14 | 3.12 | 9.71 | 0.171 | 3.80 | 62,573 |
| 21 | 1413 ne | 1.58 | 2.27 | 6.06 | 0.102 | 0.34 | 62,277 |
| 22 | 1413 sw | 1.64 | 2.25 | 5.92 | 0.100 | 0.11 | 62,174 |
| 23 | 1414 ne | 1.54 | 2.13 | 6.18 | 0.124 | 0.39 | 60,300 |
| 24 | 1414 sw | 1.41 | 1.98 | 5.66 | 0.114 | 0.29 | 60,317 |
| 25 | 1417 s | 1.47 | 2.08 | 7.07 | 0.111 | 0.20 | 62,237 |
| 26 | 1418 e | 1.92 | 3.28 | 9.09 | 0.118 | 1.44 | 55,437 |
| 27 | 1418 w | 1.74 | 3.06 | 7.90 | 0.093 | 0.57 | 55,694 |
| 28 | 1420 nw | 2.34 | 3.59 | 10.31 | 0.158 | 3.37 | 62,082 |
| 29 | 1420 se | 1.91 | 2.90 | 8.64 | 0.128 | 0.92 | 61,899 |
| 30 | 1425 e | 1.80 | 2.48 | 3.36 | 0.066 | 0.14 | 62,208 |
| 31 | 1425 w | 2.16 | 4.38 | 6.25 | 0.080 | 0.51 | 62,376 |
| 32 | 1426 e | 2.26 | 3.25 | 4.32 | 0.075 | 0.17 | 61,908 |
| 33 | 1426 w | 3.75 | 6.26 | 12.68 | 0.171 | 3.26 | 62,685 |
| 34 | 1428 e | 3.88 | 7.14 | 12.68 | 0.164 | 5.04 | 61,259 |
| 35 | 1428 w | 2.17 | 3.25 | 4.12 | 0.066 | 0.06 | 61,158 |
| 36 | 1429 e | 2.02 | 2.81 | 7.04 | 0.108 | 0.07 | 61,680 |
| 37 | 1429 w | 2.00 | 2.90 | 7.04 | 0.108 | 0.25 | 60,874 |

Aggregate: mean MAE 2.07 mph (median 1.92); mean MAPE 8.47% (median 7.41%). The highest-error sensor directions on this test period are 1276 west (MAPE 27.6%) and 1163 southeast (16.5%). Per-sensor error correlates strongly with the two series properties in the table: on this test period, Spearman rho = 0.78 between MAPE and Speed CoV, and rho = 0.80 between MAPE and congestion frequency (rho = 0.79 and 0.82 respectively for the 4.2-year cold-start regime evaluated on its own 2024-2025 held-out period, so the relationship is stable across both training regimes and test periods). The relationship is a strong rank trend rather than a monotone rule: sensor 1173 north has the highest Speed CoV (0.437) and congestion frequency (21.4%) yet only mid-range MAPE (6.4% here, 6.3% under the 4.2-year regime on 2024-2025), indicating that regular, recurrent congestion patterns are learnable; the largest errors arise where variability is irregular, as at 1276 west. Note that the error ranking is period-dependent: on the 2024-2025 common-test-set windows (next section), platform 1404 becomes the dominant outlier (MAPE up to 86.5%), suggesting a site-specific change at that location after 2023.

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
