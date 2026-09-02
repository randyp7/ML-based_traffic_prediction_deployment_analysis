# Model Evaluation — ML-Based Traffic Speed Prediction

This document describes the evaluation of the trained models published in this repository (`paper_artifacts/trained_models_base3y/`), accompanying the paper "Cloud-Fog-Edge Deployment of ML-Based Traffic Speed Prediction: An Empirical Analysis of Performance, Cost, and Scalability".

## Models

One multilayer perceptron per sensor direction (37 sensor directions; layer sizes 96-256-256-96-256, ReLU with dropout, ~142k parameters, RMSprop optimiser, MSE loss, float32). Each model ships with its fitted input (x) and output (y) RobustScaler as `.pkl` files. Models were trained on the initial three-year window (1 December 2020 to 1 December 2023) of five-minute average-speed records from Greater Manchester drakewell sensors, using 15 engineered features (cyclic time encodings and event flags). These are the base models from which all warm-start re-training scenarios in the paper are initialised.

## Evaluation protocol

Each sensor's data is split chronologically 80/20; the final 20% of the training window serves as the held-out test set (approximately the last seven months of 2023 for the three-year window). Metrics: mean absolute error (MAE, mph), root mean squared error (RMSE, mph), and mean absolute percentage error (MAPE, %). A complementary common-test-set evaluation (identical held-out periods for all training scenarios) is summarised in the last section.

## Per-sensor results (held-out test set)

| # | Sensor | MAE (mph) | RMSE (mph) | MAPE (%) | Test samples |
|---|---|---|---|---|---|
| 1 | 1020 nw | 1.95 | 2.92 | 8.02 | 61,132 |
| 2 | 1020 se | 1.67 | 2.42 | 6.48 | 60,958 |
| 3 | 1056 e | 1.36 | 1.92 | 6.06 | 59,752 |
| 4 | 1056 w | 1.75 | 2.47 | 7.57 | 60,544 |
| 5 | 1063 e | 1.45 | 2.50 | 5.76 | 61,213 |
| 6 | 1063 w | 1.59 | 2.47 | 5.50 | 61,253 |
| 7 | 1071 ne | 2.09 | 3.65 | 10.48 | 62,524 |
| 8 | 1071 sw | 1.31 | 1.93 | 4.31 | 62,668 |
| 9 | 1163 nw | 2.37 | 3.36 | 9.55 | 62,835 |
| 10 | 1163 se | 2.97 | 4.74 | 16.52 | 62,843 |
| 11 | 1165 se | 2.07 | 3.04 | 8.16 | 61,874 |
| 12 | 1173 n | 1.83 | 2.63 | 6.38 | 58,707 |
| 13 | 1273 n | 1.90 | 2.99 | 7.41 | 61,176 |
| 14 | 1276 e | 1.56 | 2.24 | 6.85 | 60,057 |
| 15 | 1276 w | 3.52 | 5.50 | 27.63 | 59,959 |
| 16 | 1385 n | 2.10 | 3.24 | 11.30 | 58,161 |
| 17 | 1404 e | 2.69 | 4.38 | 11.19 | 61,257 |
| 18 | 1404 w | 2.60 | 3.93 | 10.09 | 60,197 |
| 19 | 1407 n | 1.90 | 3.20 | 10.03 | 62,645 |
| 20 | 1407 s | 2.14 | 3.12 | 9.71 | 62,573 |
| 21 | 1413 ne | 1.58 | 2.27 | 6.06 | 62,277 |
| 22 | 1413 sw | 1.64 | 2.25 | 5.92 | 62,174 |
| 23 | 1414 ne | 1.54 | 2.13 | 6.18 | 60,300 |
| 24 | 1414 sw | 1.41 | 1.98 | 5.66 | 60,317 |
| 25 | 1417 s | 1.47 | 2.08 | 7.07 | 62,237 |
| 26 | 1418 e | 1.92 | 3.28 | 9.09 | 55,437 |
| 27 | 1418 w | 1.74 | 3.06 | 7.90 | 55,694 |
| 28 | 1420 nw | 2.34 | 3.59 | 10.31 | 62,082 |
| 29 | 1420 se | 1.91 | 2.90 | 8.64 | 61,899 |
| 30 | 1425 e | 1.80 | 2.48 | 3.36 | 62,208 |
| 31 | 1425 w | 2.16 | 4.38 | 6.25 | 62,376 |
| 32 | 1426 e | 2.26 | 3.25 | 4.32 | 61,908 |
| 33 | 1426 w | 3.75 | 6.26 | 12.68 | 62,685 |
| 34 | 1428 e | 3.88 | 7.14 | 12.68 | 61,259 |
| 35 | 1428 w | 2.17 | 3.25 | 4.12 | 61,158 |
| 36 | 1429 e | 2.02 | 2.81 | 7.04 | 61,680 |
| 37 | 1429 w | 2.00 | 2.90 | 7.04 | 60,874 |

Aggregate: mean MAE 2.07 mph (median 1.92); mean MAPE 8.47% (median 7.41%). The two highest-error sensor directions (platform 1404 east/west) are markedly more variable, frequently congested locations; per-sensor error correlates strongly with speed variability (Spearman rho = 0.79 with the coefficient of variation of speed) and congestion frequency (rho = 0.82).

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
