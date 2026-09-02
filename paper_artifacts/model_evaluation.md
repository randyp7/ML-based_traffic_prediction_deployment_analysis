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

All rows in this table come from the **full-span (4.2-year) training regime**; these are the published models in `trained_models_full42y/` and the source of the paper's ~12% aggregate MAPE. Sensor directions with MAPE above 15% are highlighted in bold.

| # | Tier group | Sensor | MAE (mph) | RMSE (mph) | MAPE (%) | Speed CoV | Within-slot CoV | Congestion (%) | Test samples |
|---|---|---|---|---|---|---|---|---|---|
| 1 | tiny | 1163 nw | 2.80 | 4.16 | 12.39 | 0.162 | 0.105 | 1.31 | 84,322 |
| 2 | small | 1071 ne | 2.22 | 4.03 | 12.45 | 0.128 | 0.097 | 1.72 | 86,622 |
| 3 | small | 1071 sw | 1.40 | 2.19 | 5.01 | 0.082 | 0.060 | 0.17 | 86,872 |
| 4 | small | 1407 n | 1.85 | 2.79 | 8.03 | 0.129 | 0.101 | 1.15 | 87,024 |
| 5 | small | 1407 s | 2.10 | 3.21 | 10.15 | 0.171 | 0.109 | 3.80 | 86,909 |
| 6 | small | 1417 s | 1.43 | 1.99 | 6.90 | 0.111 | 0.091 | 0.20 | 86,699 |
| 7 | small | 1420 nw | 1.95 | 3.33 | 11.19 | 0.158 | 0.109 | 3.37 | 86,446 |
| 8 | small | 1425 e | 2.04 | 3.54 | 4.52 | 0.066 | 0.048 | 0.14 | 86,582 |
| 9 | small | 1425 w | 2.52 | 5.11 | 8.13 | 0.080 | 0.074 | 0.51 | 86,589 |
| 10 | small | **1426 w** | 4.11 | 7.70 | **19.89** | 0.171 | 0.121 | 3.26 | 87,203 |
| | **small mean (n=9)** | | **2.18** | **3.76** | **9.59** | **0.122** | **0.090** | **1.59** | |
| 11 | medium | 1020 nw | 1.91 | 2.88 | 7.78 | 0.128 | 0.099 | 0.63 | 85,096 |
| 12 | medium | 1020 se | 1.73 | 2.50 | 6.96 | 0.139 | 0.092 | 0.41 | 84,978 |
| 13 | medium | 1056 e | 1.28 | 1.81 | 5.62 | 0.086 | 0.078 | 0.11 | 83,327 |
| 14 | medium | 1056 w | 1.33 | 1.97 | 5.98 | 0.099 | 0.087 | 0.17 | 84,435 |
| 15 | medium | 1063 e | 1.48 | 2.40 | 5.55 | 0.097 | 0.082 | 0.43 | 84,066 |
| 16 | medium | 1063 w | 1.56 | 2.33 | 5.23 | 0.110 | 0.076 | 0.52 | 84,086 |
| 17 | medium | **1163 se** | 3.05 | 4.72 | **17.06** | 0.190 | 0.140 | 5.41 | 84,329 |
| 18 | medium | 1165 se | 2.29 | 3.49 | 9.89 | 0.130 | 0.082 | 1.93 | 85,811 |
| 19 | medium | **1404 e** | 5.01 | 8.41 | **41.19** | 0.201 | 0.136 | 6.28 | 83,741 |
| 20 | medium | 1413 ne | 1.51 | 2.15 | 5.96 | 0.102 | 0.076 | 0.34 | 85,531 |
| 21 | medium | 1413 sw | 1.75 | 2.44 | 7.02 | 0.100 | 0.076 | 0.11 | 85,402 |
| 22 | medium | 1414 ne | 1.74 | 2.53 | 7.66 | 0.124 | 0.081 | 0.39 | 84,851 |
| 23 | medium | 1414 sw | 1.60 | 2.33 | 6.81 | 0.114 | 0.072 | 0.29 | 84,871 |
| 24 | medium | 1420 se | 1.95 | 2.92 | 9.18 | 0.128 | 0.106 | 0.92 | 86,213 |
| 25 | medium | 1426 e | 2.34 | 4.13 | 6.10 | 0.075 | 0.066 | 0.17 | 86,018 |
| 26 | medium | 1428 e | 3.83 | 6.65 | 12.40 | 0.164 | 0.113 | 5.04 | 85,481 |
| 27 | medium | 1428 w | 2.24 | 3.43 | 4.39 | 0.066 | 0.060 | 0.06 | 85,345 |
| | **medium mean (n=17)** | | **2.15** | **3.36** | **9.69** | **0.121** | **0.090** | **1.36** | |
| 28 | full | 1173 n | 1.69 | 2.40 | 6.14 | 0.437 | 0.091 | 21.36 | 79,013 |
| 29 | full | 1273 n | 2.04 | 3.29 | 8.50 | 0.123 | 0.095 | 0.83 | 80,007 |
| 30 | full | 1276 e | 1.58 | 2.30 | 7.16 | 0.154 | 0.091 | 1.04 | 82,264 |
| 31 | full | **1276 w** | 3.12 | 4.51 | **22.28** | 0.271 | 0.197 | 12.92 | 82,137 |
| 32 | full | **1385 n** | 3.70 | 4.54 | **15.49** | 0.190 | 0.149 | 6.18 | 80,483 |
| 33 | full | **1404 w** | 5.39 | 10.26 | **87.58** | 0.216 | 0.120 | 6.53 | 81,784 |
| 34 | full | 1418 e | 1.95 | 3.41 | 9.95 | 0.118 | 0.096 | 1.44 | 79,797 |
| 35 | full | 1418 w | 1.69 | 2.77 | 6.89 | 0.093 | 0.090 | 0.57 | 80,111 |
| 36 | full | 1429 e | 1.90 | 2.61 | 6.94 | 0.108 | 0.088 | 0.07 | 82,425 |
| 37 | full | 1429 w | 2.01 | 2.96 | 7.62 | 0.108 | 0.095 | 0.25 | 81,458 |
| | **full mean (n=10)** | | **2.51** | **3.90** | **17.86** | **0.182** | **0.111** | **5.12** | |

Aggregate: mean MAE 2.27 mph (median 1.95); mean MAPE 11.95% (median 7.66%). The highest-error sensor directions are 1404 west (MAPE 87.6%), 1404 east (41.2%), and 1276 west (22.3%).

The table groups sensors by the tier at which they enter the experiment, with a mean row per group, answering whether the full tier's additional sensors are intrinsically harder.

On average, the sensors added at the full tier are indeed more variable (Speed CoV 0.182 vs ~0.12, within-slot CoV 0.111 vs 0.090) and markedly more congested (5.1% vs ~1.4-1.6%), with correspondingly higher error. Two qualifications keep the claim honest. First, the group differences are only marginally significant (one-sided Mann-Whitney: p = 0.049 for within-slot CoV, 0.052 for congestion, 0.068 for Speed CoV), because they are concentrated in a few locations rather than spread across the group. Second, tier membership only partially aligns with difficulty: the full-added group contains 1404 west and 1276 west (the two largest errors among its members) but 1404 east - the second-largest error overall - enters at the medium tier. The defensible statement is therefore that the full tier's added sensors are on average more variable and more frequently congested, with the difference driven by a small number of demanding locations.

Per-sensor error is well explained by the variability columns of the table, and the figures below carry the argument. Against the Speed CoV and congestion frequency columns, MAPE correlates at Spearman rho = 0.79 and 0.80 respectively; against the within-slot CoV the correlation rises to rho = 0.88 (0.86 with the two 1404 directions excluded), visible in the right panel of the figure below. The left panel ranks all 37 sensors by within-slot CoV: 1276 west stands first, matching the highest error among *stable* sites, while 1173 north - the sensor with the highest Speed CoV (0.437) - sits mid-pack on both. 1173's discrepancy between its two CoV columns is a historical level shift (yearly mean speed 8.4, 14.3, 47.4, 46.8 km/h for 2020-2023, entirely inside the training portion), which inflates the Speed CoV but is absorbed by the model's year feature; in its stable regime the road is highly predictable (MAPE 6.1%). The two 1404 directions are the opposite case: only moderate intrinsic variability (within-slot CoV 0.120/0.136), but extreme error - they sit far above the trend in the scatter for a reason external to the variability columns, explained below.

![Left: within-slot CoV (2023) per sensor, sorted; highlighted bars are the sensors discussed in the text. Right: within-slot CoV against held-out MAPE under the 4.2-year regime (Spearman rho = 0.88). The two 1404 directions lie far above the trend: their error is driven by a site change during the test period, not by intrinsic variability.](within_slot_cov.png)

The figure below shows the variability mechanism directly. With all weeks of 2023 folded onto one Monday-Sunday axis, 1276 west's percentile band balloons through every daytime: the same clock time spans near-standstill to free flow depending on the week, so a calendar-feature model - which must predict a single value per slot - is bounded by that band width (a time-of-week average alone still leaves 22% MAPE there). On 1425 east, the narrow ribbon explains its 4.5% MAPE; 1173 north's 2023 band (middle panel) is nearly as narrow, confirming its predictability despite the Speed CoV. In short, for stable sites, prediction difficulty is driven almost entirely by day-to-day spread at the same time-of-week, which is exactly what the within-slot CoV column measures.

![The within-slot spread seen directly: all weeks of 2023 folded onto one Monday-Sunday axis at 5-minute resolution. At each slot, the line is the mean across the ~52 weeks and the bands the 25th-75th and 10th-90th percentiles of those same weeks. On 1276 west (top) the same clock time spans near-standstill to free flow depending on the week. 1173 north (middle) shows why the Speed CoV misleads: within 2023 its band is a narrow ribbon despite the corpus-highest Speed CoV, which stems from pre-2022 level shifts. 1425 east (bottom, lowest within-slot CoV) barely fluctuates at all. A calendar-feature model predicts one value per slot, so the band width bounds its achievable accuracy.](sensor_1276w_profile.png)

The 1404 outliers have a different, site-level cause. Their held-out test period (April 2024 - February 2025) contains a months-long degradation of the road itself: both directions slowed progressively from ~57 km/h (August 2024) to ~33 km/h (January 2025) with the monthly standard deviation (SD, annotated on the chart) rising from ~5 to 17-23 km/h - most likely due to long-term roadworks or road alteration at the site - so models trained on the earlier road are evaluated against a changed one. Recurrent December dips are visible in earlier years too (2021-2023) but recovered each January; the 2024-2025 episode differs in onset (September, months before the seasonal dip), depth (10th-percentile speeds near standstill), and persistence (no January recovery by the end of the data). Excluding the two 1404 directions, the aggregate MAPE of the remaining 35 sensors is 8.95% (median 7.62%).

![Platform 1404, both directions: monthly mean speed with 25th-75th and 10th-90th percentile bands over the full span (Dec 2020 - Feb 2025). Earlier December dips recover each January; the shaded September 2024 - January 2025 decline is deeper, starts months earlier, and shows no recovery - a site-level regime change rather than seasonality, falling inside this regime's held-out test period.](sensor_1404_regime.png)

The same site-level check was applied to the remaining highlighted high-MAPE sensors (figure below). None shows a December-2024 seasonal anomaly. One further site event was found: 1426 west is stable at ~80 km/h until December 2024 and then drops to 62 km/h in January 2025 with its monthly standard deviation tripling to ~27 km/h - a late-onset disruption inside the final month of the test period, which partially accounts for its 19.9% MAPE. The other three behave as their variability columns predict: 1276 west and 1163 southeast hold stable levels throughout (their errors are intrinsic day-to-day spread), and 1385 north shows only a modest level rise (~37 to ~41 km/h) from August 2024. Errors at the highlighted sensors therefore divide into two classes: intrinsic within-slot variability (1276 west, 1163 southeast, 1385 north) and site changes during the test period (1404 both directions, and 1426 west in its final month).

![Monthly speed level and spread for the remaining high-MAPE sensors over the full span; the shaded region is the held-out test period (April 2024 - February 2025). 1426 west shows a January 2025 disruption; 1276 west, 1163 southeast, and 1385 north show no site-level change.](sensor_highmape_monthly.png)

## Common-test-set scenario comparison

All training scenarios of the paper (3-year base, 4.2-year cold start, and progressive warm-start re-training windows PW3/PW6/PW12) were additionally evaluated on identical held-out periods. Each cell reports mean MAE (mph) / mean MAPE (%):

| Scenario | Dec 2024 - Feb 2025 | Oct 2024 - Feb 2025 | May 2024 - Feb 2025 |
|---|---|---|---|
| base (3y) | 3.09 / 19.09 | 2.83 / 16.51 | 2.44 / 12.25 |
| cold start (4.2y) | 3.02 / 22.88 | 2.71 / 18.03 | 2.32 / 12.49 |
| re-train PW3 | 2.91 / 22.16 | 2.68 / 17.78 | 2.35 / 12.52 |
| re-train PW6 | 2.89 / 21.85 | 2.64 / 17.55 | 2.31 / 12.31 |
| re-train PW12 | 2.90 / 21.97 | 2.61 / 17.46 | excluded* |

*PW12 trains up to September 2024, overlapping the nine-month window.

The two accuracy metrics disagree here, and the disagreement is informative: these evaluation windows fall inside the platform-1404 degradation period, and the two 1404 directions dominate the mean MAPE (raising it to 12-23% and making the 3-year base appear best on that metric even though it is worst on MAE in every window). Median MAPE, which reflects the typical sensor, stays at 7.5-9.2% across all scenarios and preserves the same ordering as MAE. Scenario rankings should therefore be read from the MAE column; on it, warm-start re-training improves on both cold-start baselines in every window.

Prediction accuracy is invariant to the training platform (across repeated runs and platforms, the standard deviation of aggregate MAPE is below 0.3 percentage points in over 90% of measured cells).

## Measurement imprecision (repeated runs)

Wall-clock training times are subject to run-to-run variation; every reported configuration was therefore executed repeatedly and is summarised as mean, standard deviation (SD), and a 95% confidence interval (Student's t). The table lists the training scenarios for every platform and sensor tier (Dell rows: native OS batches; runs whose telemetry showed a severely power-capped state - mean CPU clock below 1,650 MHz AND wall-clock above 1.5x the cell minimum - are excluded and documented in the repository's statistics file). "—" marks single-run cells where no interval can be formed. 

The number of runs varies across cells for four reasons. (1) Repetition was allocated differentially: short or high-variance configurations were repeated at least five times, while multi-hour configurations were kept at three repeats once their observed CoV was below ~4%, following standard practice in performance evaluation. (2) Cells pool every clean batch of the same configuration across measurement campaigns (February, March, and September 2026), so frequently exercised configurations accumulated more runs (e.g. n = 8-9 for the fog and cloud training cells). (3) The telemetry-based exclusion of power-capped fog runs subtracts from some Dell cells - in the extreme case (cold-start tiny) seven of eight archived runs ran in the capped state, leaving a single valid measurement. This cell is retained for transparency only; it supports no claim in the paper for two reasons: the paper's fog training-time results are drawn from the initial-mode and re-training cells (n = 3-9, all clean), and the tiny tier is a single-sensor minimal configuration used to validate the pipeline rather than a deployment scenario, so its cold-start wall-clock is never cited in the analysis. (4) A few cells originate from single exploratory runs that were never part of a repeated campaign and are retained for completeness.

| Platform | Scenario | Tier | n | Mean (s) | SD (s) | 95% CI half-width (s) | CoV (%) |
|---|---|---|---|---|---|---|---|
| CSF3 Cloud GPU (A100) | initial (3y) | tiny | 4 | 57.9 | 11.8 | 18.8 | 20.4 |
| CSF3 Cloud GPU (A100) | initial (3y) | small | 3 | 590.7 | 39.6 | 98.2 | 6.7 |
| CSF3 Cloud GPU (A100) | initial (3y) | medium | 3 | 1,609.5 | 35.6 | 88.3 | 2.2 |
| CSF3 Cloud GPU (A100) | initial (3y) | full | 3 | 2,135.8 | 34.5 | 85.6 | 1.6 |
| CSF3 Cloud GPU (A100) | cold start (4.2y) | tiny | 7 | 66.2 | 14.7 | 13.6 | 22.2 |
| CSF3 Cloud GPU (A100) | cold start (4.2y) | small | 9 | 690.6 | 39.2 | 30.1 | 5.7 |
| CSF3 Cloud GPU (A100) | cold start (4.2y) | medium | 9 | 1,861.5 | 79.6 | 61.3 | 4.3 |
| CSF3 Cloud GPU (A100) | cold start (4.2y) | full | 9 | 2,461.0 | 123.4 | 95.0 | 5.0 |
| Dell Precision Fog GPU | initial (3y) | tiny | 9 | 43.3 | 5.5 | 4.3 | 12.8 |
| Dell Precision Fog GPU | initial (3y) | small | 8 | 481.9 | 13.7 | 11.4 | 2.8 |
| Dell Precision Fog GPU | initial (3y) | medium | 6 | 1,286.2 | 36.6 | 38.4 | 2.8 |
| Dell Precision Fog GPU | initial (3y) | full | 8 | 1,727.0 | 28.9 | 24.1 | 1.7 |
| Dell Precision Fog GPU | cold start (4.2y) | tiny | 1 | 46.3 | 0.0 | — | 0.0 |
| Dell Precision Fog GPU | cold start (4.2y) | small | 4 | 533.7 | 68.8 | 109.4 | 12.9 |
| Dell Precision Fog GPU | cold start (4.2y) | medium | 6 | 1,374.3 | 40.9 | 42.9 | 3.0 |
| Dell Precision Fog GPU | cold start (4.2y) | full | 6 | 1,944.9 | 247.6 | 259.8 | 12.7 |
| CSF3 Jetson Sim | initial (3y) | tiny | 3 | 61.2 | 6.9 | 17.1 | 11.3 |
| CSF3 Jetson Sim | initial (3y) | small | 3 | 702.3 | 22.9 | 56.7 | 3.3 |
| CSF3 Jetson Sim | cold start (4.2y) | tiny | 6 | 66.1 | 11.2 | 11.8 | 17.0 |
| CSF3 Jetson Sim | cold start (4.2y) | small | 6 | 683.6 | 38.0 | 39.8 | 5.6 |
| CSF3 Edge RSU Sim | initial (3y) | tiny | 3 | 151.3 | 34.0 | 84.4 | 22.5 |
| CSF3 Edge RSU Sim | initial (3y) | small | 3 | 1,686.9 | 132.2 | 328.1 | 7.8 |
| CSF3 Edge RSU Sim | cold start (4.2y) | tiny | 8 | 120.1 | 39.8 | 33.2 | 33.1 |
| CSF3 Edge RSU Sim | cold start (4.2y) | small | 6 | 1,342.5 | 254.5 | 267.0 | 19.0 |
| CSF3 RPi Sim | initial (3y) | tiny | 3 | 137.1 | 45.6 | 113.3 | 33.3 |
| CSF3 RPi Sim | initial (3y) | small | 3 | 1,580.5 | 18.8 | 46.8 | 1.2 |
| CSF3 RPi Sim | cold start (4.2y) | tiny | 8 | 132.3 | 20.6 | 17.2 | 15.5 |
| CSF3 RPi Sim | cold start (4.2y) | small | 6 | 1,572.2 | 177.5 | 186.2 | 11.3 |
| AWS t2.medium RSU | initial (3y) | tiny | 5 | 159.9 | 10.8 | 13.5 | 6.8 |
| AWS t2.medium RSU | initial (3y) | small | 5 | 2,649.1 | 1,519.6 | 1889.2 | 57.4 |
| AWS t2.medium RSU | cold start (4.2y) | tiny | 10 | 160.9 | 37.9 | 27.1 | 23.5 |
| AWS t2.medium RSU | cold start (4.2y) | small | 6 | 2,760.6 | 1,451.1 | 1522.5 | 52.6 |

The warm-start re-training configurations (six windows x four tiers on the fog and Jetson-simulation platforms, plus three progressive windows on AWS) were run n = 2-6 times each; their median CoV is 3.6%. Two systematic variance mechanisms were identified from run telemetry and are reported as findings rather than averaged away: OS power-state frequency capping on the fog workstation (affected runs excluded by the stated rule) and burstable CPU-credit exhaustion on the AWS t2.medium, whose initial small-tier cell is bimodal - approximately 1,970 +/- 160 s with credits available versus ~5,360 s when exhausted - which is why its pooled SD is large.

Prediction accuracy, by contrast, is essentially noiseless across repetitions and platforms: over all repeated cells, the across-run standard deviation of aggregate MAPE has median 0.06 percentage points (90th percentile 0.27 pp), which substantiates reporting a single accuracy figure per configuration while wall-clock times carry mean +/- SD.
