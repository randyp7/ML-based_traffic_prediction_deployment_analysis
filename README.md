# ML-Based Traffic Speed Prediction — Deployment Analysis

Code, benchmark results, and trained models accompanying the paper
**"Cloud-Fog-Edge Deployment of ML-Based Traffic Speed Prediction: An Empirical
Analysis of Performance, Cost, and Scalability"** (University of Manchester).

## Repository layout

| Path | Contents |
|---|---|
| [`paper_artifacts/`](paper_artifacts/) | **Start here for reviewer materials**: trained models + scalers, the model evaluation document, the observed monitoring metrics document, and a SHA-256 manifest. |
| `benchmark/` | The cross-platform benchmark system (runner, training/inference/feature-engineering phases, per-platform configuration). |
| `results/` | Sample benchmark result records (per-run JSON with wall-clock, accuracy, and resource telemetry). |
| `model/` | Shared data-loading, feature-engineering, and evaluation modules used by the benchmark. |
| `tools/` | Configuration template (credentials blanked). |

## The benchmark in one paragraph

One ~142k-parameter multilayer perceptron per sensor direction predicts five-minute
average traffic speed from 15 calendar and event features, for 37 sensor directions in
Greater Manchester (Dec 2020 – Feb 2025 data). The identical training and inference
pipeline is benchmarked across six platforms — HPC cloud GPU (A100), a fog workstation
(RTX 4090), simulated Jetson / roadside-unit / Raspberry Pi edge tiers, and an AWS
t2.medium instance — over four sensor-count tiers and multiple training scenarios
(initial training, cold-start re-training, and warm-start re-training over recent and
progressive windows).
