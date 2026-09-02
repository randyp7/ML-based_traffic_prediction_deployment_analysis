# Cross-Platform Benchmark for MLP Traffic Speed Prediction

Benchmarking framework for evaluating a fixed MLP neural network across ITS (Intelligent Transport Systems) computing tiers — from cloud servers to edge RSUs.

## Overview

This benchmark measures **training time, inference latency, resource usage, and prediction accuracy** of a fixed-architecture MLP model across different hardware platforms representing real ITS deployment tiers:

| Platform | Tier | Example Hardware |
|---|---|---|
| `csf3_cloud_gpu` | Cloud Baseline | University HPC (A100 GPU) |
| `dell_precision_fog_gpu` | Fog Accelerator | Dell Precision 5680 (RTX 4090, i9-13900H, 32GB) |
| `hp_workstation_fog` | Fog Standard | HP Workstation (20 cores, 32GB, no GPU) |
| `hp_elitebook_edge` | Edge Intelligent | HP EliteBook (10 cores, 16GB) |
| `dell_xps_edge` | Edge Intelligent | Dell XPS 15 (4 cores, 16GB) |
| `aws_t2_rsu` | Edge RSU | AWS EC2 t2.medium (2 vCPU, 4GB) |

## Project Structure

```
src/
├── benchmark/
│   ├── config.py                 # All settings: tiers, model arch, training params
│   ├── runner.py                 # Main CLI orchestrator (4-phase pipeline)
│   ├── data_export.py            # One-time InfluxDB -> Parquet exporter
│   ├── metrics.py                # PhaseTimer, CPU/GPU monitors, system info
│   ├── requirements_benchmark.txt
│   └── phases/
│       ├── data_loading.py       # Load Parquet files for a sensor tier
│       ├── feature_eng.py        # Feature engineering wrapper
│       ├── training.py           # Model training (fixed architecture)
│       └── inference.py          # Inference benchmarking
├── data/
│   └── parquet/                  # Pre-exported data (not in git)
│       ├── manifest.json
│       ├── events.parquet
│       └── traffic/              # 37 sensor Parquet files
└── results/                      # Benchmark outputs (per-platform)

model/                            # Shared modules from main project
├── data_loader.py                # InfluxDB data loader (used by data_export)
├── feature_engineering.py        # Feature pipeline
└── evaluation.py                 # Accuracy metrics

tools/
└── config.py                     # InfluxDB connection settings
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r benchmark/requirements_benchmark.txt
```

### 2. Get the Data

The Parquet data files are pre-exported and not stored in git. Copy the `src/data/` directory from the development machine:

```
src/data/parquet/
├── manifest.json
├── events.parquet
└── traffic/
    ├── drakewell__1163__avgspeed_nw.parquet
    ├── drakewell__1426__avgspeed_w.parquet
    └── ... (37 files total, ~1.5 GB)
```

### 3. Run the Benchmark

```bash


# Quick validation (1 sensor, 1 run)
python -m benchmark.runner --platform <PLATFORM_NAME> --tiers tiny --runs 1

# Full benchmark (all tiers, 3 runs)
python -m benchmark.runner --platform <PLATFORM_NAME> --runs 3

# Specific tiers
python -m benchmark.runner --platform <PLATFORM_NAME> --tiers tiny small medium --runs 3

# Inference-only mode (for memory-constrained edge devices)
python -m benchmark.runner --platform <PLATFORM_NAME> --inference-only --pretrained-dir ./pretrained_models/ --runs 3
```

## Sensor Tiers

Progressive scaling with cumulative nesting (Tiny ⊂ Small ⊂ Medium ⊂ Full):

| Tier | Sensors | Description |
|---|---|---|
| `tiny` | 1 | BSc dissertation baseline sensor |
| `small` | 10 | Top full-span sensors by completeness |
| `medium` | 27 | Full-span + >=95% completeness |
| `full` | 37 | Full-span + >=90% completeness |

Data covers **2020-12-01 to 2025-02-01** (~4.2 years) at 5-minute intervals (288 points/day).

## Benchmark Pipeline (4 Phases)

1. **Data Loading** — Read pre-exported Parquet files, validate against manifest
2. **Feature Engineering** — Time features, event features (15 features total)
3. **Training** — Fixed MLP: Dense(128)→Dropout(0.2)→Dense(128)→Dropout(0.2)→Dense(64)→Dropout(0.1)→Dense(1). RobustScaler, 80/20 chronological split, early stopping (patience=5)
4. **Inference** — Model load time, single-sample latency (100 repeats), batch prediction, sustained 288-step prediction

## Output

Results are saved to `src/results/<platform>/<timestamp>/`:

| File | Contents |
|---|---|
| `tier_<tier>_run<n>.json` | Per-sensor accuracy (MAE, RMSE, MAPE), training time, inference latency, resource monitoring |
| `summary.json` | Aggregated metrics across runs |
| `system_info.json` | Hardware specs captured at runtime |
| `models/` | Trained `.keras` models + `.pkl` scalers |

## Data Export (Development Machine Only)

To re-export data from InfluxDB (only needed on the machine with InfluxDB access):

```bash


# Export all 37 sensors
python -m benchmark.data_export

# Export single sensor
python -m benchmark.data_export --sensor drakewell__1163 avgspeed_nw

# Verify exported data
python -m benchmark.data_export --verify
```
