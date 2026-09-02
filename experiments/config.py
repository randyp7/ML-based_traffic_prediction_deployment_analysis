"""
Experiment configuration — single source of truth for all experiment settings.

Sensor tiers from: the sensor tier selection notes
Model architecture from: best_hyperparameters.json (BSc dissertation tuned MLP)
"""

import os
from pathlib import Path

# ============================================================================
# Paths (all relative to src/ directory)
# ============================================================================

# Base directory: the src/ folder containing this experiment package
SRC_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = SRC_DIR / "data" / "parquet"
TRAFFIC_DIR = DATA_DIR / "traffic"
EVENTS_FILE = DATA_DIR / "events.parquet"
MANIFEST_FILE = DATA_DIR / "manifest.json"
RESULTS_DIR = SRC_DIR / "results"

# Existing codebase root (for data_export.py which needs InfluxDB modules)
PROJECT_ROOT = SRC_DIR.parent.parent.parent  # d:\git_d\dissertation

# ============================================================================
# Sensor Tiers — cumulative nesting: Tiny ⊂ Small ⊂ Medium ⊂ Full
# Each sensor is a (platform_id, sensor_id) tuple
# Source: the sensor tier selection notes (verified 2026-02-13)
# ============================================================================

# Tier 1: Tiny — BSc dissertation baseline (1 sensor)
TIER_TINY = [
    ("drakewell__1163", "avgspeed_nw"),
]

# Tier 2: Small — Tiny + top 9 full-span by completeness (10 sensors, 7 platforms)
TIER_SMALL = TIER_TINY + [
    ("drakewell__1426", "avgspeed_w"),
    ("drakewell__1407", "avgspeed_n"),
    ("drakewell__1407", "avgspeed_s"),
    ("drakewell__1071", "avgspeed_sw"),
    ("drakewell__1417", "avgspeed_s"),
    ("drakewell__1071", "avgspeed_ne"),
    ("drakewell__1425", "avgspeed_w"),
    ("drakewell__1425", "avgspeed_e"),
    ("drakewell__1420", "avgspeed_nw"),
]

# Tier 3: Medium — full-span + >=95% completeness (27 sensors, 15 platforms)
TIER_MEDIUM = TIER_SMALL + [
    ("drakewell__1420", "avgspeed_se"),
    ("drakewell__1426", "avgspeed_e"),
    ("drakewell__1165", "avgspeed_se"),
    ("drakewell__1413", "avgspeed_ne"),
    ("drakewell__1428", "avgspeed_e"),
    ("drakewell__1413", "avgspeed_sw"),
    ("drakewell__1428", "avgspeed_w"),
    ("drakewell__1020", "avgspeed_nw"),
    ("drakewell__1020", "avgspeed_se"),
    ("drakewell__1414", "avgspeed_sw"),
    ("drakewell__1414", "avgspeed_ne"),
    ("drakewell__1056", "avgspeed_w"),
    ("drakewell__1163", "avgspeed_se"),
    ("drakewell__1063", "avgspeed_w"),
    ("drakewell__1063", "avgspeed_e"),
    ("drakewell__1404", "avgspeed_e"),
    ("drakewell__1056", "avgspeed_e"),
]

# Tier 4: Full — full-span + >=90% completeness (37 sensors, 21 platforms)
TIER_FULL = TIER_MEDIUM + [
    ("drakewell__1429", "avgspeed_e"),
    ("drakewell__1276", "avgspeed_e"),
    ("drakewell__1276", "avgspeed_w"),
    ("drakewell__1404", "avgspeed_w"),
    ("drakewell__1429", "avgspeed_w"),
    ("drakewell__1385", "avgspeed_n"),
    ("drakewell__1418", "avgspeed_w"),
    ("drakewell__1273", "avgspeed_n"),
    ("drakewell__1418", "avgspeed_e"),
    ("drakewell__1173", "avgspeed_n"),
]

# Tier registry for CLI lookup
TIERS = {
    "tiny": TIER_TINY,
    "small": TIER_SMALL,
    "medium": TIER_MEDIUM,
    "full": TIER_FULL,
}

# ============================================================================
# Data Export Settings
# ============================================================================

DATE_RANGE_START = "2020-12-01"
DATE_RANGE_END = "2025-02-01"
BATCH_DAYS = 7  # InfluxDB batch query window
EXPECTED_POINTS_PER_SENSOR = 438624  # 288/day x 1,523 days

# ============================================================================
# Model Architecture — tuned MLP from BSc dissertation (best_hyperparameters.json)
# Fixed across all platforms for fair comparison (no per-platform tuning).
# ============================================================================

MODEL_LAYERS = [
    {"units": 96,  "activation": "relu", "dropout": 0.3},   # input layer
    {"units": 256, "activation": "relu", "dropout": 0.2},   # hidden 1
    {"units": 256, "activation": "relu", "dropout": 0.1},   # hidden 2
    {"units": 96,  "activation": "relu", "dropout": 0.3},   # hidden 3
    {"units": 256, "activation": "relu", "dropout": 0.0},   # hidden 4
]
MODEL_OPTIMIZER = "rmsprop"
MODEL_LOSS = "mse"

# ============================================================================
# Training Configuration
# ============================================================================

BATCH_SIZE = 256
EPOCHS = 50
PATIENCE = 5
VALIDATION_SPLIT = 0.2
TRAIN_TEST_SPLIT = 0.8

# ============================================================================
# Inference Configuration
# ============================================================================

SINGLE_INFERENCE_REPEATS = 100
SUSTAINED_INFERENCE_STEPS = 288  # 1 day of 5-minute intervals

# ============================================================================
# Reproducibility
# ============================================================================

RANDOM_SEED = 42

# ============================================================================
# Speed Conversion
# ============================================================================

MS_TO_MPH = 0.621371

# ============================================================================
# Platform Registry — for result directory naming
# ============================================================================

PLATFORMS = {
    # Cloud tier
    "csf3_cloud_gpu": "cloud_baseline",
    # Fog tier
    "dell_precision_fog_gpu": "fog_accelerator",
    "dell_precision_fog_cpu": "fog_standard",
    "hp_workstation_fog": "fog_standard",
    # Edge tier
    "hp_elitebook_edge": "edge_intelligent",
    "dell_xps_edge": "edge_intelligent",
    "aws_t2_rsu": "edge_rsu",
    # CSF3 simulated tiers (resource-constrained)
    "csf3_jetson_sim": "edge_intelligent",
    "csf3_edge_rsu_sim": "edge_rsu",
    "csf3_rpi_sim": "edge_constrained",
    # Development
    "local": "development",
}

# ============================================================================
# Monitoring
# ============================================================================

MONITOR_INTERVAL = 0.5  # seconds between CPU/GPU readings

# ============================================================================
# Re-training Configuration
# ============================================================================

# Initial training date range (3 years of data for base model)
INITIAL_TRAIN_DATE_FROM = "2020-12-01"
INITIAL_TRAIN_DATE_TO = "2023-12-01"

# Re-training update windows (fine-tune base model on recent data)
RETRAIN_WINDOWS = {
    "W3":  {"date_from": "2024-12-01", "date_to": "2025-02-01"},
    "W6":  {"date_from": "2024-09-01", "date_to": "2025-02-01"},
    "W12": {"date_from": "2024-03-01", "date_to": "2025-02-01"},
}

# Progressive windows — start immediately after base training (no gap)
RETRAIN_WINDOWS_PROGRESSIVE = {
    "PW3":  {"date_from": "2023-12-01", "date_to": "2024-03-01"},
    "PW6":  {"date_from": "2023-12-01", "date_to": "2024-06-01"},
    "PW12": {"date_from": "2023-12-01", "date_to": "2024-12-01"},
}


def sensor_filename(platform_id, sensor_id):
    """Generate the Parquet filename for a sensor."""
    return f"{platform_id}__{sensor_id}.parquet"


def sensor_key(platform_id, sensor_id):
    """Generate a human-readable key for a sensor."""
    return f"{platform_id}__{sensor_id}"
