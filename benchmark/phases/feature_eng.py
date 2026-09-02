"""
Phase 2: Feature engineering wrapper.

Thin wrapper around existing feature_engineering.py — no new logic.
Mirrors the pipeline in train_model_nn.py.
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root so we can import existing modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent.parent))

from model.feature_engineering import FeatureEngineering, add_time_features, remove_unused_columns

logger = logging.getLogger(__name__)


def engineer_features_for_sensor(traffic_df, events_df):
    """
    Apply full feature pipeline to one sensor's raw traffic data.

    Pipeline (mirrors train_model_nn.py):
      1. add_time_features(traffic)
      2. fe.add_event_features(df)
      3. remove_unused_columns(df)
      4. Sort by _time, separate X and y

    Weather features remain disabled (matching current train_model_nn.py behaviour).

    Args:
        traffic_df: Raw traffic DataFrame (from Parquet, same structure as InfluxDB output)
        events_df: Event DataFrame (shared across sensors)

    Returns:
        X: numpy array of features
        y: numpy array of target values (speed in m/s)
        dates: pandas Series of _time values
    """
    # Work on a copy to avoid modifying the cached DataFrame
    df = traffic_df.copy()

    # Feature engineering (weather=None to disable weather features)
    fe = FeatureEngineering(events_df, None)
    df = add_time_features(df)
    df = fe.add_event_features(df)
    df = remove_unused_columns(df)

    # Sort by time
    df = df.sort_values(by="_time")
    dates = df["_time"].copy()
    df = df.drop(columns=["_time"])

    # Separate features and target
    X = df.drop(columns=["value"]).values
    y = df["value"].values.reshape(-1, 1)

    return X, y, dates


def engineer_features_for_tier(sensor_data_dict, events_df):
    """
    Run feature engineering on all sensors in a tier.

    Args:
        sensor_data_dict: dict mapping (platform_id, sensor_id) → raw DataFrame
        events_df: Event DataFrame

    Returns:
        dict mapping (platform_id, sensor_id) → (X, y, dates)
    """
    results = {}

    for (platform_id, sensor_id), traffic_df in sensor_data_dict.items():
        key = f"{platform_id}__{sensor_id}"
        logger.info("Feature engineering: %s (%d rows)", key, len(traffic_df))

        X, y, dates = engineer_features_for_sensor(traffic_df, events_df)
        results[(platform_id, sensor_id)] = (X, y, dates)

        logger.info("  → %d samples, %d features", X.shape[0], X.shape[1])

    return results
