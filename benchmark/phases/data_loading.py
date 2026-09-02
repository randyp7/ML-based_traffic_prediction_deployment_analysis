"""
Phase 1: Load pre-exported Parquet data for a sensor tier.
"""

import json
import logging

import pandas as pd

from ..config import TRAFFIC_DIR, EVENTS_FILE, MANIFEST_FILE, sensor_filename

logger = logging.getLogger(__name__)


def load_sensor_data(sensor_list, traffic_dir=None, events_file=None):
    """
    Load raw traffic Parquet files for the given sensor tier + events.

    Args:
        sensor_list: list of (platform_id, sensor_id) tuples
        traffic_dir: Path to traffic parquet directory (default: config.TRAFFIC_DIR)
        events_file: Path to events parquet file (default: config.EVENTS_FILE)

    Returns:
        sensor_data: dict mapping (platform_id, sensor_id) → pd.DataFrame
        events_df: pd.DataFrame of event data
    """
    if traffic_dir is None:
        traffic_dir = TRAFFIC_DIR
    if events_file is None:
        events_file = EVENTS_FILE

    # Load manifest for validation if available
    manifest_rows = {}
    if MANIFEST_FILE.exists():
        with open(MANIFEST_FILE) as f:
            manifest = json.load(f)
        for s in manifest.get("sensors", []):
            key = (s["platform_id"], s["sensor_id"])
            manifest_rows[key] = s["rows"]

    sensor_data = {}
    total_rows = 0

    for platform_id, sensor_id in sensor_list:
        fname = sensor_filename(platform_id, sensor_id)
        fpath = traffic_dir / fname

        if not fpath.exists():
            logger.warning("Missing Parquet file: %s — skipping sensor", fname)
            continue

        df = pd.read_parquet(fpath)
        rows = len(df)
        total_rows += rows

        # Validate against manifest
        expected = manifest_rows.get((platform_id, sensor_id))
        if expected is not None and rows != expected:
            logger.warning(
                "Row count mismatch for %s: expected %d, got %d",
                fname, expected, rows,
            )

        sensor_data[(platform_id, sensor_id)] = df

    logger.info(
        "Loaded %d sensors, %d total rows",
        len(sensor_data), total_rows,
    )

    # Load events
    events_df = pd.DataFrame()
    if events_file.exists():
        events_df = pd.read_parquet(events_file)
        logger.info("Loaded %d event rows", len(events_df))
    else:
        logger.warning("Events file not found: %s", events_file)

    return sensor_data, events_df
