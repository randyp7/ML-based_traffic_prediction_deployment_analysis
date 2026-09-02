"""
One-time data export: InfluxDB -> Parquet files.

Runs on the development machine (with InfluxDB access) only.
Target machines use the pre-exported Parquet files directly.

Usage:
    python -m benchmark.data_export                                      # Export all 37 sensors
    python -m benchmark.data_export --sensor drakewell__1163 avgspeed_nw # Single sensor
    python -m benchmark.data_export --verify                             # Validate manifest vs files
"""

import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

# Add project root so we can import existing modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))

from model.data_loader import DataLoader
from benchmark.config import (
    TIER_FULL, DATE_RANGE_START, DATE_RANGE_END, BATCH_DAYS,
    EXPECTED_POINTS_PER_SENSOR, TRAFFIC_DIR, EVENTS_FILE,
    MANIFEST_FILE, DATA_DIR, sensor_filename,
)


def export_sensor(loader, platform_id, sensor_id, start_date, end_date):
    """Export one sensor's traffic data to Parquet."""
    print(f"\n  Exporting {platform_id} / {sensor_id}...")
    traffic = loader.batch_query_traffic(start_date, end_date, BATCH_DAYS, platform_id, sensor_id)

    if traffic is None or traffic.empty:
        print(f"    WARNING: No data returned for {platform_id} / {sensor_id}")
        return None

    out_path = TRAFFIC_DIR / sensor_filename(platform_id, sensor_id)
    traffic.to_parquet(out_path, index=False)

    rows = len(traffic)
    completeness = round(rows / EXPECTED_POINTS_PER_SENSOR * 100, 1)
    print(f"    Saved {rows:,} rows ({completeness}% completeness) -> {out_path.name}")

    return {
        "platform_id": platform_id,
        "sensor_id": sensor_id,
        "rows": rows,
        "completeness_pct": completeness,
        "file": f"traffic/{sensor_filename(platform_id, sensor_id)}",
    }


def export_events(loader, start_date, end_date):
    """Export shared event data to Parquet."""
    print("\n  Exporting events...")
    events = loader.query_event_data(start_date, end_date)

    if events is None or (isinstance(events, pd.DataFrame) and events.empty):
        print("    WARNING: No event data returned")
        return {"rows": 0, "file": "events.parquet"}

    # Handle case where query returns a list of DataFrames
    if isinstance(events, list):
        events = pd.concat(events, ignore_index=True)

    events.to_parquet(EVENTS_FILE, index=False)
    rows = len(events)
    print(f"    Saved {rows:,} event rows -> {EVENTS_FILE.name}")

    return {"rows": rows, "file": "events.parquet"}


def export_all(sensors=None):
    """Export all sensors and events to Parquet, generate manifest."""
    start_date = datetime.fromisoformat(DATE_RANGE_START)
    end_date = datetime.fromisoformat(DATE_RANGE_END)

    if sensors is None:
        sensors = TIER_FULL

    # Create output directories
    TRAFFIC_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Benchmark Data Export")
    print(f"  Date range: {DATE_RANGE_START} to {DATE_RANGE_END}")
    print(f"  Sensors to export: {len(sensors)}")
    print(f"  Output: {DATA_DIR}")

    loader = DataLoader()
    manifest_sensors = []

    try:
        # Export each sensor
        for platform_id, sensor_id in sensors:
            result = export_sensor(loader, platform_id, sensor_id, start_date, end_date)
            if result:
                manifest_sensors.append(result)

        # Export events
        events_info = export_events(loader, start_date, end_date)
    finally:
        loader.close()

    # Generate manifest
    manifest = {
        "export_timestamp": datetime.now().isoformat(),
        "date_range": {"start": DATE_RANGE_START, "end": DATE_RANGE_END},
        "expected_points_per_sensor": EXPECTED_POINTS_PER_SENSOR,
        "sensors": manifest_sensors,
        "events": events_info,
    }

    with open(MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n  Manifest saved -> {MANIFEST_FILE.name}")
    print(f"\nExport complete: {len(manifest_sensors)} sensors + events")

    total_rows = sum(s["rows"] for s in manifest_sensors)
    print(f"Total traffic rows: {total_rows:,}")

    return manifest


def verify_manifest():
    """Validate that all files listed in manifest exist and have correct row counts."""
    if not MANIFEST_FILE.exists():
        print(f"ERROR: Manifest not found at {MANIFEST_FILE}")
        return False

    with open(MANIFEST_FILE) as f:
        manifest = json.load(f)

    ok = True
    print(f"Verifying manifest ({len(manifest['sensors'])} sensors)...\n")

    for sensor in manifest["sensors"]:
        fpath = DATA_DIR / sensor["file"]
        if not fpath.exists():
            print(f"  MISSING: {sensor['file']}")
            ok = False
            continue

        df = pd.read_parquet(fpath)
        actual_rows = len(df)
        expected_rows = sensor["rows"]

        if actual_rows != expected_rows:
            print(f"  MISMATCH: {sensor['file']} — expected {expected_rows}, got {actual_rows}")
            ok = False
        else:
            print(f"  OK: {sensor['file']} ({actual_rows:,} rows)")

    # Check events
    if EVENTS_FILE.exists():
        events_df = pd.read_parquet(EVENTS_FILE)
        print(f"  OK: events.parquet ({len(events_df):,} rows)")
    else:
        print(f"  MISSING: events.parquet")
        ok = False

    print(f"\nVerification: {'PASSED' if ok else 'FAILED'}")
    return ok


def main():
    parser = argparse.ArgumentParser(description="Export InfluxDB data to Parquet for benchmark")
    parser.add_argument("--sensor", nargs=2, metavar=("PLATFORM_ID", "SENSOR_ID"),
                        help="Export a single sensor (e.g., --sensor drakewell__1163 avgspeed_nw)")
    parser.add_argument("--verify", action="store_true",
                        help="Validate manifest against existing Parquet files")

    args = parser.parse_args()

    if args.verify:
        verify_manifest()
    elif args.sensor:
        export_all(sensors=[(args.sensor[0], args.sensor[1])])
    else:
        export_all()


if __name__ == "__main__":
    main()
