"""
Main CLI orchestrator for the cross-platform benchmark.

Usage:
    # Full benchmark (all tiers, 3 runs each)
    python -m experiments.runner --platform dell_precision_fog_gpu --runs 3

    # Quick validation
    python -m experiments.runner --platform local --tiers tiny --runs 1

    # Specific tiers
    python -m experiments.runner --platform hp_workstation_fog --tiers tiny small medium --runs 3

    # Inference only (for platforms where training OOMs)
    python -m experiments.runner --platform aws_t2_rsu --inference-only --pretrained-dir ./pretrained_models/ --runs 3
"""

import os
import sys
import json
import logging
import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from .config import (
    TIERS, PLATFORMS, RESULTS_DIR, RANDOM_SEED,
    sensor_key, TRAIN_TEST_SPLIT,
    RETRAIN_WINDOWS, RETRAIN_WINDOWS_PROGRESSIVE,
    INITIAL_TRAIN_DATE_FROM, INITIAL_TRAIN_DATE_TO,
)
from .metrics import PhaseTimer, CPUMonitor, GPUMonitor, get_system_info
from .phases.data_loading import load_sensor_data
from .phases.feature_eng import engineer_features_for_tier
from .phases.training import train_single_sensor, retrain_single_sensor, prepare_data
from .phases.inference import benchmark_inference

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def set_seeds():
    """Set random seeds for reproducibility."""
    np.random.seed(RANDOM_SEED)
    try:
        import tensorflow as tf
        tf.random.set_seed(RANDOM_SEED)
    except ImportError:
        pass


def run_tier(tier_name, sensor_list, run_id, platform_name, run_dir,
             inference_only=False, pretrained_dir=None,
             mode="scratch", window=None, date_from=None, date_to=None):
    """
    Run a full benchmark for one tier in one run.

    Args:
        mode: "scratch" (default) or "retrain"
        window: Window ID ("W3", "W6", "W12") when mode=retrain
        date_from: Optional date filter for data loading
        date_to: Optional date filter for data loading

    Returns:
        dict: Tier-run results (JSON-serialisable)
    """
    set_seeds()

    models_dir = os.path.join(run_dir, "models")
    experiment_id = f"{platform_name}-{tier_name}-run{run_id}"

    logger.info("=" * 60)
    logger.info("Starting: %s (%d sensors) [mode=%s]", experiment_id, len(sensor_list), mode)
    logger.info("=" * 60)

    result = {
        "experiment_id": experiment_id,
        "platform": platform_name,
        "platform_tier": PLATFORMS.get(platform_name, "unknown"),
        "sensor_tier": tier_name,
        "sensor_count": len(sensor_list),
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "mode": mode,
        "phases": {},
        "per_sensor_results": [],
        "errors": [],
    }

    # Add retrain-specific metadata
    if mode == "retrain" and window:
        result["update_window"] = window
        result["window_type"] = "progressive" if window.startswith("PW") else "recent"
        result["window_date_from"] = date_from
        result["window_date_to"] = date_to
        result["pretrained_dir"] = pretrained_dir
    if date_from or date_to:
        result["data_date_from"] = date_from
        result["data_date_to"] = date_to

    # ---- Phase 1: Data Loading ----
    if date_from or date_to:
        logger.info("[1/4] Loading Parquet data (filtered: %s to %s)...", date_from, date_to)
    else:
        logger.info("[1/4] Loading Parquet data...")
    try:
        with PhaseTimer("data_loading") as dt:
            sensor_data, events_df = load_sensor_data(
                sensor_list, date_from=date_from, date_to=date_to,
            )

        total_rows = sum(len(df) for df in sensor_data.values())
        phase_info = dt.to_dict()
        phase_info["total_rows_loaded"] = total_rows
        phase_info["sensors_loaded"] = len(sensor_data)
        result["phases"]["data_loading"] = phase_info

        logger.info(
            "  Data loaded: %d sensors, %d rows in %.2fs",
            len(sensor_data), total_rows, dt.wall_clock_sec,
        )
    except (MemoryError, Exception) as e:
        logger.error("Data loading failed: %s", e)
        result["errors"].append({
            "phase": "data_loading", "error": type(e).__name__, "detail": str(e),
        })
        result["total_wall_clock_sec"] = 0
        return result

    # ---- Phase 2: Feature Engineering ----
    logger.info("[2/4] Feature engineering...")
    try:
        with PhaseTimer("feature_engineering") as ft:
            fe_results = engineer_features_for_tier(sensor_data, events_df)

        phase_info = ft.to_dict()
        phase_info["sensors_processed"] = len(fe_results)
        result["phases"]["feature_engineering"] = phase_info

        logger.info(
            "  Feature engineering done: %d sensors in %.2fs",
            len(fe_results), ft.wall_clock_sec,
        )
    except (MemoryError, Exception) as e:
        logger.error("Feature engineering failed: %s", e)
        result["errors"].append({
            "phase": "feature_engineering", "error": type(e).__name__, "detail": str(e),
        })
        result["total_wall_clock_sec"] = sum(
            p.get("wall_clock_sec", 0) for p in result["phases"].values()
        )
        return result

    # Free raw data from memory
    del sensor_data, events_df

    # ---- Phase 3: Training ----
    if not inference_only:
        logger.info("[3/4] Training models (%s mode)...", mode)
        cpu_monitor = CPUMonitor()
        gpu_monitor = GPUMonitor()
        cpu_monitor.start()
        gpu_monitor.start()

        try:
            with PhaseTimer("training") as tt:
                for (platform_id, sensor_id), (X, y, dates) in fe_results.items():
                    try:
                        if mode == "retrain" and pretrained_dir:
                            key = sensor_key(platform_id, sensor_id)
                            pt_model = os.path.join(pretrained_dir, "models", f"{key}.keras")
                            pt_x_scaler = os.path.join(pretrained_dir, "models", f"{key}_x_scaler.pkl")
                            pt_y_scaler = os.path.join(pretrained_dir, "models", f"{key}_y_scaler.pkl")
                            sensor_result = retrain_single_sensor(
                                X, y, platform_id, sensor_id, models_dir,
                                pretrained_model_path=pt_model,
                                pretrained_x_scaler_path=pt_x_scaler,
                                pretrained_y_scaler_path=pt_y_scaler,
                            )
                        else:
                            sensor_result = train_single_sensor(
                                X, y, platform_id, sensor_id, models_dir,
                            )
                        result["per_sensor_results"].append(sensor_result)
                    except (MemoryError, Exception) as e:
                        import tensorflow as tf
                        error_name = type(e).__name__
                        if isinstance(e, tf.errors.ResourceExhaustedError):
                            error_name = "OOM_GPU"
                        logger.error("  Training failed for %s__%s: %s",
                                     platform_id, sensor_id, e)
                        result["per_sensor_results"].append({
                            "platform_id": platform_id,
                            "sensor_id": sensor_id,
                            "error": error_name,
                            "detail": str(e),
                        })

            result["phases"]["training"] = tt.to_dict()

        except (MemoryError, Exception) as e:
            logger.error("Training phase failed entirely: %s", e)
            result["errors"].append({
                "phase": "training", "error": type(e).__name__, "detail": str(e),
            })
        finally:
            cpu_stats = cpu_monitor.stop()
            gpu_stats = gpu_monitor.stop()
            result["resource_monitoring"] = {**cpu_stats, **gpu_stats}

        logger.info(
            "  Training done in %.2fs",
            result["phases"].get("training", {}).get("wall_clock_sec", 0),
        )
    else:
        logger.info("[3/4] Skipping training (inference-only mode)")

    # ---- Phase 4: Inference Benchmarks ----
    logger.info("[4/4] Running inference benchmarks...")
    inference_dir = pretrained_dir if inference_only else models_dir

    try:
        with PhaseTimer("inference") as it:
            inference_results = []
            for (platform_id, sensor_id), (X, y, dates) in fe_results.items():
                # Use test split for inference
                split_index = int(len(X) * TRAIN_TEST_SPLIT)
                X_test = X[split_index:]

                try:
                    inf_result = benchmark_inference(
                        platform_id, sensor_id, X_test,
                        models_dir=inference_dir,
                    )
                    inference_results.append(inf_result)

                    # Merge inference metrics into per_sensor_results
                    for sr in result["per_sensor_results"]:
                        if (sr.get("platform_id") == platform_id and
                                sr.get("sensor_id") == sensor_id and
                                "error" not in sr):
                            sr["single_inference_ms"] = inf_result.get("single_prediction_ms")
                            sr["batch_inference_ms"] = inf_result.get("batch_prediction_ms")
                            sr["sustained_288_ms"] = inf_result.get("sustained_288_ms")
                            sr["model_load_ms"] = inf_result.get("model_load_ms")
                            break
                    else:
                        # Sensor wasn't in per_sensor_results (inference-only mode)
                        if inference_only:
                            result["per_sensor_results"].append(inf_result)

                except Exception as e:
                    logger.error("  Inference failed for %s__%s: %s",
                                 platform_id, sensor_id, e)
                    inference_results.append({
                        "platform_id": platform_id,
                        "sensor_id": sensor_id,
                        "error": type(e).__name__,
                        "detail": str(e),
                    })

        result["phases"]["inference"] = it.to_dict()
        result["inference_results"] = inference_results

    except (MemoryError, Exception) as e:
        logger.error("Inference phase failed: %s", e)
        result["errors"].append({
            "phase": "inference", "error": type(e).__name__, "detail": str(e),
        })

    # ---- Aggregate Accuracy ----
    successful = [
        sr for sr in result["per_sensor_results"]
        if "error" not in sr and "mae_mph" in sr
    ]
    if successful:
        result["aggregate_accuracy"] = {
            "mean_mae_mph": round(np.mean([s["mae_mph"] for s in successful]), 4),
            "std_mae_mph": round(np.std([s["mae_mph"] for s in successful]), 4),
            "mean_rmse_mph": round(np.mean([s["rmse_mph"] for s in successful]), 4),
            "mean_mape_pct": round(np.mean([s["mape_pct"] for s in successful]), 2),
        }

    # Total wall clock
    result["total_wall_clock_sec"] = round(
        sum(p.get("wall_clock_sec", 0) for p in result["phases"].values()), 2
    )

    return result


def run_benchmark(platform_name, tier_names, runs, inference_only=False, pretrained_dir=None,
                  mode="scratch", window=None, date_from=None, date_to=None):
    """
    Run the full benchmark suite across specified tiers and runs.
    """
    # Include mode/window in directory name for easy identification
    if mode == "retrain" and window:
        dir_suffix = f"retrain_{window}"
    elif date_from and date_to:
        dir_suffix = "initial"
    else:
        dir_suffix = "scratch"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_DIR / platform_name / f"{timestamp}_{dir_suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save system info
    sys_info = get_system_info()
    sys_info_path = run_dir / "system_info.json"
    with open(sys_info_path, "w") as f:
        json.dump(sys_info, f, indent=2)
    logger.info("System info saved to %s", sys_info_path)

    all_results = []

    for tier_name in tier_names:
        if tier_name not in TIERS:
            logger.error("Unknown tier: %s (available: %s)", tier_name, list(TIERS.keys()))
            continue

        sensor_list = TIERS[tier_name]

        for run_id in range(1, runs + 1):
            logger.info("\n%s", "=" * 70)
            logger.info("TIER: %s | RUN: %d/%d | PLATFORM: %s",
                        tier_name.upper(), run_id, runs, platform_name)
            logger.info("=" * 70)

            result = run_tier(
                tier_name=tier_name,
                sensor_list=sensor_list,
                run_id=run_id,
                platform_name=platform_name,
                run_dir=str(run_dir),
                inference_only=inference_only,
                pretrained_dir=pretrained_dir,
                mode=mode,
                window=window,
                date_from=date_from,
                date_to=date_to,
            )

            # Save per-tier-run result
            result_file = run_dir / f"tier_{tier_name}_run{run_id}.json"
            with open(result_file, "w") as f:
                json.dump(result, f, indent=2, default=str)
            logger.info("Result saved: %s", result_file)

            all_results.append(result)

    # ---- Generate Summary ----
    summary = {
        "platform": platform_name,
        "system_info": sys_info,
        "timestamp": timestamp,
        "tiers_completed": [],
        "tiers_failed": [],
        "per_tier_summary": {},
    }

    for tier_name in tier_names:
        tier_runs = [r for r in all_results if r["sensor_tier"] == tier_name]
        if not tier_runs:
            continue

        successful_runs = [r for r in tier_runs if not r["errors"]]
        if successful_runs:
            summary["tiers_completed"].append(tier_name)
            summary["per_tier_summary"][tier_name] = {
                "runs": len(tier_runs),
                "successful_runs": len(successful_runs),
                "mean_total_sec": round(
                    np.mean([r["total_wall_clock_sec"] for r in successful_runs]), 2
                ),
                "mean_mape": round(
                    np.mean([
                        r.get("aggregate_accuracy", {}).get("mean_mape_pct", 0)
                        for r in successful_runs
                        if r.get("aggregate_accuracy")
                    ]) if any(r.get("aggregate_accuracy") for r in successful_runs) else 0, 2
                ),
                "peak_memory_mb": round(
                    max(
                        r["phases"].get("training", {}).get("peak_memory_mb", 0)
                        for r in successful_runs
                    ), 2
                ),
            }
        else:
            summary["tiers_failed"].append(tier_name)

    summary_file = run_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("\nSummary saved: %s", summary_file)

    # Print summary table
    print_summary(summary)

    return all_results


def print_summary(summary):
    """Print a compact summary table."""
    print("\n" + "=" * 80)
    print(f"BENCHMARK SUMMARY — {summary['platform']}")
    print("=" * 80)
    print(f"{'Tier':<10} {'Runs':<6} {'Time(s)':<10} {'MAPE(%)':<10} {'Peak Mem(MB)':<12}")
    print("-" * 80)

    for tier_name, tier_info in summary.get("per_tier_summary", {}).items():
        print(
            f"{tier_name:<10} "
            f"{tier_info['successful_runs']:<6} "
            f"{tier_info['mean_total_sec']:<10.1f} "
            f"{tier_info['mean_mape']:<10.1f} "
            f"{tier_info['peak_memory_mb']:<12.1f}"
        )

    if summary["tiers_failed"]:
        print(f"\nFailed tiers: {', '.join(summary['tiers_failed'])}")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Cross-platform benchmark for ML traffic prediction",
    )
    parser.add_argument(
        "--platform", type=str, required=True,
        help=f"Platform name (registered: {', '.join(PLATFORMS.keys())})",
    )
    parser.add_argument(
        "--tiers", type=str, nargs="+",
        default=["tiny", "small", "medium", "full"],
        choices=["tiny", "small", "medium", "full"],
        help="Sensor tiers to benchmark (default: all)",
    )
    parser.add_argument(
        "--runs", type=int, default=3,
        help="Number of runs per tier (default: 3)",
    )
    parser.add_argument(
        "--inference-only", action="store_true",
        help="Skip training, run inference with pretrained models",
    )
    parser.add_argument(
        "--pretrained-dir", type=str, default=None,
        help="Directory containing pretrained .keras models and .pkl scalers",
    )
    parser.add_argument(
        "--gpu-memory-mb", type=int, default=None,
        help="Limit GPU memory to this many MB (e.g. 16384 for 16GB)",
    )
    parser.add_argument(
        "--mode", type=str, default="scratch", choices=["scratch", "retrain"],
        help="Training mode: scratch (default) or retrain (warm-start fine-tuning)",
    )
    parser.add_argument(
        "--window", type=str, default=None,
        choices=["W3", "W6", "W12", "PW3", "PW6", "PW12"],
        help="Re-training window: W3/W6/W12 (recent) or PW3/PW6/PW12 (progressive)",
    )
    parser.add_argument(
        "--initial-only", action="store_true",
        help="Train only on the initial 3-year date range (Dec 2020 - Dec 2023)",
    )

    args = parser.parse_args()

    if args.inference_only and not args.pretrained_dir:
        parser.error("--inference-only requires --pretrained-dir")
    if args.mode == "retrain" and not args.window:
        parser.error("--mode=retrain requires --window (W3, W6, W12, PW3, PW6, or PW12)")
    if args.mode == "retrain" and not args.pretrained_dir:
        parser.error("--mode=retrain requires --pretrained-dir")

    # Apply GPU memory limit before any TF operations
    if args.gpu_memory_mb:
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices("GPU")
            if gpus:
                tf.config.set_logical_device_configuration(
                    gpus[0],
                    [tf.config.LogicalDeviceConfiguration(
                        memory_limit=args.gpu_memory_mb
                    )],
                )
                logger.info("GPU memory limited to %d MB", args.gpu_memory_mb)
            else:
                logger.warning("--gpu-memory-mb specified but no GPU found")
        except Exception as e:
            logger.error("Failed to set GPU memory limit: %s", e)

    # Determine date filters based on mode
    date_from = None
    date_to = None
    mode = args.mode
    window = args.window

    if args.initial_only:
        date_from = INITIAL_TRAIN_DATE_FROM
        date_to = INITIAL_TRAIN_DATE_TO
        logger.info("Initial-only mode: training on %s to %s", date_from, date_to)
    elif mode == "retrain" and window:
        if window in RETRAIN_WINDOWS:
            win_cfg = RETRAIN_WINDOWS[window]
        elif window in RETRAIN_WINDOWS_PROGRESSIVE:
            win_cfg = RETRAIN_WINDOWS_PROGRESSIVE[window]
        else:
            parser.error(f"Unknown window: {window}")
        date_from = win_cfg["date_from"]
        date_to = win_cfg["date_to"]
        logger.info("Retrain mode: window=%s (%s to %s)", window, date_from, date_to)

    logger.info("Benchmark starting: platform=%s, tiers=%s, runs=%d, mode=%s",
                args.platform, args.tiers, args.runs, mode)

    run_benchmark(
        platform_name=args.platform,
        tier_names=args.tiers,
        runs=args.runs,
        inference_only=args.inference_only,
        pretrained_dir=args.pretrained_dir,
        mode=mode,
        window=window,
        date_from=date_from,
        date_to=date_to,
    )


if __name__ == "__main__":
    main()
