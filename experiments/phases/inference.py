"""
Phase 4: Inference benchmarks — model load time, single/batch/sustained prediction latency.
"""

import os
import time
import logging

import numpy as np
import joblib

from ..config import SINGLE_INFERENCE_REPEATS, SUSTAINED_INFERENCE_STEPS, sensor_key

logger = logging.getLogger(__name__)


def benchmark_inference(platform_id, sensor_id, X_test, models_dir=None,
                        model_path=None, x_scaler_path=None, y_scaler_path=None):
    """
    Run all inference scenarios for one sensor.

    Supports two modes:
    1. models_dir: auto-discovers model/scaler files by sensor key convention
    2. Explicit paths: model_path, x_scaler_path, y_scaler_path (for pretrained models)

    Args:
        platform_id: Sensor platform identifier
        sensor_id: Sensor direction identifier
        X_test: Unscaled test feature array
        models_dir: Directory containing model/scaler files
        model_path: Explicit path to .keras model
        x_scaler_path: Explicit path to x_scaler .pkl
        y_scaler_path: Explicit path to y_scaler .pkl

    Returns:
        dict with inference latency metrics
    """
    key = sensor_key(platform_id, sensor_id)

    # Resolve file paths
    if model_path is None:
        if models_dir is None:
            raise ValueError("Either models_dir or explicit paths must be provided")
        model_path = os.path.join(models_dir, f"{key}.keras")
        x_scaler_path = os.path.join(models_dir, f"{key}_x_scaler.pkl")
        y_scaler_path = os.path.join(models_dir, f"{key}_y_scaler.pkl")

    if not os.path.exists(model_path):
        logger.warning("Model not found: %s — skipping inference", model_path)
        return {"error": "model_not_found", "model_path": model_path}

    # 1. Model load time (cold start)
    import tensorflow as tf

    t0 = time.perf_counter()
    model = tf.keras.models.load_model(model_path)
    model_load_ms = (time.perf_counter() - t0) * 1000

    # Load scalers
    x_scaler = joblib.load(x_scaler_path)
    y_scaler = joblib.load(y_scaler_path)

    # Scale test data
    X_test_scaled = x_scaler.transform(X_test)

    # Warm-up prediction (first predict has overhead)
    model.predict(X_test_scaled[:1], verbose=0)

    # 2. Single prediction latency (average over N repeats)
    single_sample = X_test_scaled[:1]
    t0 = time.perf_counter()
    for _ in range(SINGLE_INFERENCE_REPEATS):
        model.predict(single_sample, verbose=0)
    single_total = time.perf_counter() - t0
    single_prediction_ms = (single_total / SINGLE_INFERENCE_REPEATS) * 1000

    # 3. Batch prediction (full test set at once)
    t0 = time.perf_counter()
    model.predict(X_test_scaled, verbose=0)
    batch_prediction_ms = (time.perf_counter() - t0) * 1000

    # 4. Sustained inference (288 sequential predictions = 1 day of 5-min intervals)
    n_steps = min(SUSTAINED_INFERENCE_STEPS, len(X_test_scaled))
    t0 = time.perf_counter()
    for i in range(n_steps):
        model.predict(X_test_scaled[i:i + 1], verbose=0)
    sustained_ms = (time.perf_counter() - t0) * 1000

    # Peak memory during inference
    import psutil
    inference_rss_mb = psutil.Process().memory_info().rss / (1024 ** 2)

    result = {
        "platform_id": platform_id,
        "sensor_id": sensor_id,
        "model_load_ms": round(model_load_ms, 2),
        "single_prediction_ms": round(single_prediction_ms, 3),
        "batch_prediction_ms": round(batch_prediction_ms, 2),
        "batch_size": len(X_test_scaled),
        "sustained_288_ms": round(sustained_ms, 2),
        "sustained_steps": n_steps,
        "inference_rss_mb": round(inference_rss_mb, 2),
    }

    logger.info(
        "  %s: load=%.0fms, single=%.1fms, batch=%.0fms, sustained(%d)=%.0fms",
        key, model_load_ms, single_prediction_ms, batch_prediction_ms,
        n_steps, sustained_ms,
    )

    return result
