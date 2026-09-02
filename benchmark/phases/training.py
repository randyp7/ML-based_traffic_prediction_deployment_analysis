"""
Phase 3: Model training — build, train, and evaluate one model per sensor.

Uses the fixed benchmark MLP architecture (128-128-64-1 with dropout).
No hyperparameter tuning — deterministic across all platforms.
"""

import os
import logging

import numpy as np
import joblib
from sklearn.preprocessing import RobustScaler

from ..config import (
    MODEL_LAYERS, MODEL_OPTIMIZER, MODEL_LOSS,
    BATCH_SIZE, EPOCHS, PATIENCE, VALIDATION_SPLIT,
    TRAIN_TEST_SPLIT, RANDOM_SEED, MS_TO_MPH, sensor_key,
)

logger = logging.getLogger(__name__)


def build_model(input_dim):
    """Build the fixed benchmark MLP architecture."""
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Input

    tf.random.set_seed(RANDOM_SEED)

    layers = [Input(shape=(input_dim,))]
    for layer_cfg in MODEL_LAYERS:
        layers.append(Dense(layer_cfg["units"], activation=layer_cfg["activation"]))
        layers.append(Dropout(layer_cfg["dropout"]))
    layers.append(Dense(1))

    model = Sequential(layers)
    model.compile(optimizer=MODEL_OPTIMIZER, loss=MODEL_LOSS)
    return model


def prepare_data(X, y):
    """
    Scale data with RobustScaler and perform 80/20 chronological split.

    Returns:
        X_train, X_test, y_train, y_test, x_scaler, y_scaler
    """
    x_scaler = RobustScaler()
    y_scaler = RobustScaler()

    X_scaled = x_scaler.fit_transform(X)
    y_scaled = y_scaler.fit_transform(y)

    split_index = int(len(X_scaled) * TRAIN_TEST_SPLIT)
    X_train = X_scaled[:split_index]
    X_test = X_scaled[split_index:]
    y_train = y_scaled[:split_index]
    y_test = y_scaled[split_index:]

    return X_train, X_test, y_train, y_test, x_scaler, y_scaler


def train_single_sensor(X, y, platform_id, sensor_id, models_dir):
    """
    Train one model for one sensor and evaluate on test set.

    Args:
        X: Feature array (unscaled)
        y: Target array (unscaled, m/s)
        platform_id: Sensor platform identifier
        sensor_id: Sensor direction identifier
        models_dir: Path to save .keras model and .pkl scalers

    Returns:
        dict with training results, accuracy metrics, and file paths
    """
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping

    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    key = sensor_key(platform_id, sensor_id)

    # Prepare data
    X_train, X_test, y_train, y_test, x_scaler, y_scaler = prepare_data(X, y)

    # Build model
    model = build_model(X_train.shape[1])

    # Train
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=0,
    )

    import time
    t0 = time.perf_counter()
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        callbacks=[early_stopping],
        verbose=0,
    )
    training_sec = time.perf_counter() - t0

    epochs_completed = len(history.history['loss'])
    val_losses = history.history['val_loss']
    best_epoch = int(np.argmin(val_losses) + 1)

    # Evaluate
    y_pred = model.predict(X_test, verbose=0)
    y_test_unscaled = y_scaler.inverse_transform(y_test).flatten()
    y_pred_unscaled = y_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    # Convert m/s to mph
    y_test_mph = y_test_unscaled * MS_TO_MPH
    y_pred_mph = y_pred_unscaled * MS_TO_MPH

    from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
    mae = mean_absolute_error(y_test_mph, y_pred_mph)
    mse = mean_squared_error(y_test_mph, y_pred_mph)
    mape = mean_absolute_percentage_error(y_test_mph, y_pred_mph)
    rmse = np.sqrt(mse)

    # Save model and scalers
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, f"{key}.keras")
    x_scaler_path = os.path.join(models_dir, f"{key}_x_scaler.pkl")
    y_scaler_path = os.path.join(models_dir, f"{key}_y_scaler.pkl")

    model.save(model_path)
    joblib.dump(x_scaler, x_scaler_path)
    joblib.dump(y_scaler, y_scaler_path)

    model_size_mb = os.path.getsize(model_path) / (1024 ** 2)

    result = {
        "platform_id": platform_id,
        "sensor_id": sensor_id,
        "rows": len(X),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "features_count": X_train.shape[1],
        "training_sec": round(training_sec, 4),
        "epochs_completed": epochs_completed,
        "best_epoch": best_epoch,
        "final_train_loss": float(history.history['loss'][-1]),
        "final_val_loss": float(val_losses[-1]),
        "mae_mph": round(mae, 4),
        "rmse_mph": round(rmse, 4),
        "mape_pct": round(mape * 100, 2),
        "model_size_mb": round(model_size_mb, 3),
        "model_path": model_path,
        "x_scaler_path": x_scaler_path,
        "y_scaler_path": y_scaler_path,
    }

    logger.info(
        "  %s: %d epochs, MAPE=%.2f%%, MAE=%.2f mph, time=%.1fs",
        key, epochs_completed, mape * 100, mae, training_sec,
    )

    return result
