# Paper Artifacts

Reviewer-facing materials for the paper **"Cloud-Fog-Edge Deployment of ML-Based Traffic
Speed Prediction: An Empirical Analysis of Performance, Cost, and Scalability"**.

| Item | Description |
|---|---|
| `trained_models_base3y/` | The 37 per-sensor-direction MLP models (`.keras`) with their fitted input/output RobustScalers (`.pkl`), trained on the initial three-year window (Dec 2020 – Dec 2023). These are the base models from which all warm-start re-training scenarios in the paper are initialised. |
| `Model_Evaluation.pdf` | Evaluation of the trained models: protocol, per-sensor MAE/RMSE/MAPE on the held-out test set, and the common-test-set comparison of all training scenarios. |
| `Observed_Monitoring_Metrics.pdf` | Resource-utilisation, thermal, and power metrics observed during the experiments across all six platforms, with collection methodology and reliability caveats. |
| `SHA256_MANIFEST.txt` | SHA-256 checksums of every model, scaler, and document in this directory. |

Markdown sources of both PDFs are included (`model_evaluation.md`,
`observed_monitoring_metrics.md`).

## Loading a model

```python
import pickle
from tensorflow import keras

model = keras.models.load_model("trained_models_base3y/drakewell__1163__avgspeed_nw.keras")
x_scaler = pickle.load(open("trained_models_base3y/drakewell__1163__avgspeed_nw_x_scaler.pkl", "rb"))
y_scaler = pickle.load(open("trained_models_base3y/drakewell__1163__avgspeed_nw_y_scaler.pkl", "rb"))

# X: (n_samples, 15) engineered features — see the paper and experiment code
y_pred = y_scaler.inverse_transform(model.predict(x_scaler.transform(X)))
```

The feature-engineering pipeline that produces the 15 input features is in
`experiments/phases/feature_eng.py` and
`model/feature_engineering.py` in this repository.
