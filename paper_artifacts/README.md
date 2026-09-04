# Paper Artifacts

Reviewer-facing materials for the paper **"Cloud-Fog-Edge Deployment of ML-Based Traffic
Speed Prediction: An Empirical Analysis of Performance, Cost, and Scalability"**.

| Item | Description |
|---|---|
| `trained_models_full42y/` | The 37 per-sensor-direction MLP models (`.keras`) with their fitted input/output RobustScalers (`.pkl`), trained on the full data span (Dec 2020 – Feb 2025). These are the headline models evaluated in `Model_Evaluation.pdf`, matching the paper's ~12% aggregate MAPE. |
| `trained_models_base3y/` | The three-year base models (trained to Dec 2023) from which all warm-start re-training scenarios in the paper are initialised; included so the re-training experiments are reproducible. |
| `Model_Evaluation.pdf` | Evaluation of the trained models: protocol, per-sensor MAE/RMSE/MAPE on the held-out test set, and the common-test-set comparison of all training scenarios. |
| `Observed_Monitoring_Metrics.pdf` | Resource-utilisation, thermal, and power metrics observed during the experiments across all six platforms, with collection methodology and reliability caveats. |
| `SHA256_MANIFEST.txt` | SHA-256 checksums of every model, scaler, and document in this directory. |

LaTeX sources of both PDFs and their figures are in `sources/`.

## Loading a model

```python
import pickle
from tensorflow import keras

model = keras.models.load_model("trained_models_full42y/drakewell__1163__avgspeed_nw.keras")
x_scaler = pickle.load(open("trained_models_full42y/drakewell__1163__avgspeed_nw_x_scaler.pkl", "rb"))
y_scaler = pickle.load(open("trained_models_full42y/drakewell__1163__avgspeed_nw_y_scaler.pkl", "rb"))

# X: (n_samples, 15) engineered features — see the paper and experiment code
y_pred = y_scaler.inverse_transform(model.predict(x_scaler.transform(X)))
```

The feature-engineering pipeline that produces the 15 input features is in
`experiments/phases/feature_eng.py` and
`model/feature_engineering.py` in this repository.

## Rebuilding the PDFs

Both documents are maintained as LaTeX in `sources/` and built with pdflatex (run 2--3 times; longtable needs multiple passes):

```bash
cd sources
pdflatex model_evaluation.tex && pdflatex model_evaluation.tex
pdflatex observed_monitoring_metrics.tex && pdflatex observed_monitoring_metrics.tex
mv model_evaluation.pdf ../Model_Evaluation.pdf
mv observed_monitoring_metrics.pdf ../Observed_Monitoring_Metrics.pdf
```
