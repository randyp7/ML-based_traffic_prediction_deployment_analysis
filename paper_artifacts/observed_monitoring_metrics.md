# Observed Monitoring Metrics — ML-Based Traffic Speed Prediction

Companion document to the paper "Cloud-Fog-Edge Deployment of ML-Based Traffic Speed Prediction: An Empirical Analysis of Performance, Cost, and Scalability". It reports the resource-utilisation, thermal, and power metrics observed during the experiments, which the paper references but does not present in full.

## Collection methodology

A background monitoring process sampled system metrics every 0.5 seconds during each experiment. CPU variables: system-wide utilisation (%), per-process utilisation (%), operating frequency (MHz), package temperature (deg C), and — where accessible — power (W) via Intel RAPL. GPU variables (via nvidia-smi, on GPU-equipped platforms only): utilisation (%), memory use (MB), temperature (deg C), and power (W).

## Reliability caveats (important)

1. **Attribution.** CPU-system and GPU readings are whole-node/whole-device: on shared or virtualised platforms (CSF3 nodes, AWS EC2) they may include activity not attributable to the experiment process. They should be read as qualitative context, not precise per-process measurements; repetition of runs does not remove this systematic attribution error. Only the Dell Precision platform (dedicated hardware, no virtualisation) provides fully representative CPU and GPU readings.
2. **Power.** CPU power via Intel RAPL was accessible only on the Dell Precision (average CPU power 10-46 W across tiers; 0.1-8.7 Wh per training run). On CSF3 the powercap energy counters are root-restricted; AWS EC2 does not expose hardware power to guests. GPU power (nvidia-smi) is available on the GPU platforms.
3. **Diagnostic value.** The telemetry proved essential for outlier attribution: inflated fog-platform runs in an early campaign showed mean CPU clocks of 0.5-1.6 GHz at low package temperature — the signature of an OS power-saving state, not thermal throttling or contention — and were re-measured on mains power. On AWS t2.medium, elevated steal time and run-position-dependent slowdowns identified burstable CPU-credit exhaustion.

## Observed metrics by platform and sensor tier

Values aggregate the repeated runs of each platform's initial-training batch: mean of per-run averages; "max" is the maximum across runs. Empty cells: metric not available on that platform.

### CSF3 Cloud GPU (A100)

| Tier | CPU sys avg % | CPU proc avg % | CPU freq avg MHz | CPU temp avg/max C | GPU util avg % | GPU mem avg MB | GPU power avg/max W | GPU temp avg/max C |
|---|---|---|---|---|---|---|---|---|
| tiny | 19.4 | 147.2 | 2,892 | 45.0/46.8 | 6.1 | 16,730 | 65.2/83.2 | 25.0/26.0 |
| small | 16.7 | 134.7 | 2,889 | 44.7/48.0 | 7.5 | 16,834 | 65.3/67.8 | 25.6/26.0 |
| medium | 16.6 | 130.1 | 2,887 | 43.7/47.2 | 7.5 | 16,861 | 65.0/81.9 | 24.8/26.0 |
| full | 15.7 | 125.2 | 2,873 | 42.6/45.8 | 7.2 | 16,907 | 64.9/68.1 | 24.5/25.0 |

### Dell Precision Fog GPU (RTX 4090)

| Tier | CPU sys avg % | CPU proc avg % | CPU freq avg MHz | CPU temp avg/max C | GPU util avg % | GPU mem avg MB | GPU power avg/max W | GPU temp avg/max C |
|---|---|---|---|---|---|---|---|---|
| tiny | 13.1 | 180.0 | 1,967 | 83.3/94.0 | 5.4 | 14,373 | 28.7/31.9 | 56.1/58.0 |
| small | 10.9 | 150.2 | 2,189 | 86.3/100.0 | 7.4 | 14,450 | 30.3/32.5 | 59.2/60.0 |
| medium | 11.1 | 152.0 | 2,163 | 86.6/100.0 | 6.9 | 14,477 | 30.7/33.4 | 59.7/62.0 |
| full | 10.0 | 134.1 | 2,127 | 86.3/100.0 | 6.7 | 14,607 | 30.4/34.0 | 59.6/61.0 |

### CSF3 Jetson Sim

| Tier | CPU sys avg % | CPU proc avg % | CPU freq avg MHz | CPU temp avg/max C | GPU util avg % | GPU mem avg MB | GPU power avg/max W | GPU temp avg/max C |
|---|---|---|---|---|---|---|---|---|
| tiny | 9.5 | 127.2 | 2,785 | 47.7/49.5 | 4.8 | 8,596 | 68.1/70.4 | 25.9/26.0 |
| small | 9.3 | 123.5 | 2,801 | 49.4/51.0 | 6.3 | 8,643 | 68.7/81.1 | 26.5/27.0 |

### CSF3 Edge RSU Sim

| Tier | CPU sys avg % | CPU proc avg % | CPU freq avg MHz | CPU temp avg/max C | GPU util avg % | GPU mem avg MB | GPU power avg/max W | GPU temp avg/max C |
|---|---|---|---|---|---|---|---|---|
| tiny | 62.4 | 93.1 | 2,250 | 84.2/85.0 | — | — | —/— | —/— |
| small | 60.5 | 90.8 | 2,250 | 84.1/86.2 | — | — | —/— | —/— |

### CSF3 RPi Sim

| Tier | CPU sys avg % | CPU proc avg % | CPU freq avg MHz | CPU temp avg/max C | GPU util avg % | GPU mem avg MB | GPU power avg/max W | GPU temp avg/max C |
|---|---|---|---|---|---|---|---|---|
| tiny | 85.9 | 131.5 | 2,250 | 84.7/86.0 | — | — | —/— | —/— |
| small | 78.0 | 138.2 | 2,250 | 84.3/86.6 | — | — | —/— | —/— |

### AWS t2.medium RSU

| Tier | CPU sys avg % | CPU proc avg % | CPU freq avg MHz | CPU temp avg/max C | GPU util avg % | GPU mem avg MB | GPU power avg/max W | GPU temp avg/max C |
|---|---|---|---|---|---|---|---|---|
| tiny | 64.9 | 128.9 | 2,300 | —/— | — | — | —/— | —/— |
| small | 63.6 | 125.9 | 2,300 | —/— | — | — | —/— | —/— |

## Interpretation notes

- CPU process utilisation above 100% indicates multi-threaded data loading and feature engineering phases; training itself is GPU-bound on GPU platforms (GPU utilisation averages are low because per-step kernels for a ~142k-parameter MLP are dominated by launch overhead, consistent with the paper's finding that tensor-core throughput is not the bottleneck).
- On the fog laptop workstation, sustained combined CPU+GPU load settles at reduced CPU clocks (1.2-1.7 GHz) at 80-88 deg C: normal power sharing that does not measurably affect GPU-bound wall-clock times.
- The AWS t2.medium rows reflect a burstable instance: sustained full-CPU work beyond ~1.5 hours exhausts CPU credits and caps the instance at its baseline rate; short workloads run at burst speed. Deployment schedules on burstable instances should be budgeted at baseline speed.
