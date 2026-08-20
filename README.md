# Unified AgTech Edge AI Engine

**Real-time agricultural disease monitoring on resource-constrained edge devices.**

Edge AI platform for commercial greenhouse deployment with zero cloud dependency, developed during
an internship at Urban Farm Tech (Jan–Apr 2026) and running on a Raspberry Pi 5.

The repository hosts two related systems built on a shared deployment stack:

| | Task | Model | Status |
| --- | --- | --- | --- |
| **A. Disease classification** | 3-class whole-frame classification (Background / Healthy / Disease) with temporal debouncing and Telegram alerting | MobileNetV2 → ONNX | Materials for the manuscript described below |
| **B. Pest detection** | 5-class object detection with bounding boxes (Fungal, Leaf Damage, Mealybugs, Miner, Mite) | YOLOv8s → ONNX | Separate line of work |

Both share the same duty-cycle scheduler, ONNX Runtime inference path, CSV telemetry logging and
Telegram alerting layer. Sections 1 and 2 below document them in turn.

---

# 1. Disease classification system (manuscript materials)

Code, raw telemetry and analysis scripts for the manuscript *"Containerized Edge Intelligence for
Greenhouse Disease Monitoring: A Lightweight Classification System with Real-Time Alerting on
Raspberry Pi 5."*

## Contents

```
├── classification.py                 # edge inference + logging + alerting loop
├── Dockerfile                        # containerised runtime (python:3.9-slim-bullseye)
├── Mobilenet.ipynb                   # training / ONNX export notebook
├── DEPLOYMENT_GUIDE.md               # provisioning SOP for a fresh Raspberry Pi
├── new_sd_card_setup.md              # SD-card level setup notes
├── basil_experiments/                # training, ablation and offline analysis scripts
├── scripts/
│   └── analyse_profiling.py          # reproduces manuscript Tables 3 and 4
└── data/
    ├── profiling_runs/               # the four controlled runs of Section 4.3
    │   ├── bare_metal_A/             #   night, native execution
    │   ├── bare_metal_B/             #   day,   native execution
    │   ├── docker_A/                 #   night, containerised
    │   └── docker_B/                 #   day,   containerised
    └── supplementary_session/        # extended duty-cycle session of Section 4.5
```

## Data

### Hardware profiling runs (manuscript Section 4.3)

Four independent three-hour runs under a 60 s active / 15 s sleep duty cycle — two native and two
containerised — interleaved across day and night so that ambient temperature is balanced between
conditions rather than confounded with them. All four runs used identical hardware, the same USB
(V4L2) camera, the same ONNX model file and the same inference script; the only difference was
whether the script executed natively or inside the container defined by `Dockerfile`.

Each run directory contains `basil_data.csv.gz` (per-frame telemetry, gzipped to stay within
GitHub file-size limits), `cycle_events.csv` (duty-cycle transitions with the temperature at each
boundary) and `RUN_INFO.txt` (conditions, start time, hardware, OS and runtime versions, throttling
status). `pandas.read_csv` opens the gzipped files directly; no manual decompression is needed.

`basil_data.csv` columns:

| Column | Meaning |
| --- | --- |
| `Timestamp` | local time, `HH:MM:SS.mmm` |
| `Latency_ms` | ONNX Runtime forward-pass time for the frame |
| `FPS` | instantaneous loop throughput |
| `CPU_%`, `RAM_%` | process-level utilisation |
| `Temp_C` | SoC temperature from `/sys/class/thermal/thermal_zone0` |
| `Throttled` | host throttling flag (`Unknown` inside containers, where `vcgencmd` is unavailable; verified on the host before and after every run) |
| `Predicted_Class`, `Confidence` | classifier output |
| `Bus_V_mV`, `Bus_P_mW` | supply voltage and board-level power draw |
| `Bat_V_mV`, `Bat_I_mA`, `Bat_Pct` | battery state of the UPS module |

Power figures are **board-level** — Raspberry Pi 5, UPS module, USB camera and storage combined,
with the charging input connected — read over I2C from a Waveshare UPS HAT (E). They are an upper
bound on the SoC-attributable difference between conditions and are not comparable with published
SoC-only benchmarks.

Two run-specific notes: `docker_A` overran its intended stop and contains ≈6 h 25 min (308 cycles),
so the analysis script truncates every run to the first 10 800 s to equalise length; timestamps in
`docker_A` are UTC, since the timezone environment variable was only added for the later run. The
analysis works from elapsed time and is unaffected by both.

### Supplementary extended duty-cycle session (manuscript Section 4.5)

`data/supplementary_session/` holds a 3 h 33 min indoor session of 313,011 frames recorded under a
longer 180 s active / 45 s sleep duty cycle. It was recorded to characterise behaviour under longer
uninterrupted inference periods and to collect large-sample out-of-distribution observations, and
is the basis of the residual-class analysis in manuscript Section 5.2.1. The camera was not
directed at basil foliage during this session, so the predicted-class distribution (99.99%
Background, 0.0096% Disease, 0% Healthy) reflects behaviour on an essentially empty scene rather
than classification accuracy on basil.

Because its duty cycle differs from the 60 s/15 s protocol, this session is **not** comparable with
Tables 3 and 4 and is reported separately in the manuscript. The log also predates the addition of
the UPS power columns, so it carries no `Bus_*` or `Bat_*` fields.

### Field imagery

The greenhouse image dataset is **not** included. It is subject to confidentiality arrangements
with the collaborating commercial greenhouse and is available from the corresponding author on
reasonable request. The same applies to the in-situ field session analysed in manuscript
Sections 4.2.1, 4.4 and 5.2.2.

## Reproducing the manuscript tables

```bash
git clone https://github.com/xiaolin200206/unified-agtech-engine.git
cd unified-agtech-engine
pip install pandas
python3 scripts/analyse_profiling.py
```

This prints the per-run aggregates of Table 3, the condition means of Table 4, and the day/night
ambient check quoted in Section 4.3. The script truncates each run to three hours, discards the
first ten minutes as thermal warm-up, and takes cyclic peak temperatures from the
`CYCLE_SLEEP_START` events.

Expected condition means:

| Metric | Native | Docker | Delta |
| --- | --- | --- | --- |
| Inference latency (ms) | 18.20 ± 0.07 | 26.50 ± 0.07 | +45.6% |
| CPU utilisation (%) | 49.02 ± 0.79 | 39.79 ± 2.99 | −9.23 pp |
| RAM utilisation (%) | 7.66 ± 0.05 | 7.88 ± 0.06 | +0.22 pp |
| Mean SoC temperature (°C) | 64.18 ± 0.54 | 60.64 ± 0.57 | −3.55 °C |
| Mean cyclic peak temperature (°C) | 65.16 ± 0.55 | 61.46 ± 0.83 | −3.71 °C |
| Board power draw (W) | 10.20 ± 0.03 | 9.73 ± 0.27 | −4.6% |

Containerisation lowers processor load, operating temperature and board-level power draw, and
raises per-frame inference latency. Every effect direction replicates across both pairs. The
day/night interleaving puts the ambient contribution at ≈1.1 °C in both conditions, roughly a third
of the 3.55 °C difference attributed to containerisation.

> Earlier revisions of this repository quoted a larger containerisation benefit (≈49 percentage
> points of CPU and ≈15 °C) from a single unmatched pair of runs. Those figures are superseded by
> the four matched runs above and should not be used.

## Running the classification node

Native:

```bash
python3 -m venv --system-site-packages ~/venv
~/venv/bin/pip install onnxruntime numpy opencv-python-headless psutil requests smbus2
python3 classification.py
```

Containerised:

```bash
docker build -t basil-edge .
docker run --rm \
  --device /dev/video0:/dev/video0 \
  --device /dev/i2c-1 \
  -v /sys:/sys:ro \
  -e TZ=Asia/Kuala_Lumpur \
  -v "$PWD/logs:/app/basil_logs" \
  basil-edge
```

`--device /dev/i2c-1` is only needed if a UPS module is present and power telemetry is wanted.
Passing the video device into the container monopolises it, so a second camera container cannot run
concurrently.

Configuration lives at the top of `classification.py`:

| Parameter | Default | Description |
| --- | --- | --- |
| `CLASSIFICATION_THRESHOLD` | 0.70 | Minimum confidence τ before a prediction is accepted |
| `INFERENCE_SIZE` | 224 | Input resolution for MobileNetV2 |
| `CYCLE_ACTIVE_SEC` | 60 | Active inference duration per cycle |
| `CYCLE_SLEEP_SEC` | 15 | Sleep duration per cycle |
| `MAX_TEMP_LIMIT` | 82.0 °C | SoC throttling threshold used for monitoring |
| `TELEGRAM_COOLDOWN_SEC` | 60.0 | Minimum interval between alerts |
| `SAVE_IMG_INTERVAL` | 2.0 s | Minimum interval between saved frames |

The model file (`basil_mobilenet.onnx`) is not included; export it with `Mobilenet.ipynb` or
`basil_experiments/04_deployment/convert_to_onnx.py`.

## Experiment scripts

`basil_experiments/` contains the training, ablation and analysis scripts referenced in the
manuscript — the 11-architecture Real-Only vs Real+Proxy comparison, the temporal smoothing
ablation, and the OOD proxy analysis. See `basil_experiments/README.md` for a script-to-section
mapping. Scripts are provided with English and Chinese comments (`en/`, `zh/`).

## Known limitations

Documented in full in Section 5.4 of the manuscript. In brief: field data come from a single
commercial greenhouse and the in-situ session is short; the profiling comparison rests on two runs
per condition, which establishes effect direction but does not support significance testing;
profiling used a USB camera in a laboratory setting rather than the Camera Module 3 in the
greenhouse enclosure, so absolute values are configuration-specific while the between-condition
comparison remains internally valid; only ONNX Runtime was benchmarked; and the classifier exhibits
an out-of-distribution failure mode in which novel non-plant objects are absorbed into the Disease
class rather than rejected.

---

# 2. Pest detection system (YOLOv8)

A separate line of work targeting a different task: localising and counting pest damage across
multiple plants from a fixed overhead camera, rather than assigning a single label to the frame.

## What it does

- Captures live video from a Raspberry Pi Camera Module 3
- Runs YOLOv8s ONNX inference at ~5.6 FPS on CPU (no GPU required)
- Detects 5 basil disease/pest classes: Fungal, Leaf Damage, Mealybugs, Miner, Mite
- Sends Telegram alerts with annotated images on detection
- Manages thermal load with a 180 s / 45 s duty cycle
- Logs detections, latency, temperature and CPU metrics to CSV

## Why detection rather than classification for this task

The two systems answer different questions, and the choice of model follows from the question:

| | Classification (Part 1) | Detection (Part 2) |
| --- | --- | --- |
| Model | MobileNetV2 | YOLOv8s |
| Input | 224×224 whole frame | 640×640 frame |
| Output | Class label + confidence | Bounding boxes + labels |
| Answers | *is disease present in view?* | *where, and how many instances?* |
| Suited to | a fixed camera monitoring one canopy section for onset | a fixed camera surveying multiple plants for localisation and counting |

Whole-frame classification cannot localise, and downsampling a full-HD frame to 224×224 costs the
spatial detail needed for physically small targets such as mite spotting and mealybug colonies.
Detection addresses both, at the cost of a heavier model and lower frame rate. The deployment stack
— duty-cycle scheduler, ONNX Runtime session, CSV telemetry, Telegram webhook — is shared, so
either model can be dropped into the same pipeline.

## Detection classes

| Class | Description |
| --- | --- |
| Fungal | Fungal leaf infections |
| Leaf Damage | Physical or environmental leaf damage |
| Mealybugs | Mealybug pest infestation |
| Miner | Leaf miner pest damage |
| Mite | Spider mite damage (small white spots) |

## Architecture

```
Camera (1080p) → Frame Capture → YOLOv8s ONNX Inference
                                        ↓
                              Confidence Thresholding (>= 0.70)
                                        ↓
                    +-------------------+-------------------+
                    |                                       |
            CSV Logging                          Telegram Alert
        (latency, FPS, temp)                 (annotated image + metadata)
```

Duty cycle: `[Active 180s] -> [Sleep 45s] -> [Active 180s] -> ...`, chosen to keep the SoC clear of
the 82 °C throttling threshold, above which inference latency becomes unpredictable.

## Known limitations

- **Mite and mealybug detection**: performance degrades beyond ~30 cm due to target size. Camera
  Module 3 autofocus helps but does not fully resolve this; close-range inspection or
  higher-resolution capture is recommended for these classes.
- **OOD rejection**: limited rejection of out-of-distribution inputs (reflective clothing, dramatic
  lighting changes). Fixed camera positioning mitigates but does not remove this.
- **Single crop**: trained on basil only; other crops require retraining.

## Training pipeline

```
Field data collection (greenhouse, 3 weeks)
        |
Manual annotation (CVAT)
        |
Augmentation (Albumentations)
        |
YOLOv8s transfer learning (PyTorch)
        |
ONNX export (ARM64-optimised)
        |
Edge deployment (Raspberry Pi 5)
```

Dataset: 1,710 images collected on-site at a commercial greenhouse in Kuala Lumpur, Malaysia. Not
included in this repository.

---

## Hardware

| Component | Specification |
| --- | --- |
| Edge device | Raspberry Pi 5 (8 GB RAM) |
| Camera | Raspberry Pi Camera Module 3 (field deployment) or USB / V4L2 camera (bench profiling) |
| OS | Raspberry Pi OS 64-bit (Debian 13 for the profiling runs) |
| Storage | 32 GB+ microSD |
| Power | Official 27 W USB-C PSU, or Waveshare UPS HAT (E) |

## Roadmap

- [ ] Active learning pipeline (buffer low-confidence and OOD-flagged frames for expert relabelling)
- [ ] OTA model update via the existing Docker image distribution path
- [ ] LoRa-based offline alerting for connectivity-limited sites
- [ ] Multi-node deployment with MQTT in place of per-node HTTPS webhooks
- [ ] Multi-crop support

## Author

**Lin Ding Shan**
Faculty of Computer Science (Data Science), UCSI University, Kuala Lumpur, Malaysia
GitHub: [xiaolin200206](https://github.com/xiaolin200206)

## License

See `LICENSE`.

## Reproducing the paper

| Manuscript item | Script | Data |
|---|---|---|
| Tables 3-4 (profiling) | `scripts/analyse_profiling.py` | `data/profiling_runs/` |
| Table 2 (threshold x window replay) | `basil_experiments/03_analysis/threshold_sensitivity_analysis.py` | `data/field_log/` |
| Figs 3, 4, 5, 8 | `scripts/figures/fig3.py`, `scripts/figures/fig4_5_8.py` | values embedded |
| Fig 6 (thermal/CPU traces) | `scripts/figures/fig6.py` (run from repo root) | `data/profiling_runs/` |
| Section 4.5 supplementary session | summary statistics in manuscript | `data/supplementary_session/` |
| Table 1 (11-architecture comparison) | `basil_experiments/02_baseline_comparison/` | `basil_experiments/02_baseline_comparison/results/` (metrics CSVs + MobileNetV2 checkpoints) |

Note: the supplementary session ran under a 180 s/45 s duty cycle at a frame-level threshold
below the deployed tau = 0.70 (Disease confirmations occur at confidences down to 0.57,
consistent with the deployment script's 0.50 default; see data/supplementary_session/RUN_INFO.txt),
and is not comparable with the 60 s/15 s profiling runs (Tables 3-4) or the deployed alerting
configuration (tau = 0.70, 3-of-5 window). Its execution mode (native vs. containerised) was
not recorded.

The trained model export `basil_mobilenet.onnx` (MobileNetV2, 3-class, opset 19, 8.9 MB)
is included at the repository root, so `docker build` works out of the box; it can be
regenerated from a training checkpoint with `basil_experiments/04_deployment/convert_to_onnx.py`.
