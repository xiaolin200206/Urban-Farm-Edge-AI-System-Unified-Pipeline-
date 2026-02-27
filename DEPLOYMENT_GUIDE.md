Urban Farm Edge AI - Deployment & Operations Playbook
The definitive engineering reference for deploying, scaling, and troubleshooting the Urban Farm Edge AI system. Built for resource-constrained Raspberry Pi 5 hardware operating in hostile, high-thermal agricultural environments.
Table of Contents

1.Edge Architecture & Inference Pipeline

2.Bare-Metal Deployment (Dev Mode)

3.Containerized Deployment (Production SaaS)

4.Telemetry & Alerting Engine

5.Multi-Tenant Service Management

6.Model Over-The-Air (OTA) Management

7/Audit Trails & Log Analytics

8.System Configuration Registry

9.Edge Troubleshooting Playbook

10.Engineering Benchmarks: Native vs. Docker

1.Edge Architecture & Inference Pipeline

The system is designed to bridge the gap between heavy cloud AI and lightweight edge execution, ensuring zero-latency autonomy even in network-dead zones.

🧠 The Micro-Pest Optimized Pipeline (Edge Architecture)

[ Physical Environment ]  
📷 1080p Camera Module (Captures RGB888 high-res frames)  
│  
▼  

[ Core AI Engine (Raspberry Pi 5) ]  
├── ✂️ Grid Patch Cropper: Slices 1080p frames into 400x400 patches (Crucial for Micro-Pest preservation)  
├── 🧠 MobileNetV2 (ONNX): Executes ~50 ms sub-inference on local edge CPU  
└── ⏱️ Logic Controller:  
    ├── Anti-Jitter: Requires 3/5 frames consensus  
    └── Thresholds: Filters results (e.g., Confidence ≥ 0.70)  
│  
▼  

[ Telemetry & Execution ]  
├── 🖥️ OpenCV Display (For local debugging)  
├── 📊 CSV Telemetry (Logs CPU temp, latency, FPS)  
├── 🖼️ Captured Anomalies (Saves high-res patches to local disk)  
└── 🚨 Telegram Webhook (Pushes alerts to Cloud / End User)  
    
                                     


🧠 The Micro-Pest Optimized Pipeline

Unlike standard vision pipelines that squash full frames (destroying micro-features), this pipeline utilizes Grid Patch Cropping to retain 100% fidelity of extremely small objects (e.g., Mites) before feeding them into the ONNX Runtime.

1. 1080p Frame Capture (Camera)
   │
   ▼
2. Grid Patch Slicing (e.g., slice into 400x400 patches)  <-- Critical for Micro-Pests
   │
   ▼
3. Local Resize (224x224) & ImageNet Normalize
   │
   ▼
4. ONNX Runtime Inference (~50ms per patch on Pi5 CPU)
   │
   ▼
5. Softmax Confidence Evaluation
   │
   ▼
6. Temporal Anti-Jitter Consensus (Requires 3/5 frames to trigger)
   │
   ▼
7. Threshold Filter (e.g., ≥ 0.70 Confidence)
   │
   ▼
8. Execution: Alert Dispatch / Logging / Visual Feedback

2. Bare-Metal Deployment

Recommended for local development, rapid prototyping, and dataset collection.

# 1. Clone the production branch
cd /home/raspberry
git clone https://github.com/YOUR_USERNAME/edge-ai-core.git farm_edge
cd farm_edge

# 2. Isolate environment
python3 -m venv env_edge
source env_edge/bin/activate

# 3. Install edge dependencies
pip install -r requirements.txt

# 4. Initiate inference engine
python3 core_inference.py

3. Containerized Deployment (Docker)

The mandated deployment method for B2B SaaS rollout, ensuring environment parity and isolated I/O operations.

3.1 Bootstrap Docker

curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker raspberry
sudo reboot

3.2 Build the Edge Container

cd /home/raspberry/farm_edge/docker_env
docker build -t urbanfarm-edge-ai .

3.3 Spin Up (Headless Production Mode)

docker run -d --rm \
    --device /dev/video0 \
    -v /home/raspberry/farm_edge/farm_logs:/app/logs \
    --name farm-container \
    urbanfarm-edge-ai

4. Telemetry & Alerting Engine

To prevent "Alert Fatigue" for farm operators, the system employs a rigorous multi-stage verification filter before dispatching payloads via Telegram webhooks.
Alert Logic Gates:

    1.Feature Lock: Confidence score must exceed CLASSIFICATION_THRESHOLD (Default: 0.70).

    2.Temporal Lock: Must survive the Anti-Jitter buffer (3 out of 5 sequential frames positive).

    3.Cooldown Lock: Must respect the TELEGRAM_COOLDOWN_SEC (Default: 60s) to prevent spam during continuous exposure.  

  Configuration (core_inference.py)

TELEGRAM_BOT_TOKEN = "YOUR_SECURE_TOKEN"
TELEGRAM_CHAT_ID = "YOUR_TARGET_CHAT_ID"
TELEGRAM_COOLDOWN_SEC = 60.0    # Throttling limit

5. Multi-Tenant Service Management

How to seamlessly context-switch the Edge Node between different agricultural tasks using systemd.

# Switch deployment context to Basil Classification
sudo systemctl stop farm_durian_det.service
sudo systemctl disable farm_durian_det.service
sudo systemctl enable farm_basil_class.service
sudo systemctl start farm_basil_class.service

# Check health telemetry
systemctl status farm_basil_class.service
journalctl -u farm_basil_class.service -f

6. Model Over-The-Air (OTA) Management
⚠️ Critical PyTorch Export Bug (Pi 5 Architecture)

When exporting fine-tuned models from the Cloud/Colab to the Edge, you must force the legacy ONNX exporter. New PyTorch distributions split weights into .onnx.data fragments, which triggers fatal I/O overhead on the Pi 5's ARM architecture.

Correct Export Protocol:

import torch

# Force single-file monolith export for Pi 5 compatibility
torch.onnx.export(
    model, dummy_input, "mobilenet_v2_optimized.onnx",
    opset_version=11,
    export_params=True,
    dynamo=False,  # <--- CRITICAL: Disables weight fracturing
)

7. Audit Trails & Log Analytics

Every execution session generates an immutable audit trail for offline backend analysis and data-flywheel retraining.

farm_logs/session_20260226_150000/
├── telemetry.csv            # Microsecond-level performance metrics
├── lifecycle_events.csv     # Boot/Sleep/Thermal throttle events
└── captured_anomalies/      # High-res patches saved for Active Learning

Quick Edge Analytics (Bash):

# Check thermal limits reached during session
awk -F',' 'NR>1 {if($6>max) max=$6} END {print "Peak Core Temp: " max "°C"}' telemetry.csv

# Calculate true FPS throughput
awk -F',' 'NR>1 {sum+=$3; n++} END {print "Avg Throughput: " sum/n " FPS"}' telemetry.csv

8. System Configuration Registry

(Adjust via config.yaml or script headers)

Parameter, Default, Engineering Rationale
CONFIDENCE_THRESH, 0.70, Balance between recall and false-positive suppression.
ENABLE_PATCH_CROP, True, Crucial for Micro-Pest detection. Bypasses standard resize loss.
CYCLE_ACTIVE_SEC, 60, Sustained inference window.
CYCLE_SLEEP_SEC, 15, Thermal dissipation window to prevent hardware throttling (Limit: 82°C).

9. Edge Troubleshooting Playbook

Issue: RuntimeError: Failed to acquire camera: Device or resource busy

    1.Root Cause: A zombie systemd background process is holding the /dev/video0 hardware lock.

    2.Resolution:
    
    sudo systemctl stop farm_basil_class.service
    pkill -f core_inference.py

Issue: False Positives on Extreme Micro-Pests

    1.Root Cause: The system fell back to standard Resize(224) instead of Patch Cropping.

    2.Resolution: Verify ENABLE_PATCH_CROP = True is set. Standard resize evaporates 10-pixel bugs into noise artifacts.

10. Engineering Benchmarks: Native vs. Docker

Empirical validation for Academic & Enterprise stakeholders regarding containerization overhead on ARM Cortex-A76.

Methodology: A continuous 3-hour stress test operating under tropical ambient conditions, comparing Bare-Metal execution vs. Docker Containerization.

Metric,Bare-Metal,Docker SaaS,Delta (Overhead)
Avg Latency,48.2 ms,50.1 ms,+3.9%
Throughput,19.8 FPS,19.1 FPS,-3.5%
CPU Load,45%,47%,+2.0%
RAM Footprint,32%,35%,+3.0%
Thermal Peak,52.1°C,53.4°C,+1.3°C
