A robust, commercial-grade Edge AI platform designed for real-time agricultural disease and pest monitoring. Supports seamless switching between Image Classification (e.g., MobileNet for micro-pests) and Object Detection (e.g., YOLOv8 for fungal spots) on edge devices (Raspberry Pi 5 / NVIDIA Jetson), with built-in Enterprise SaaS capabilities.
🔄 The Complete AI Lifecycle

To bridge the domain gap between controlled greenhouse environments and dynamic edge deployments, this system is governed by two distinct pipelines:

1. MLOps Training Pipeline (Offline / Cloud)
How we build the model before edge deployment:
Data Collection (1080p Greenhouse Videos) → Frame Extraction & Grid Cropping → Human-in-the-Loop Labeling → Albumentations (Augmentation) → Transfer Learning (MobileNetV2/YOLO) → ONNX Export

2. Edge Inference Pipeline (Real-time on Raspberry Pi)
How the system executes in the physical world (Detailed below):
Camera Capture (1080p) → Grid Slicing (Memory Efficient) → ONNX Inference → Confidence Thresholding → Telegram Alert
🏗️ Core Architecture (Current)
1. Unified Inference Pipeline

Designed to be model-agnostic. The system auto-routes the inference logic based on the loaded ONNX model topology:

Classification Mode (MobileNet/ResNet): * Pipeline: 1080p Extraction → Grid Patch Cropping → Resize(224) → Normalize → Softmax

Use Case: Micro-pests (Mites), global leaf health assessment.

Architectural Note: Introduced Grid-based Patch Cropping to completely eliminate the "pixel evaporation" effect when squashing full HD frames. This preserves high-fidelity features for extremely small objects, drastically reducing false positives caused by background noise.

Detection Mode (YOLOv8/v11): * Pipeline: Resize(640, keep_ratio) → Padding → NMS (Non-Maximum Suppression) → Bounding Box Output

Use Case: Localized fungal spots, fruit disease counting (e.g., Durian/Tomato).

2. Edge Device Management (systemd)

Multi-tenant Service Switching: Seamlessly switch between different agricultural deployment configurations without rebooting:
    Bash

sudo systemctl start farm_basil_class.service  # Run Basil Classification
sudo systemctl start farm_durian_det.service   # Run Durian Detection

Thermal & Duty Cycle Control: Configurable ACTIVE_SEC and SLEEP_SEC to prevent Edge CPU thermal throttling (Limit: 82°C) in harsh greenhouse environments.

3. Containerized Deployment (Docker)

Unified Dockerfile supporting OpenCV (Camera/Video) and ONNX Runtime.

Headless execution with volume mapping for local telemetry logs (/app/logs).

🚀 Enterprise SaaS Roadmap (The Data Flywheel)

Note for Deployment: The following modules are designed for B2B commercialization and subscription-based deployment.

    Phase 1: Edge Security & DRM (Hardware ID Binding) Strictly binds ONNX model decryption and script execution to the specific Raspberry Pi MAC address + CPU Serial Number, preventing unauthorized SD card cloning.

    Phase 2: Cloud Telemetry (Real-time Sync) Edge devices push live metrics (CPU%, Temp, Uptime, Inference FPS) to a centralized Cloud Dashboard via MQTT/WebSockets.

    Phase 3: Active Learning & Smart Hard-Mining If model confidence is ambiguous (e.g., 0.40 - 0.70), the edge device auto-saves the high-res patch. These edge cases are pushed to AWS S3 during network idle time for Human-in-the-loop (HITL) re-labeling and CI/CD model OTA updates.

📂 Project Structure
Plaintext

edge_ai_core/
├── core_inference.py            # Unified inference engine
├── weights/                     # ONNX models directory
│   ├── mobilenet_v2_basil.onnx  
│   └── yolov8_durian.onnx       
├── configs/                     # YAML configuration files
├── requirements.txt             
├── farm_logs/                   # Auto-generated telemetry & alerts
├── docker_env/                  # Docker build directory
└── README.md

⚙️ Model Configuration (config.yaml)

System configurations are decoupled for easy SaaS deployment:
YAML

mode: "classification" # or "detection"
model_path: "weights/mobilenet_v2_basil.onnx"
input_size: [224, 224] # [640, 640] for YOLO
confidence_threshold: 0.70

# Patch Cropping (For micro-pests in classification mode)
enable_patch_crop: True
patch_size: [400, 400]

# Duty Cycle & Thermal Limits
cycle_active_sec: 60
cycle_sleep_sec: 15
max_temp_limit: 82.0

# DRM & Cloud (WIP)
hwid_check: False
cloud_sync_url: "wss://api.urbanfarm.tech/stream"

🛠️ Quick Start

1. Clone & Setup
Bash

git clone https://github.com/YOUR_USERNAME/edge-ai-core.git
cd edge-ai-core
pip install -r requirements.txt

2. Run via Docker (with display)
Bash

docker run -it --rm \
    --device /dev/video0 \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $(pwd)/farm_logs:/app/logs \
    --name farm-edge-container \
    farm-ai-core

⚠️ Known Limitations

    Small Object Feature Collapse vs. Compute Trade-off: Pure classification networks struggle with micro-pests when resizing full HD frames. This is heavily mitigated via our Grid Patch Cropping pipeline, which introduces a slight CPU overhead tradeoff (processing 4-6 patches per frame instead of 1).

    Out-of-Distribution (OOD) Rejection: The classification module has limited rejection mechanisms for OOD inputs compared to detection models. It is optimized for fixed greenhouse camera deployments with relatively stable backgrounds.
