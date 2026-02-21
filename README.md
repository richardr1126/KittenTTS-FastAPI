# Kitten TTS Server (FastAPI)

**A high-performance, lightweight Text-to-Speech (TTS) server built with FastAPI, wrapping the [KittenTTS v0.8 model](https://github.com/KittenML/KittenTTS).**

This project provides a robust, production-ready interface for the ultra-lightweight KittenTTS engine (15M parameters). It features a modern Web UI, true GPU acceleration via ONNX Runtime, and full OpenAI API compatibility for easy integration into existing workflows.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)
[![Python Version](https://img.shields.io/badge/Python-3.13+-blue.svg?style=for-the-badge)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/Framework-FastAPI-green.svg?style=for-the-badge)](https://fastapi.tiangolo.com/)
[![Model Source](https://img.shields.io/badge/Model-KittenML/KittenTTS-orange.svg?style=for-the-badge)](https://github.com/KittenML/KittenTTS)
[![Docker](https://img.shields.io/badge/Docker-Supported-blue.svg?style=for-the-badge)](https://www.docker.com/)
[![Web UI](https://img.shields.io/badge/Web_UI-Included-4285F4?style=for-the-badge&logo=googlechrome&logoColor=white)](#)
[![CUDA Compatible](https://img.shields.io/badge/NVIDIA_CUDA-Compatible-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-zone)
[![API](https://img.shields.io/badge/OpenAI_Compatible_API-Ready-000000?style=for-the-badge&logo=openai&logoColor=white)](https://platform.openai.com/docs/api-reference)

<div align="center">
  <img src="static/screenshot-d.png" alt="Kitten TTS Server Web UI - Dark Mode" width="33%" />
  <img src="static/screenshot-l.png" alt="Kitten TTS Server Web UI - Light Mode" width="33%" />
</div>

---

## �️ Overview: Enhanced KittenTTS Generation

The [KittenTTS model by KittenML](https://github.com/KittenML/KittenTTS) provides a foundation for generating high-quality speech from a model smaller than 25MB. This project elevates that foundation into a production-ready service by providing a robust [FastAPI](https://fastapi.tiangolo.com/) server that makes KittenTTS significantly easier to use, more powerful, and drastically faster.

We solve the complexity of setting up and running the model by offering:

*   **Modern Web UI**: Easy experimentation, preset loading, and speed adjustment.
*   **True GPU Acceleration**: High-performance inference for NVIDIA GPUs.
*   **Large Text Handling**: Intelligently splits long texts into manageable chunks for audiobooks.
*   **OpenAI Compatibility**: Seamlessly integrate with any app expecting OpenAI's TTS API.
*   **Built-in Voices**: A fixed list of 8 ready-to-use voices (Bella, Jasper, Luna, Bruno, Rosie, Hugo, Kiki, Leo).

## 🔥 High-Performance GPU Acceleration

A standout feature is the implementation of **high-performance GPU acceleration**, a capability not available in the original KittenTTS project. While the base model is CPU-optimized, this server unlocks the full potential of your hardware:

*   **Optimized ONNX Runtime Pipeline**: We leverage `onnxruntime-gpu` to move the entire inference process to your NVIDIA graphics card.
*   **Eliminated I/O Bottlenecks**: The server uses advanced **I/O Binding**. This technique pre-allocates memory directly on the GPU for both model inputs and outputs, drastically reducing the latency caused by copying data between system RAM and the GPU's VRAM.
*   **True Performance Gains**: This isn't just running the model on the GPU; it's an optimized pipeline designed to minimize latency and maximize throughput.

## 🔄 Alternative to Piper TTS

The [KittenTTS model](https://github.com/KittenML/KittenTTS) serves as an excellent alternative to [Piper TTS](https://github.com/rhasspy/piper) for fast generation on limited compute.

**KittenTTS Model Advantages:**
*   **Extreme Efficiency**: Just 15 million parameters and under 25MB.
*   **Universal Compatibility**: CPU-optimized to run anywhere.
*   **Real-time Performance**: Optimized for low-latency speech synthesis even on resource-constrained hardware.

---

## 🛠️ Installation

### 1. Prerequisites

*   **Python:** 3.13+
*   **uv:** [Install uv](https://docs.astral.sh/uv/getting-started/installation/)
*   **eSpeak NG:** Required for phonemization.
    *   **Linux:** `sudo apt install espeak-ng`
    *   **macOS/Windows:** Install via your package manager or official installers.

### 2. Local Setup

```bash
# Clone the repository
git clone https://github.com/richardr1126/KittenTTS-FastAPI.git
cd KittenTTS-FastAPI

# Workaround for uv lock file issue with kittentts incorrect wheel filename
export UV_SKIP_WHEEL_FILENAME_CHECK=1

# Sync dependencies and create virtual environment
uv sync

# Run the server
uv run src/server.py
```

After startup, the server logs the exact UI URL to visit (typically `http://localhost:8005`).

### 🍓 Raspberry Pi 5 Support

Raspberry Pi 5 works out-of-the-box with the standard Linux installation. 
**Installation Steps:**
```bash
export UV_SKIP_WHEEL_FILENAME_CHECK=1
sudo apt update && sudo apt install -y espeak-ng libsndfile1 ffmpeg python3-pip
uv sync
uv run src/server.py
```

---

## 🐳 Docker Deployment

The fastest way to deploy is using Docker Compose.

### NVIDIA GPU (Recommended)
Make sure you have the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed.

```bash
docker compose up -d --build
```

### CPU Only
```bash
docker compose -f docker-compose-cpu.yml up -d --build
```

---

## 📖 API Usage

### OpenAI Compatible Endpoint (`/v1/audio/speech`)
```bash
curl http://localhost:8005/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "kitten-tts",
    "input": "Hello from the Kitten TTS FastAPI server!",
    "voice": "Jasper",
    "speed": 1.1,
    "response_format": "mp3"
  }' \
  --output speech.mp3
```

### Interactive Docs
Visit `http://localhost:8005/docs` for the full Swagger UI.

---

## ⚙️ Configuration

Settings are stored in `config.yaml`. The server will generate a default one on first run.

*   **`tts_engine.device`**: Set to `auto`, `cuda`, or `cpu`.
*   **`audio_output.format`**: Default format (`wav`, `mp3`, `opus`).
*   **`ui_state`**: Automatically stores your last used text, voice, and theme.

---

## 🛠️ Troubleshooting

*   **Phonemizer / eSpeak Errors**: Ensure you have installed **eSpeak NG** and restarted your terminal.
*   **GPU Not Used**: Ensure `torch.cuda.is_available()` is `True`. The container/host must have NVIDIA drivers and the Container Toolkit.
*   **Audio Errors on Linux**: Ensure `libsndfile1` is installed (`sudo apt install libsndfile1`).

---

## 🙏 Acknowledgements

*   **[KittenTTS](https://github.com/KittenML/KittenTTS)**: The core model by [KittenML](https://github.com/KittenML).
*   **Original Server**: Based on the work by [devnen](https://github.com/devnen/Kitten-TTS-Server).

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.
