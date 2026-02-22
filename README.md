# KittenTTS FastAPI

**A high-performance, lightweight Text-to-Speech (TTS) server built with FastAPI, wrapping the onnx [KittenTTS v0.8 model](https://github.com/KittenML/KittenTTS), which can run very efficiently on CPU with optional GPU acceleration.**

This project provides a robust, production-ready interface for the ultra-lightweight KittenTTS model and engine. It features a modern Web UI, true GPU acceleration via ONNX Runtime, and full OpenAI API compatibility for easy integration into existing workflows.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=for-the-badge)](LICENSE)
[![Python Version](https://img.shields.io/badge/Python-3.13+-blue.svg?style=for-the-badge)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/Framework-FastAPI-green.svg?style=for-the-badge)](https://fastapi.tiangolo.com/)
[![Model Source](https://img.shields.io/badge/Model-KittenML/KittenTTS-orange.svg?style=for-the-badge)](https://github.com/KittenML/KittenTTS)
[![Docker](https://img.shields.io/badge/Docker-Supported-blue.svg?style=for-the-badge)](https://www.docker.com/)
[![Web UI](https://img.shields.io/badge/Web_UI-Included-4285F4?style=for-the-badge&logo=googlechrome&logoColor=white)](#)
[![CUDA Compatible](https://img.shields.io/badge/NVIDIA_CUDA-Compatible-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-zone)
[![API](https://img.shields.io/badge/OpenAI_Compatible_API-Ready-000000?style=for-the-badge&logo=openai&logoColor=white)](https://platform.openai.com/docs/api-reference)

## ✨ Features

Production-ready [FastAPI](https://fastapi.tiangolo.com/) wrapper around [KittenTTS](https://github.com/KittenML/KittenTTS), focused on fast local/self-hosted deployment.

*   **Modern Web UI**: Text input, voice controls, playback, and download.
*   **OpenAI-Compatible API**: Includes `/v1/models`, `/v1/audio/speech`, and `/v1/audio/voices`.
*   **GPU Acceleration**: Uses ONNX Runtime GPU providers when available.
*   **CPU Friendly**: Lightweight model (~15M params, under 25MB).
*   **Long-Text Support**: Optional chunking and merged output.
*   **Robust Text Preprocessing**: Cleans noisy artifacts and normalizes tricky text forms for more stable synthesis.
*   **Env-Based Config**: Configure runtime with `.env` and `KITTEN_*` vars.
*   **Browser UI State**: UI preferences are stored in local browser storage.
*   **Built-in Voices**: 8 voices included (Bella, Jasper, Luna, Bruno, Rosie, Hugo, Kiki, Leo).
*   **Piper Alternative**: Compact self-hosted TTS with low overhead.

---

## 🐳 Docker Quickstart

Fastest way to run on CPU with the published image:

```bash
docker run -it -d \
  --name kittentts-fastapi \
  --restart unless-stopped \
  -e KITTEN_MODEL_REPO_ID="KittenML/kitten-tts-nano-0.8-fp32" \
  -p 8005:8005 \
  ghcr.io/richardr1126/kittentts-fastapi-cpu
```

> Works well on Raspberry Pi (64-bit OS) as well.

## Environment Variables (Optional)

Supported environment variables:

*   `KITTEN_SERVER_HOST` (default: `0.0.0.0`)
*   `KITTEN_SERVER_PORT` (default: `8005`)
*   `KITTEN_SERVER_ENABLE_PERFORMANCE_MONITOR` (default: `false`)
*   `KITTEN_MODEL_REPO_ID` (default: `KittenML/kitten-tts-nano-0.8-fp32`)
*   `KITTEN_TTS_DEVICE` (default: `auto`, options: `auto`, `cpu`, `cuda`)
*   `KITTEN_MODEL_CACHE` (default: `model_cache`)
*   `KITTEN_GEN_DEFAULT_SPEED` (default: `1.1`)
*   `KITTEN_GEN_DEFAULT_LANGUAGE` (default: `en`)
*   `KITTEN_AUDIO_FORMAT` (default: `wav`, options: `wav`, `mp3`, `opus`, `aac`)
*   `KITTEN_AUDIO_SAMPLE_RATE` (default: `24000`)
*   `KITTEN_FILTER_TABLE_ARTIFACTS` (default: `true`)
*   `KITTEN_FILTER_REFERENCE_ARTIFACTS` (default: `true`)
*   `KITTEN_FILTER_SYMBOL_NOISE` (default: `true`)
*   `KITTEN_UI_TITLE` (default: `Kitten TTS Server`)
*   `KITTEN_UI_SHOW_LANGUAGE_SELECT` (default: `true`)

## 🛠️ Local Installation

### 1. Prerequisites

*   **Python:** 3.13+
*   **uv:** [Install uv](https://docs.astral.sh/uv/getting-started/installation/)
*   **Audio runtime libs:** `libsndfile` and `ffmpeg` available on system path.

### 2. Local Setup

```bash
# Clone the repository
git clone https://github.com/richardr1126/KittenTTS-FastAPI.git
cd KittenTTS-FastAPI

# Create local environment config
cp .env.example .env

# Sync dependencies and create virtual environment (CPU/default)
uv sync

# Run the server
uv run src/server.py
```

For NVIDIA GPU local installs, sync the dedicated dependency group:

```bash
uv sync --group nvidia
```

Then set `KITTEN_TTS_DEVICE=cuda` in `.env` (or export it in your shell) before starting the server.

After startup, the server logs the exact UI URL to visit (typically `http://localhost:8005/`).

## 🐳 Docker Compose Setup

The fastest way to deploy is using Docker Compose.
Create `.env` first (`cp .env.example .env`), then run:

### CPU (Default)

```bash
docker compose up -d --build
```

### NVIDIA GPU
Make sure you have the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed.

```bash
docker compose -f docker-compose-gpu.yml up -d --build
```

---

## 📖 API Usage

### OpenAI Compatible Endpoint (`/v1/audio/speech`)
```bash
curl http://localhost:8005/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "Hello from the Kitten TTS FastAPI server!",
    "voice": "Jasper",
    "speed": 1.1,
    "response_format": "mp3"
  }' \
  --output speech.mp3
```
`model` accepts canonical `tts-1` and also supports aliases `KittenTTS` and `kitten-tts`.

### Model List Endpoint (`/v1/models`)
```bash
curl http://localhost:8005/v1/models
```

### Voice List Endpoint (`/v1/audio/voices`)
```bash
curl http://localhost:8005/v1/audio/voices
```

### Interactive Docs
Visit `http://localhost:8005/docs` for the full Swagger UI.

---

## ⚙️ Configuration

Server settings are loaded from environment variables (`.env` for local/dev).
Copy `.env.example` to `.env` and edit values as needed, then restart the server.

*   **`KITTEN_TTS_DEVICE`**: `auto`, `cuda`, or `cpu`.
*   **`KITTEN_AUDIO_FORMAT`**: `wav`, `mp3`, `opus`, or `aac`.
*   **`KITTEN_MODEL_REPO_ID`**: Hugging Face model repo.
*   **`KITTEN_MODEL_CACHE`**: Model cache directory path.
*   **Text preprocessing** is enabled by default and includes cleanup/normalization steps (for example, mixed alphanumeric tokens like `gpt4` are normalized to improve phonemization stability).
*   **UI state** (last text, voice, theme) is stored in browser `localStorage`, not on the API server.

---

## 🛠️ Troubleshooting

*   **Phonemizer Errors**: The app uses bundled `espeakng_loader`; if your platform blocks dynamic libraries, install system `espeak-ng`.
*   **GPU Not Used**: Ensure `torch.cuda.is_available()` is `True`. The container/host must have NVIDIA drivers and the Container Toolkit.
*   **Audio Errors on Linux**: Ensure `libsndfile1` is installed (`sudo apt install libsndfile1`).

---

## 🙏 Acknowledgements

*   **[KittenTTS](https://github.com/KittenML/KittenTTS)**: The core model by [KittenML](https://github.com/KittenML).
*   **Original Server**: Based on the work by [devnen](https://github.com/devnen/Kitten-TTS-Server).

---

## 📄 License

This project is licensed under the **Apache License 2.0**. See the [LICENSE](LICENSE) file for details. This license choice aligns with the upstream KittenTTS project/model licensing used by this repository.
