# KittenTTS-FastAPI Helm Chart

A Helm chart for KittenTTS-FastAPI based on the `bjw-s` [`app-template`](https://github.com/bjw-s/helm-charts/tree/main/charts/other/app-template).

## Prerequisites

- [Helm](https://helm.sh) >= 3.0
- Kubernetes cluster
- Optionally: NVIDIA GPU operator installed in the cluster (if using GPU tag)

## Setup

First add the `bjw-s` repository:

```bash
helm repo add bjw-s https://bjw-s-labs.github.io/helm-charts
helm repo update
```

Update dependencies for this chart:

```bash
cd charts/kittentts-fastapi
helm dependency update
```

## Installation

To install or upgrade the chart with the release name `kittentts`:

```bash
helm upgrade --install kittentts charts/kittentts-fastapi -n kittentts --create-namespace
```

## Configuration

This chart is built using the [`app-template`](https://github.com/bjw-s/helm-charts/tree/main/charts/other/app-template) from bjw-s. 
All standard options from `app-template` are available.

Important options specifically adjusted in `values.yaml` for KittenTTS-FastAPI:
- **`controllers.main.containers.main.image.tag`**: Update the image tag (e.g., `cpu` or `nvidia` depending on your setup).
- **`persistence.model-cache`**: An `emptyDir` mount configured at `/app/model_cache` for storing HuggingFace models. You can change this to `persistentVolumeClaim` to persist downloaded models across pod restarts.
- **GPU Passthrough**: You can uncomment the `limits` -> `nvidia.com/gpu: 1` section in `values.yaml` to assign a GPU to the container (requires `nvidia` tag).

### App Environment Variables

Set app-specific runtime options under:
`app-template.controllers.main.containers.main.env`

Commonly used keys:
- `KITTEN_MODEL_REPO_ID`
- `KITTEN_TTS_DEVICE`
- `KITTEN_MODEL_CACHE`
- `KITTEN_AUDIO_FORMAT`
- `KITTEN_TEXT_PROFILE` (`balanced`, `narration`, `dialogue`)
- `KITTEN_UI_TITLE`
- `KITTEN_UI_SHOW_LANGUAGE_SELECT`

Text preprocessing behavior is profile-driven. Select the active profile with
`KITTEN_TEXT_PROFILE`.
