# Use the official NVIDIA CUDA runtime as the base image
# This provides the necessary CUDA libraries for GPU support and works for CPU too.
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Define a build-time argument to switch between CPU and GPU installation
ARG RUNTIME=nvidia

# Set environment variables for Python and Hugging Face
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
# Set the Hugging Face home directory to a path inside the container for better caching
ENV HF_HOME=/app/model_cache

# Install system dependencies required for the application
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    ffmpeg \
    espeak-ng \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


# Set the working directory inside the container
WORKDIR /app

# Set uv environment variables for dependency installation
ENV UV_COMPILE_BYTECODE=1
ENV UV_SKIP_WHEEL_FILENAME_CHECK=1

# Install uv from pre-built image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Use uv to fetch Python 3.13
RUN uv python install 3.13

# Copy dependency files first to leverage Docker's layer caching
COPY pyproject.toml uv.lock ./

# Sync the dependencies in the virtual environment
RUN uv sync --python 3.13 --frozen --no-install-project --no-dev

# --- Conditionally Install GPU Dependencies ---
# If the RUNTIME argument is 'nvidia', install onnxruntime-gpu.
# We also explicitly install CUDA versions of torch to override the CPU default.
RUN if [ "$RUNTIME" = "nvidia" ]; then \
    echo "RUNTIME=nvidia, installing GPU dependencies..."; \
    uv pip install onnxruntime-gpu; \
    uv pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu121; \
    else \
    echo "RUNTIME=cpu, skipping GPU dependencies."; \
    fi

# Copy the rest of the application code into the container
COPY . .

RUN mkdir -p model_cache

# Expose the port the application will run on (aligned with docker-compose.yml)
EXPOSE 8005

# Place the virtual environment in the PATH to use it for execution
ENV PATH="/app/.venv/bin:$PATH"

# The command to run when the container starts
CMD ["python", "src/server.py"]
