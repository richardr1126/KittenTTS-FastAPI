# Use a build-time argument for the base image
ARG BASE_IMAGE=python:3.13-slim
FROM ${BASE_IMAGE}

# Define a build-time argument to switch between CPU and GPU installation
ARG RUNTIME=cpu

# Set environment variables for Python and Hugging Face
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
# Set the Hugging Face home directory to a path inside the container for better caching
ENV HF_HOME=/app/model_cache

# Set uv environment variables for dependency installation
ENV UV_COMPILE_BYTECODE=1

# Install system dependencies required for the application
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    ffmpeg \
    curl \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


# Set the working directory inside the container
WORKDIR /app

# Install uv from pre-built image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Use uv to fetch Python 3.13
RUN uv python install 3.13

# Copy dependency files first to leverage Docker's layer caching
COPY pyproject.toml uv.lock ./

# Sync dependencies in the virtual environment.
# For NVIDIA builds, include the dedicated nvidia dependency group.
RUN if [ "$RUNTIME" = "nvidia" ]; then \
    echo "RUNTIME=nvidia, syncing dependencies with NVIDIA group..."; \
    uv sync --python 3.13 --frozen --no-install-project --no-dev --group nvidia; \
    else \
    echo "RUNTIME=cpu, syncing base dependencies."; \
    uv sync --python 3.13 --frozen --no-install-project --no-dev; \
    fi

# Copy the rest of the application code into the container
COPY . .

RUN mkdir -p model_cache

# Expose the port the application will run on
EXPOSE 8005

# Place the virtual environment in the PATH to use it for execution
ENV PATH="/app/.venv/bin:$PATH"

# The command to run when the container starts
CMD ["python", "src/server.py"]
