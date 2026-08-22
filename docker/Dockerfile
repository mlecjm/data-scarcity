# syntax=docker/dockerfile:1.6
# -----------------------------------------------------------------------------
# Reproducibility image for the AI2M4RI revised manuscript.
#   docker build -t data-scarcity:latest -f docker/Dockerfile .
#   docker run --rm -v "$PWD/data:/workspace/data" -v "$PWD/reports:/workspace/reports" \
#       data-scarcity:latest make reproduce
# -----------------------------------------------------------------------------
ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/workspace \
    DS_SEED=42

WORKDIR /workspace

# OS-level deps for matplotlib + locale + git for source-of-truth metadata
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential gcc git locales ca-certificates curl \
 && sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen \
 && locale-gen \
 && rm -rf /var/lib/apt/lists/*

ENV LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8

# Install Python deps first (better layer caching)
COPY requirements.txt /workspace/requirements.txt
RUN pip install --upgrade pip \
 && pip install -r /workspace/requirements.txt

# Copy source last
COPY . /workspace

# Smoke-test the import at build time so broken images fail fast
RUN python -c "from data_scarcity.modeling.compare_models import main; print('OK')"

# Default command: print the help
CMD ["make", "help"]
