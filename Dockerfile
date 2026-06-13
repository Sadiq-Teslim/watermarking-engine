# FPWM — FairPlay Watermark Service
# Pinned Python + system ffmpeg. This image is both the dev runtime and the prod artifact.
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# ffmpeg (with libvmaf where available) for media I/O + quality metrics; curl for healthcheck.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first for layer caching.
COPY requirements.txt requirements-dev.txt requirements-neural-image.txt requirements-neural.txt ./
ARG INSTALL_DEV=false
ARG INSTALL_NEURAL=false
ARG INSTALL_NEURAL_IMAGE=false
ARG INSTALL_NEURAL_FULL=false
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && if [ "$INSTALL_DEV" = "true" ]; then pip install -r requirements-dev.txt; fi \
    && if [ "$INSTALL_NEURAL" = "true" ] || [ "$INSTALL_NEURAL_IMAGE" = "true" ]; then pip install -r requirements-neural-image.txt; fi \
    && if [ "$INSTALL_NEURAL_FULL" = "true" ]; then pip install -r requirements-neural.txt; fi

COPY . .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT:-8000}/healthz" || exit 1

# Default: all-in-one (worker + web). Honors $PORT (Render sets it).
# docker-compose and the Render blueprint override this to split web/worker.
CMD ["sh", "scripts/start.sh"]
