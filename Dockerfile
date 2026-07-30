# Invoice Extraction Pipeline
# Multi-stage build for optimized production image

# =============================================================================
# Stage 1: Builder
# =============================================================================
FROM python:3.11-slim AS builder

# Prevent Python from writing bytecode and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first for layer caching
COPY pyproject.toml .
COPY src/ src/

# Create wheel for the package
RUN pip wheel --no-deps --wheel-dir /wheels .

# Install all runtime dependencies into wheels (from pyproject.toml + optional extras)
RUN pip wheel --wheel-dir /wheels \
    ".[azure,mlflow]"

# =============================================================================
# Stage 2: Production
# =============================================================================
FROM python:3.11-slim AS production

# Security: Run as non-root user
RUN groupadd --gid 1000 invoicex \
    && useradd --uid 1000 --gid 1000 --shell /bin/bash --create-home invoicex

# Prevent Python from writing bytecode and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Application configuration
    INVOICEX_LOG_LEVEL=INFO \
    INVOICEX_LOG_FORMAT=json \
    # Model configuration
    MODEL_ID=unscored-baseline

WORKDIR /app

# Ensure /app is owned by the non-root user (WORKDIR creates as root)
RUN chown invoicex:invoicex /app

# Pre-create the HuggingFace cache dir owned by the runtime user so
# transformers never falls back to /root/.cache (read-only for invoicex).
# Must match HF_HOME in docker-compose.yml worker environment.
RUN mkdir -p /app/.cache/huggingface && chown -R invoicex:invoicex /app/.cache/huggingface

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    # PDF processing dependencies
    libpoppler-cpp-dev \
    # Health check
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy wheels from builder
COPY --from=builder /wheels /wheels

# Install wheels
RUN pip install --no-cache-dir /wheels/*.whl \
    && rm -rf /wheels

# Download spaCy language model for candidate anchor detection (Wave G).
# en_core_web_sm is used by views._nlp_pipeline() at runtime.
RUN python -m spacy download en_core_web_sm

# LiLT is an opt-in build feature. WITH_LILT=1 installs torch and bakes the
# weights; WITH_LILT=0 skips both and the worker runs the native decode path
# (heuristic + ranker). model_loader returns None when torch is absent.
ARG WITH_LILT=0

# LiLT token-classifier serving deps — CPU-only torch (no CUDA libs).
# Versions match the Modal training image (torch 2.2.2 / transformers 4.51.3).
RUN if [ "$WITH_LILT" = "1" ]; then \
      pip install --no-cache-dir \
        torch==2.2.2 transformers==4.51.3 tokenizers \
        --extra-index-url https://download.pytorch.org/whl/cpu; \
    else \
      echo "[lilt] skipped torch install (WITH_LILT=$WITH_LILT)"; \
    fi

# Copy application code
COPY --chown=invoicex:invoicex pyproject.toml .
COPY --chown=invoicex:invoicex src/ src/

# LiLT config, tokenizer and calibration. The weights file is added below only when WITH_LILT=1.
COPY --chown=invoicex:invoicex models/ models/

# Only when WITH_LILT=1: pull the frozen LiLT weights (496MB
# model.safetensors) from a GitHub Release at build time — too large for a
# git blob. Point GH_REPO/
# LILT_RELEASE_TAG at your own release; needs a GITHUB_TOKEN with repo
# read access (passed as a BuildKit secret, see ci.yml). FATAL by design:
# the build fails loudly if the token, release, asset, or sha256/size
# verification fails — so a successful WITH_LILT=1 image provably bakes the
# exact trained LiLT weights (no silent heuristic fallback in that build).
ARG GH_REPO=stochastic-sisyphus/text-feature-span-extractor
ARG LILT_RELEASE_TAG=lilt-weights-v1
RUN --mount=type=secret,id=gh_token bash -c '\
  set -euo pipefail; \
  if [ "$WITH_LILT" != "1" ]; then echo "[lilt] skipped weights fetch (WITH_LILT=$WITH_LILT)"; exit 0; fi; \
  EXPECT_SHA="e1c6b9c97c4d930c499a81fbfd9ede4f8bf35440413c38f38d119bcecd6db9f0"; \
  EXPECT_BYTES="520782932"; \
  if [ ! -f /run/secrets/gh_token ]; then echo "[lilt] FATAL: gh_token secret not mounted"; exit 1; fi; \
  TOKEN="$(cat /run/secrets/gh_token)"; \
  API="https://api.github.com/repos/${GH_REPO}"; \
  AID="$(curl -fsSL -H "Authorization: Bearer $TOKEN" "$API/releases/tags/${LILT_RELEASE_TAG}" | python -c "import sys,json; d=json.load(sys.stdin); print(next((a[\"id\"] for a in d.get(\"assets\",[]) if a[\"name\"]==\"model.safetensors\"), \"\"))")"; \
  if [ -z "$AID" ]; then echo "[lilt] FATAL: release ${LILT_RELEASE_TAG}/model.safetensors not found on ${GH_REPO}"; exit 1; fi; \
  curl -fsSL -H "Authorization: Bearer $TOKEN" -H "Accept: application/octet-stream" "$API/releases/assets/$AID" -o models/lilt/model.safetensors; \
  GOT_BYTES="$(wc -c < models/lilt/model.safetensors)"; \
  if [ "$GOT_BYTES" != "$EXPECT_BYTES" ]; then echo "[lilt] FATAL: size mismatch got=$GOT_BYTES expect=$EXPECT_BYTES"; exit 1; fi; \
  GOT_SHA="$(sha256sum models/lilt/model.safetensors | cut -d" " -f1)"; \
  if [ "$GOT_SHA" != "$EXPECT_SHA" ]; then echo "[lilt] FATAL: sha256 mismatch got=$GOT_SHA expect=$EXPECT_SHA"; exit 1; fi; \
  chown invoicex:invoicex models/lilt/model.safetensors; \
  echo "[lilt] weights verified & baked: $GOT_BYTES bytes sha256=$GOT_SHA"'

# Switch to non-root user
USER invoicex

# No default entrypoint — compose command: overrides drive worker/queue startup.
# The api service is gone; the :8080 HEALTHCHECK it exposed is dead.
# Worker healthcheck is disabled at compose level (no HTTP port on the worker).
CMD ["python", "-m", "invoices.queue"]

