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
RUN pip install --upgrade pip setuptools wheel \
    && pip wheel --no-deps --wheel-dir /wheels .

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
    MODEL_ID=unscored-baseline \
    MODEL_PATH=/models

WORKDIR /app

# Ensure /app is owned by the non-root user (WORKDIR creates as root)
RUN chown invoicex:invoicex /app

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

# Copy application code
COPY --chown=invoicex:invoicex pyproject.toml .
COPY --chown=invoicex:invoicex src/ src/
COPY --chown=invoicex:invoicex schema/ schema/
# Create data directories
RUN mkdir -p /data/ingest/raw /data/tokens /data/candidates \
    /data/predictions /data/review /data/labels /data/logs /models \
    && chown -R invoicex:invoicex /data /models

# Switch to non-root user
USER invoicex

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from invoices.health import liveness_probe; exit(0 if liveness_probe() else 1)"

# Default command: show help
ENTRYPOINT ["invoicex"]
CMD ["--help"]
