# =============================================================================
# GreenPack Industries — EPR Compliance System
# Dockerfile
# =============================================================================
# Multi-stage build:
#   Stage 1 (builder): Install Python dependencies in a venv
#   Stage 2 (runtime): Copy venv + app code into a slim image
#
# Architecture note: Ollama is NOT included in this container — it must run
# on the host or as a sidecar container. The OLLAMA_BASE_URL env var points
# the app to the Ollama service.
# =============================================================================

# --- Stage 1: Builder ---
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies (needed for faiss-cpu, numpy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --- Stage 2: Runtime ---
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy venv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY app/ ./app/
COPY documents/ ./documents/
COPY mock_data/ ./mock_data/
COPY run.py .
COPY .env .

# Create directories for runtime data
RUN mkdir -p faiss_index

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Entrypoint
CMD ["python", "run.py"]
