# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Prevent writing bytecode and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src:/app

# Install curl for HEALTHCHECK
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code, datasets, and build scripts
COPY src/ ./src/
COPY data/ ./data/
COPY scripts/ ./scripts/
# Validated responses used to pre-warm the semantic cache at startup
COPY results.jsonl ./results.jsonl
# Extra cached answers collected by scripts/warm_cache.py (the vectorizer .pkl is
# excluded via .dockerignore and rebuilt below)
COPY artifacts/ ./artifacts/

# Build TF-IDF vectorizer at image build time
RUN python scripts/build_corpus_vectorizer.py

# Expose default HTTP port
EXPOSE 8000

# Hosts like Render inject $PORT; default to 8000 for local `docker run`.
ENV PORT=8000

# Health check against /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f "http://localhost:${PORT}/health" || exit 1

# Start uvicorn with api:app imported from src/ (shell form so $PORT expands)
CMD uvicorn api:app --app-dir src --host 0.0.0.0 --port "${PORT}"
