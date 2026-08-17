FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/app/data/models \
    TRANSFORMERS_CACHE=/app/data/models \
    HF_DATASETS_CACHE=/app/data/hf_cache \
    VAANI_DATA_DIR=/app/data \
    VAANI_HOST=0.0.0.0 \
    VAANI_PORT=8080

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ffmpeg \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
COPY web ./web
COPY scripts ./scripts
RUN pip install --no-cache-dir -e . \
    && useradd --create-home --uid 10001 vaani \
    && mkdir -p /app/data \
    && chown -R vaani:vaani /app

USER vaani
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8080/api/health || exit 1
CMD ["python", "-m", "vaani.api"]
