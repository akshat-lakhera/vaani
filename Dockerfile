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
COPY scripts/start.sh /app/start.sh
# CPU torch only — the default Linux wheel pulls a multi-GB CUDA toolkit.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install --no-cache-dir -e . \
    && useradd --create-home --uid 10001 vaani \
    && mkdir -p /app/data /opt/e5-small \
    && python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-small').save('/opt/e5-small')" \
    && chown -R vaani:vaani /app /opt/e5-small

# Run as root so a Railway/Fly volume mounted at /app/data is writable
# (bind mounts are root-owned; uid 10001 cannot mkdir raw/reports).
# The e5 snapshot lives in the image; the 57k index lives on the volume
# because Trial volumes are capped at 0.5GB.
ENV VAANI_LOCAL_MODEL=/opt/e5-small
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8080/api/health || exit 1
ENTRYPOINT ["/bin/sh", "/app/start.sh"]
