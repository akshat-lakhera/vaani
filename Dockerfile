FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/data/models \
    TRANSFORMERS_CACHE=/app/data/models \
    HF_DATASETS_CACHE=/app/data/hf_cache

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
COPY web ./web
COPY scripts ./scripts
RUN pip install --no-cache-dir -e .

EXPOSE 8080
CMD ["python", "-m", "vaani.api"]
