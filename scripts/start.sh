#!/bin/sh
set -eu
mkdir -p /app/data/indexes/shipped /app/data/raw /app/data/reports /app/data/hf_cache /app/data/models
META=/app/data/indexes/shipped/meta.json
if [ ! -s "$META" ] && [ -f /opt/vaani-meta.json ]; then
  cp /opt/vaani-meta.json "$META"
fi
exec python -m vaani.api
