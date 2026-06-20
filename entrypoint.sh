#!/bin/sh
# Starts BOTH processes inside one container:
#   1. FastAPI (uvicorn) on the INTERNAL port 8000 — not exposed outside.
#   2. Streamlit UI on $PORT (default 8501) — this is what the user opens.
# ui.py talks to the API over http://localhost:8000 (same container = localhost).
set -e

# 1. API in the background.
uvicorn app.api:app --host 127.0.0.1 --port 8000 &

# 2. UI in the foreground (PID 1 signal handling via exec).
#    XSRF/CORS are disabled so the app works inside the Hugging Face iframe.
exec streamlit run app/ui.py \
  --server.port "${PORT:-8501}" \
  --server.address 0.0.0.0 \
  --server.headless true \
  --server.enableCORS false \
  --server.enableXsrfProtection false \
  --browser.gatherUsageStats false
