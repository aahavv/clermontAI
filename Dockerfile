# ClermontAI — single image running BOTH the FastAPI API and the Streamlit UI.
# Match the Python version of the env the artifacts were tested under (3.14).
FROM python:3.14-slim

# LightGBM needs the OpenMP runtime at import time; slim images lack it.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces run containers as a non-root user (uid 1000).
RUN useradd --create-home --uid 1000 appuser

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/app \
    PORT=8501

WORKDIR /app

# Install deps first (this layer is cached unless requirements change).
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# Copy ONLY the files the serving path actually needs (keeps the image lean).
#   app/      -> api.py + ui.py
#   notebooks -> score.py + features.py  (imported by api.py)
#   models    -> the 4 frozen artifacts
COPY app/api.py app/ui.py app/
COPY notebooks/score.py notebooks/features.py notebooks/
COPY models/lgbm_tuned.joblib models/calibrator.joblib \
     models/transformers.joblib models/model_contract.json models/
COPY entrypoint.sh .

RUN chmod +x entrypoint.sh && chown -R appuser:appuser /app
USER appuser

# The Streamlit UI. (The API on 8000 stays internal to the container.)
EXPOSE 8501

ENTRYPOINT ["./entrypoint.sh"]
