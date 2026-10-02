# Redraw API for a Hugging Face Docker Space (free CPU basic: 2 vCPU, 16 GB RAM).
# Built from the folder that deploy/hf-space/build.py assembles; see deploy/hf-space/DEPLOY.md.
FROM python:3.11-slim

# Spaces run the container as uid 1000
RUN useradd -m -u 1000 user
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code runs from /app (pipeline.config resolves data/config relative to the repo root)
COPY --chown=user . .

USER user
ENV PYTHONUNBUFFERED=1 \
    REDRAW_DATA_DIR=/app/data/processed \
    SIM_WORKERS=2 \
    COOKIE_SECURE=true

EXPOSE 7860
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860", "--proxy-headers", "--forwarded-allow-ips", "*"]
