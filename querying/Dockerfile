# ----------------------
# Stage 1: Builder
# ----------------------
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies if needed (e.g., gcc for certain libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
# Install requirements + production server inside the venv
RUN pip install --no-cache-dir -r requirements.txt gunicorn uvicorn

# ----------------------
# Stage 2: Runtime
# ----------------------
FROM python:3.11-slim

RUN useradd --create-home appuser
WORKDIR /app

COPY --from=builder /opt/venv /opt/venv

# 1. Add /app/app to PYTHONPATH so "import telemetry" works
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH="/app/app:${PYTHONPATH}"
ENV PYTHONUNBUFFERED=1

COPY querying/app/ ./app/

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# 2. Run gunicorn from the /app directory
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "app.query_api:app", "--bind", "0.0.0.0:8080"]