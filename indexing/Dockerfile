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

# Create non-root user first
RUN useradd --create-home appuser
WORKDIR /app

# Copy the virtual environment from the builder
COPY --from=builder /opt/venv /opt/venv

# Copy application code
# Note: Ensure the path matches your CMD (app/index_api.py)
COPY indexing/app/ ./app/

# Set permissions
RUN chown -R appuser:appuser /app /opt/venv

# Set environment variables
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

USER appuser

EXPOSE 8000

# Use the absolute path to gunicorn inside the venv to be safe
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "app.index_api:app", "--bind", "0.0.0.0:8000"]