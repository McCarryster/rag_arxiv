# syntax=docker/dockerfile:1.4
########################################################
# Builder stage
########################################################
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /wheels

COPY requirements.txt /wheels/requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel \
    && pip wheel --no-deps --wheel-dir=/wheels -r /wheels/requirements.txt

########################################################
# Runtime stage
########################################################
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ca-certificates \
       curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

COPY --from=builder /wheels /wheels
COPY requirements.txt /app/requirements.txt

RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir --no-index --find-links=/wheels -r /app/requirements.txt

# Copy ONLY application code
COPY vector_db_service/app /app

RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 80

ENV PORT=80 \
    GUNICORN_WORKERS=4

CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:80", "vector_db_api:app", "--access-logfile", "-", "--log-level", "info"]