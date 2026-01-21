FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=80 \
    GUNICORN_WORKERS=4

RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

COPY index_service/app /app

RUN useradd --create-home --shell /bin/bash appuser

RUN mkdir -p /app/data/pdfs \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 80
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:80", "index_api:app"]
