# Phase 1 CPU-only serving image.
# Intentionally excludes training/ (see .dockerignore) and heavy ML stacks.
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only serving / bench / ops-related application code
COPY app ./app
COPY bench ./bench

RUN mkdir -p /app/logs /app/bench/results

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
