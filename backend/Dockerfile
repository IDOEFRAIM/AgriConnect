# ── AgriConnect Backend ──────────────────────────────────────
FROM python:3.13-slim AS base

LABEL maintainer="AgConnect Team"
LABEL description="Agriculture Multi-Agent System"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on

WORKDIR /app

# Dépendances système
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git gcc python3-dev libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"
RUN poetry config virtualenvs.create false

# Dépendances Python (cache Docker si pyproject.toml ne change pas)
COPY pyproject.toml poetry.lock /app/
RUN poetry install --no-interaction --no-ansi --no-root

# Code source - copy only source and packaging files to leverage Docker layer caching
# Copy source into /app/src so the package layout under `src/` is preserved
COPY src/ /app/src/
COPY pyproject.toml poetry.lock README.md /app/
ENV PYTHONPATH=/app/src
RUN poetry install --no-interaction --no-ansi

# Dossiers runtime
RUN mkdir -p audio_output logs data/corpus

# ── Non-root user for security ──────────────────────────────
RUN groupadd --gid 1001 appgroup && \
    useradd --uid 1001 --gid appgroup --shell /bin/bash --create-home appuser && \
    chown -R appuser:appgroup /app /home/appuser
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the FastAPI app via the installed package entrypoint (src/agriconnect)
CMD ["uvicorn", "agriconnect.main:app", "--host", "0.0.0.0", "--port", "8000"]
