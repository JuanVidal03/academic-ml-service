# syntax=docker/dockerfile:1

# ---------- Build stage ----------
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Install dependencies first so the layer caches independently of source changes.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

COPY . .

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# ---------- Runtime stage ----------
FROM python:3.12-slim-bookworm

WORKDIR /app

RUN useradd --create-home --shell /bin/bash appuser

# The virtual environment, the source, and the trained model. Training data is
# deliberately excluded: training is an offline step, so the image carries only the
# artefact it needs to serve, which also keeps it immutable and small.
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv
COPY --from=builder --chown=appuser:appuser /app/src /app/src
COPY --from=builder --chown=appuser:appuser /app/models /app/models
# Migrations, so a release step can run `alembic upgrade head`.
COPY --from=builder --chown=appuser:appuser /app/alembic /app/alembic
COPY --from=builder --chown=appuser:appuser /app/alembic.ini /app/alembic.ini

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

# No ENV defaults for the required settings: baking them in would just recreate the
# fallbacks the config deliberately removed. Supply them at run time, e.g.
#   docker run -p 8000:8000 --env-file .env dropout-service:1.0.0
# PORT overrides the default where a platform injects it; otherwise 8000.
CMD ["sh", "-c", "uvicorn dropout_service.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
