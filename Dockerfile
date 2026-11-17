# syntax=docker/dockerfile:1
#
# Self-contained image: the synthetic data is generated and the forecast model is trained at BUILD
# time, so the container needs no database, no model download and no internet except the LLM.
#
# Stages (only the last one is shipped):
#   deps    runtime-only virtualenv (no mlflow, mcp, pytest or ruff)
#   train   throwaway: generate_data.py -> data/sales.db, then src.forecast.train -> models/
#   assets  builds the policy vector index and caches the offline embedding model
#   final   slim, non-root, copies only what the API needs
#
# Build for Azure Container Apps (linux/amd64). On an Apple Silicon Mac add --platform linux/amd64
# if you want an image you can push; CI already builds amd64.

ARG PYTHON_IMAGE=python:3.11-slim-bookworm

FROM ${PYTHON_IMAGE} AS base
# libgomp1 = OpenMP, which LightGBM needs at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

FROM base AS uv
COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /usr/local/bin/uv
ENV UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_COMPILE_BYTECODE=1
WORKDIR /app
COPY pyproject.toml uv.lock ./

# ---- runtime dependencies only ---------------------------------------------------------------
FROM uv AS deps
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-default-groups --no-install-project
# Slimming (about 300 MB): packages ship their own test suites, and chromadb pulls in the kubernetes
# client for a distributed mode we do not use. Neither is imported by this app; the test suite and
# the CI smoke test would catch it if that ever changed.
RUN find /app/.venv -type d -name tests -prune -exec rm -rf {} + \
    && rm -rf /app/.venv/lib/python3.11/site-packages/kubernetes \
              /app/.venv/lib/python3.11/site-packages/kubernetes-*.dist-info

# ---- build-time training (mlflow lives here and never reaches the final image) ----------------
FROM uv AS train
ENV UV_PROJECT_ENVIRONMENT=/opt/train-venv
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-default-groups --group train --no-install-project
COPY src ./src
COPY scripts ./scripts
COPY docs ./docs
ENV PATH="/opt/train-venv/bin:$PATH"
# Seeded, so every build produces the same data and a comparable model.
RUN python scripts/generate_data.py \
    && python -m src.forecast.train

# ---- policy index + offline embedding model ----------------------------------------------------
FROM deps AS assets
ENV PATH="/app/.venv/bin:$PATH" \
    HOME=/home/app
COPY src ./src
COPY docs ./docs
# Embeds docs/*.md into Chroma. The first call also downloads the small ONNX embedding model into
# $HOME/.cache; baking both in means a cold start never downloads anything.
RUN python -c "from src.tools.search_policies import build_index; build_index()" \
    && rm -f /home/app/.cache/chroma/onnx_models/all-MiniLM-L6-v2/onnx.tar.gz
# (the downloaded archive is not needed once it has been extracted)

# ---- final image ------------------------------------------------------------------------------
FROM base AS final
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --home-dir /home/app --create-home app
WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH" \
    HOME=/home/app \
    API_HOST=0.0.0.0 \
    PORT=8000

COPY --from=deps   --chown=app:app /app/.venv ./.venv
COPY --from=assets --chown=app:app /app/chroma_db ./chroma_db
COPY --from=assets --chown=app:app /home/app/.cache /home/app/.cache
COPY --from=train  --chown=app:app /app/data/sales.db ./data/sales.db
COPY --from=train  --chown=app:app /app/models ./models
COPY --chown=app:app src ./src
COPY --chown=app:app docs ./docs

USER app
EXPOSE 8000
# 200 when ready; a 503 (data/model missing) raises and marks the container unhealthy.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/health', timeout=4)"
CMD ["python", "-m", "src.api"]
