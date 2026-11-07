# syntax=docker/dockerfile:1.7
#
# Mortgage Capital reconciliation agent — OCI image, cloud-agnostic.
#
#   docker build -t mc-recon .
#   docker run -p 8000:8000 -v mc-recon-data:/app/var mc-recon
#
# Multi-arch:   docker buildx build --platform linux/amd64,linux/arm64 -t <registry>/mc-recon .
# Behind a TLS-intercepting proxy, pass its CA as a build secret (never baked into the image):
#   docker build --secret id=pip_ca,src=/path/to/ca.pem ...
#
# Pin the base by digest for reproducible production builds, e.g.
#   --build-arg PYTHON_IMAGE=python:3.12-slim@sha256:<digest>

ARG PYTHON_IMAGE=python:3.12-slim

# ---------------------------------------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS build

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
WORKDIR /build

# Dependencies first (cached unless a lock file changes). Every package is hash-verified.
COPY requirements/runtime.lock.txt requirements/build.lock.txt requirements/
RUN --mount=type=secret,id=pip_ca,required=false \
    if [ -s /run/secrets/pip_ca ]; then export PIP_CERT=/run/secrets/pip_ca; fi; \
    pip install --require-hashes --no-deps -r requirements/runtime.lock.txt \
 && pip install --require-hashes --no-deps --target /tmp/buildtools -r requirements/build.lock.txt

# The application itself: built with the locked hatchling, no network, no extra deps.
COPY pyproject.toml README.md ./
COPY src ./src
RUN PYTHONPATH=/tmp/buildtools pip install --no-deps --no-build-isolation --no-index . \
 && python -m compileall -q /opt/venv \
 && pip check

# ---------------------------------------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS runtime

ARG VERSION=0.1.0
ARG REVISION=unknown
LABEL org.opencontainers.image.title="mc-recon" \
      org.opencontainers.image.description="Mortgage Capital reconciliation & anomaly detection agent" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${REVISION}"

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8000 \
    MC_DATA_ROOT=/app/data/synthetic \
    MC_DB_PATH=/app/var/mc_recon.sqlite \
    MC_LLM_PROVIDER=fake \
    MC_POLICY_GATE=fake

RUN groupadd --system --gid 10001 app \
 && useradd --system --uid 10001 --gid app --home-dir /app --shell /usr/sbin/nologin app

COPY --from=build /opt/venv /opt/venv
WORKDIR /app
# Demo datasets + oracle recordings (read-only). Mount real data over MC_DATA_ROOT instead.
COPY data/synthetic /app/data/synthetic
# The only writable path: run store + checkpoints. Mount a persistent volume here.
RUN mkdir -p /app/var && chown app:app /app/var

USER 10001:10001
EXPOSE 8000

# Docker / ECS / Compose use this; Kubernetes uses the probes in deploy/kubernetes.
HEALTHCHECK --interval=30s --timeout=3s --start-period=15s --retries=3 \
  CMD ["python", "-c", "import os,urllib.request; urllib.request.urlopen(f\"http://127.0.0.1:{os.environ.get('PORT','8000')}/healthz\", timeout=2)"]

STOPSIGNAL SIGTERM
CMD ["mc-recon", "serve"]
