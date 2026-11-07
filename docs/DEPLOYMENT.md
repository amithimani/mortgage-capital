# Deployment — container image

The agent ships as one OCI image that runs unchanged on any container platform. Everything that
differs between environments comes from environment variables.

## What the image is

| Property | Value |
|---|---|
| Base | `python:3.12-slim` (Debian). Pin by digest for production: `--build-arg PYTHON_IMAGE=python:3.12-slim@sha256:…` |
| Build | Two stages. Every dependency is installed from hash-pinned lock files (`requirements/*.lock.txt`, `pip --require-hashes`). The app is built offline with the locked build backend, then `pip check` runs |
| Size | ~300 MB |
| User | Non-root, uid/gid `10001` |
| Filesystem | Works with a read-only root. The only writable paths are `/app/var` (state) and `/tmp` |
| Entrypoint | `mc-recon serve`: one uvicorn worker, graceful SIGTERM shutdown (≤ 20 s) |
| Port | `$PORT`, default `8000`, bound to `0.0.0.0` |
| Probes | `GET /healthz` (liveness: the process is up), `GET /readyz` (readiness: the store answers; returns 503 otherwise) |
| Docker `HEALTHCHECK` | Built in (used by Docker, Compose and ECS; Kubernetes uses its own probes) |
| Bundled data | `data/synthetic` (demo datasets + oracle recordings for the fake LLM), read-only |

Verified locally on the built image: the smoke test passes with `--read-only --cap-drop ALL
--security-opt no-new-privileges`. A run started in one container finishes after
`docker stop`, and a fresh container on the same volume resumes it (with a different `PORT`).
Shutdown exits with code 0.

## Configuration

| Variable | Default (in image) | Purpose |
|---|---|---|
| `PORT` | `8000` | Listen port. Cloud Run, App Service and others inject it |
| `HOST` | `0.0.0.0` | Bind address |
| `MC_DB_PATH` | `/app/var/mc_recon.sqlite` | Run store + LangGraph checkpoints. **Put it on a persistent volume** |
| `MC_DATA_ROOT` | `/app/data/synthetic` | Directory of datasets a run can name. Mount real data here |
| `MC_LLM_PROVIDER` | `fake` | Only `fake` today; Claude/Gemini plug in later |
| `MC_POLICY_GATE` | `fake` | Only `fake` today; `disseqt` once credentials exist |
| `MC_LOG_LEVEL` | `info` | uvicorn log level; logs go to stdout/stderr for the platform to collect |

Configuration is validated at startup. Any invalid value stops the process with exit code 2 and
a message listing every problem, so it never starts half-configured.

**Secrets** (LLM and Disseqt keys, when they arrive) are injected as environment variables from
the platform's secret manager: AWS Secrets Manager, Azure Key Vault, GCP Secret Manager, or a
Kubernetes Secret. They are never put in the image or the repo.

## Build, run, test

```bash
make image                                   # docker build -t mc-recon:<git-sha> -t mc-recon:latest .
make run                                     # hardened local run on a named volume
make smoke                                   # scripts/smoke_test.py against localhost
docker compose up --build                    # same, via Compose

# multi-arch (Graviton / Ampere / Apple silicon + x86)
docker buildx build --platform linux/amd64,linux/arm64 -t <registry>/mc-recon:<tag> --push .
```

`scripts/smoke_test.py <url>` has no dependencies beyond Python. Run it after any deploy, against
any URL. CI (`deploy/ci/github-actions-ci.yml`) runs the tests, builds the image, starts it hardened, and
smoke-tests it on every push. To activate it, move the file to `.github/workflows/ci.yml`.

If you build behind a TLS-intercepting proxy, pass its CA as a BuildKit secret. It is used
only during `pip install` and never stored in the image:
`docker build --secret id=pip_ca,src=/path/ca.pem .`

### Pushing to a registry

```bash
# AWS ECR
aws ecr get-login-password | docker login --username AWS --password-stdin <acct>.dkr.ecr.<region>.amazonaws.com
docker tag mc-recon:latest <acct>.dkr.ecr.<region>.amazonaws.com/mc-recon:<tag>
# Azure ACR
az acr login -n <registry> && docker tag mc-recon:latest <registry>.azurecr.io/mc-recon:<tag>
# Google Artifact Registry
gcloud auth configure-docker <region>-docker.pkg.dev
docker tag mc-recon:latest <region>-docker.pkg.dev/<project>/<repo>/mc-recon:<tag>
```

## Where it runs today, and what limits it

The image runs everywhere. **Keeping state** is what varies, because the PoC stores runs,
checkpoints and the append-only audit log in SQLite at `MC_DB_PATH`. SQLite needs a local block
volume and a single writer.

| Platform | Run the image | State in the PoC (SQLite) |
|---|---|---|
| **Kubernetes**: EKS, AKS, GKE, OpenShift, on-prem | `deploy/kubernetes/mc-recon.yaml` (PVC, probes, hardened security context) | ✅ Durable. The PVC is a block volume (EBS / Azure Disk / PD). 1 replica, `Recreate` |
| **Docker / Compose on a VM**: EC2, Azure VM, GCE | `docker compose up -d` | ✅ Durable on the named volume |
| **AWS ECS on EC2** | Task definition, port 8000, EBS-backed volume at `/app/var` | ✅ Durable (single task) |
| **AWS ECS Fargate**, **Azure Container Apps** | Works as-is | ⚠️ Their shared volumes are network file systems (EFS / Azure Files). SQLite locking on NFS/SMB is unreliable, so treat these as demo-only until Postgres |
| **Google Cloud Run**, **AWS App Runner** | Works as-is (`PORT` is honoured) | ⚠️ The filesystem is ephemeral, so state is lost when an instance restarts. Demo only until Postgres |

### Scaling and state: the next step

To run more than one replica, or to get durable state on serverless platforms (Cloud Run,
Fargate, Container Apps, App Runner), the state has to move to managed Postgres (RDS / Aurora,
Azure Database for PostgreSQL, Cloud SQL):

1. Swap the checkpointer for `langgraph-checkpoint-postgres`. It's the same LangGraph interface
   and a one-line change in `service.py`.
2. Port `RunStore` to Postgres. Keep the append-only triggers, using
   `CREATE RULE` or `BEFORE UPDATE/DELETE` triggers.
3. Replace the in-process per-run locks with `SELECT … FOR UPDATE` or advisory locks.
4. Select the backend with `MC_DB_URL`, and keep SQLite as the default for local runs and tests.

Until then the deployment is deliberately **single replica**: the Kubernetes manifest uses
`replicas: 1` with `strategy: Recreate`, so two pods never write the same SQLite file.

Graph calls currently run inside the HTTP request. That's fine with the fake LLM. A real model
will make `POST /runs` and mapping confirmation take seconds to minutes; at that point they move
to a background worker and return `202`, and platform request timeouts (Cloud Run 60 min by
default, ALB 60 s) need checking.
