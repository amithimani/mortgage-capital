"""Production entrypoint: ``mc-recon serve`` (the container's CMD).

One process, one worker. The run store and checkpointer are SQLite, and per-run locks live in
memory, so horizontal scaling needs the Postgres backends first (see docs/DEPLOYMENT.md).
uvicorn handles SIGTERM with a bounded graceful shutdown, which is what every container
platform sends before killing the task.
"""

from __future__ import annotations

import sys

from .config import ConfigError, Settings


def serve(settings: Settings | None = None) -> int:
    try:
        settings = settings or Settings.from_env()
        from .api import build_service, create_app

        app = create_app(build_service(settings), settings.analyst_tokens)
    except ConfigError as exc:
        print(f"mc-recon: {exc}", file=sys.stderr)
        return 2

    if not settings.analyst_tokens:
        print(
            "mc-recon: WARNING no MC_ANALYST_TOKENS set; identity is an unverified header "
            "and reads are unauthenticated. Local PoC use only.",
            file=sys.stderr,
            flush=True,
        )

    import uvicorn

    print(
        f"mc-recon: serving on {settings.host}:{settings.port} "
        f"(llm={settings.llm_provider}, policy_gate={settings.policy_gate}, "
        f"db={settings.db_path}, data_root={settings.data_root})",
        file=sys.stderr,
        flush=True,
    )
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
        workers=1,
        timeout_graceful_shutdown=20,
        server_header=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(serve())
