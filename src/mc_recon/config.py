"""Runtime configuration from environment variables (12-factor; identical on every cloud).

Validated once at startup: a bad or unsupported value stops the process with a clear message,
instead of surfacing as a 500 on the first request.

| Variable          | Default                  | Notes                                             |
|-------------------|--------------------------|---------------------------------------------------|
| ``PORT``          | ``8000``                 | Most platforms (Cloud Run, App Service, …) set it |
| ``HOST``          | ``0.0.0.0``              |                                                   |
| ``MC_DATA_ROOT``  | ``data/synthetic``       | Datasets a run may reference by name              |
| ``MC_DB_PATH``    | ``var/mc_recon.sqlite``  | Run store + checkpoints; put it on a volume       |
| ``MC_LLM_PROVIDER`` | ``fake``               | ``fake`` (recorded); real providers pending       |
| ``MC_POLICY_GATE``  | ``fake``               | ``fake``; ``disseqt`` pending credentials         |
| ``MC_LOG_LEVEL``  | ``info``                 | uvicorn log level                                 |
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

LLM_PROVIDERS = ("fake",)
POLICY_GATES = ("fake",)
LOG_LEVELS = ("critical", "error", "warning", "info", "debug", "trace")


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Settings:
    host: str = "0.0.0.0"
    port: int = 8000
    data_root: Path = Path("data/synthetic")
    db_path: str = "var/mc_recon.sqlite"
    llm_provider: str = "fake"
    policy_gate: str = "fake"
    log_level: str = "info"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        env = os.environ if env is None else env
        problems: list[str] = []

        raw_port = env.get("PORT", "8000")
        try:
            port = int(raw_port)
            if not 1 <= port <= 65535:
                raise ValueError
        except ValueError:
            problems.append(f"PORT must be an integer 1-65535, got {raw_port!r}")
            port = 8000

        def choice(name: str, default: str, allowed: tuple[str, ...]) -> str:
            value = env.get(name, default).strip().lower()
            if value not in allowed:
                problems.append(
                    f"{name}={value!r} is not supported (allowed: {', '.join(allowed)})"
                )
            return value

        settings = cls(
            host=env.get("HOST", "0.0.0.0"),
            port=port,
            data_root=Path(env.get("MC_DATA_ROOT", "data/synthetic")),
            db_path=env.get("MC_DB_PATH", "var/mc_recon.sqlite"),
            llm_provider=choice("MC_LLM_PROVIDER", "fake", LLM_PROVIDERS),
            policy_gate=choice("MC_POLICY_GATE", "fake", POLICY_GATES),
            log_level=choice("MC_LOG_LEVEL", "info", LOG_LEVELS),
        )
        if not settings.data_root.is_dir():
            problems.append(f"MC_DATA_ROOT={str(settings.data_root)!r} is not a directory")
        if problems:
            raise ConfigError("Invalid configuration:\n  - " + "\n  - ".join(problems))
        return settings
