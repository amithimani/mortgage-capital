"""Runtime configuration from environment variables (12-factor; identical on every cloud).

Validated once at startup: a bad or unsupported value stops the process with a clear message,
instead of surfacing as a 500 on the first request.

| Variable                 | Default                 | Notes                                   |
|--------------------------|-------------------------|-----------------------------------------|
| ``PORT`` / ``HOST``      | ``8000`` / ``0.0.0.0``  |                                         |
| ``MC_DATA_ROOT``         | ``data/synthetic``      | Datasets a run may reference by name    |
| ``MC_DB_PATH``           | ``var/mc_recon.sqlite`` | Run store + checkpoints; use a volume   |
| ``MC_LOG_LEVEL``         | ``info``                | uvicorn log level                       |
| ``MC_LLM_PROVIDER``      | ``fake``                | ``fake`` or ``http`` (OpenAI-compatible)|
| ``MC_LLM_BASE_URL``      |                         | Required for ``http``                   |
| ``MC_LLM_API_KEY``       |                         |                                         |
| ``MC_LLM_MODEL``         | ``mock-llm``            |                                         |
| ``MC_POLICY_GATE``       | ``fake``                | ``fake`` or ``http`` (Disseqt surface)  |
| ``MC_DISSEQT_URL``       |                         | Required for ``http``; also ships traces|
| ``MC_DISSEQT_API_KEY``   |                         |                                         |
| ``MC_DISSEQT_APPLICATION_ID`` |                    | Registered automatically if empty       |
| ``MC_DISSEQT_POLICY_ID`` | ``exception-policy``    |                                         |
| ``MC_HTTP_TIMEOUT``      | ``30``                  | Seconds, per external HTTP call         |
| ``MC_ANALYST_TOKENS``    |                         | ``token:analyst,...`` turns on auth     |
| ``MC_MAPPING_MEMORY_DAYS`` | ``90``                | ``0`` disables long-term memory         |
| ``MC_GOVERNANCE``        | ``on``                  | Use Case 2 controls: ``on`` or ``off``  |
| ``MC_LIMIT_<NAME>``      | see ``limits.py``       | e.g. ``MC_LIMIT_MAX_REVIEW_ROUNDS=2``   |
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from .limits import AgentLimits

LLM_PROVIDERS = ("fake", "http")
POLICY_GATES = ("fake", "http")
LOG_LEVELS = ("critical", "error", "warning", "info", "debug", "trace")


class ConfigError(ValueError):
    pass


def _parse_tokens(raw: str, problems: list[str]) -> dict[str, str]:
    tokens: dict[str, str] = {}
    for pair in filter(None, (p.strip() for p in raw.split(","))):
        token, sep, analyst = pair.partition(":")
        if not (sep and token and analyst):
            problems.append("MC_ANALYST_TOKENS entries must look like token:analyst")
            continue
        tokens[token] = analyst
    return tokens


def _parse_limits(env: Mapping[str, str], problems: list[str]) -> AgentLimits:
    defaults = AgentLimits()
    values = {}
    for f in dataclasses.fields(AgentLimits):
        name = f"MC_LIMIT_{f.name.upper()}"
        default = getattr(defaults, f.name)
        if name not in env:
            values[f.name] = default
            continue
        try:
            value = type(default)(env[name])
            if value < 0:
                raise ValueError
            values[f.name] = value
        except ValueError:
            problems.append(f"{name} must be a non-negative {type(default).__name__}")
            values[f.name] = default
    try:
        return AgentLimits(**values)
    except ValueError as exc:
        problems.append(str(exc))
        return defaults


@dataclass(frozen=True)
class Settings:
    host: str = "0.0.0.0"
    port: int = 8000
    data_root: Path = Path("data/synthetic")
    db_path: str = "var/mc_recon.sqlite"
    llm_provider: str = "fake"
    policy_gate: str = "fake"
    log_level: str = "info"
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = "mock-llm"
    disseqt_url: str = ""
    disseqt_api_key: str = ""
    disseqt_application_id: str = ""
    disseqt_policy_id: str = "exception-policy"
    http_timeout: float = 30.0
    analyst_tokens: dict[str, str] = field(default_factory=dict)
    mapping_memory_days: int = 90
    governance: bool = True
    limits: AgentLimits = field(default_factory=AgentLimits)

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

        def number(name: str, default: float, kind: type) -> float:
            try:
                return kind(env.get(name, default))
            except ValueError:
                problems.append(f"{name} must be a number")
                return default

        settings = cls(
            host=env.get("HOST", "0.0.0.0"),
            port=port,
            data_root=Path(env.get("MC_DATA_ROOT", "data/synthetic")),
            db_path=env.get("MC_DB_PATH", "var/mc_recon.sqlite"),
            llm_provider=choice("MC_LLM_PROVIDER", "fake", LLM_PROVIDERS),
            policy_gate=choice("MC_POLICY_GATE", "fake", POLICY_GATES),
            log_level=choice("MC_LOG_LEVEL", "info", LOG_LEVELS),
            llm_base_url=env.get("MC_LLM_BASE_URL", ""),
            llm_api_key=env.get("MC_LLM_API_KEY", ""),
            llm_model=env.get("MC_LLM_MODEL", "mock-llm"),
            disseqt_url=env.get("MC_DISSEQT_URL", ""),
            disseqt_api_key=env.get("MC_DISSEQT_API_KEY", ""),
            disseqt_application_id=env.get("MC_DISSEQT_APPLICATION_ID", ""),
            disseqt_policy_id=env.get("MC_DISSEQT_POLICY_ID", "exception-policy"),
            http_timeout=number("MC_HTTP_TIMEOUT", 30.0, float),
            analyst_tokens=_parse_tokens(env.get("MC_ANALYST_TOKENS", ""), problems),
            mapping_memory_days=int(number("MC_MAPPING_MEMORY_DAYS", 90, int)),
            governance=choice("MC_GOVERNANCE", "on", ("on", "off")) == "on",
            limits=_parse_limits(env, problems),
        )
        if not settings.data_root.is_dir():
            problems.append(f"MC_DATA_ROOT={str(settings.data_root)!r} is not a directory")
        if settings.llm_provider == "http" and not settings.llm_base_url:
            problems.append("MC_LLM_PROVIDER=http needs MC_LLM_BASE_URL")
        if settings.policy_gate == "http" and not settings.disseqt_url:
            problems.append("MC_POLICY_GATE=http needs MC_DISSEQT_URL")
        if problems:
            raise ConfigError("Invalid configuration:\n  - " + "\n  - ".join(problems))
        return settings
