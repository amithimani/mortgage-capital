"""Deployment contract: env configuration, probes, lock files and container files stay in sync."""

from __future__ import annotations

import re
import tomllib

import pytest
from agent_helpers import make_service
from conftest import REPO_ROOT, SYNTHETIC_DIR
from fastapi.testclient import TestClient

from mc_recon.api import build_service, create_app
from mc_recon.config import ConfigError, Settings
from mc_recon.server import serve


def test_settings_defaults_and_overrides():
    s = Settings.from_env(
        {"PORT": "9000", "MC_DATA_ROOT": str(SYNTHETIC_DIR), "MC_DB_PATH": "/x/db.sqlite"}
    )
    assert (s.port, s.host, s.db_path, s.llm_provider, s.policy_gate) == (
        9000,
        "0.0.0.0",
        "/x/db.sqlite",
        "fake",
        "fake",
    )


def test_settings_report_every_problem():
    with pytest.raises(ConfigError) as exc:
        Settings.from_env(
            {"PORT": "70000", "MC_LLM_PROVIDER": "gpt", "MC_DATA_ROOT": "/definitely/not/here"}
        )
    msg = str(exc.value)
    assert "PORT" in msg and "MC_LLM_PROVIDER" in msg and "MC_DATA_ROOT" in msg


def test_serve_exits_2_on_bad_config(monkeypatch, capsys):
    monkeypatch.setenv("MC_POLICY_GATE", "nope")
    monkeypatch.setenv("MC_DATA_ROOT", str(SYNTHETIC_DIR))
    assert serve() == 2
    assert "MC_POLICY_GATE" in capsys.readouterr().err


def test_fake_provider_needs_recordings(tmp_path):
    with pytest.raises(ConfigError, match="oracle_recordings"):
        build_service(Settings(data_root=tmp_path, db_path=str(tmp_path / "db.sqlite")))


def test_build_service_from_settings_runs_end_to_end(tmp_path):
    svc = build_service(Settings(data_root=SYNTHETIC_DIR, db_path=str(tmp_path / "db.sqlite")))
    client = TestClient(create_app(svc))
    assert client.get("/readyz").json() == {"ready": True, "store": True, "llm": True}
    rid = client.post("/runs", json={"dataset": "pair_01"}, headers={"X-Analyst-Id": "a"}).json()[
        "run_id"
    ]
    r = client.post(f"/runs/{rid}/mapping/confirm", json={}, headers={"X-Analyst-Id": "a"})
    assert r.json()["status"] == "completed"


def test_readyz_503_when_store_down():
    svc, _ = make_service()
    client = TestClient(create_app(svc))
    svc.store.close()
    r = client.get("/readyz")
    assert r.status_code == 503 and r.json()["store"] is False
    assert client.get("/healthz").status_code == 200  # liveness stays up: don't restart-loop


def _lock_names(path) -> dict[str, str]:
    pins = re.findall(r"^([A-Za-z0-9_.\-]+)==([^\s;\\]+)", path.read_text(), re.MULTILINE)
    return {name.lower().replace("_", "-"): version for name, version in pins}


def test_runtime_lock_covers_declared_dependencies():
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["project"]
    declared = project["dependencies"] + project["optional-dependencies"]["agent"]
    locked = _lock_names(REPO_ROOT / "requirements" / "runtime.lock.txt")
    for req in declared:
        name = re.split(r"[<>=!~\[ ;]", req, maxsplit=1)[0].lower().replace("_", "-")
        assert name in locked, f"{name} missing from runtime.lock.txt — run `make lock`"
    assert "hatchling" in _lock_names(REPO_ROOT / "requirements" / "build.lock.txt")


def test_lock_files_are_fully_hashed():
    for name in ("runtime.lock.txt", "build.lock.txt"):
        text = (REPO_ROOT / "requirements" / name).read_text()
        pins = len(re.findall(r"^[A-Za-z0-9_.\-]+==", text, re.MULTILINE))
        assert pins and text.count("--hash=sha256:") >= pins


def test_container_files_reference_real_paths():
    dockerfile = (REPO_ROOT / "Dockerfile").read_text()
    for path in ("requirements/runtime.lock.txt", "requirements/build.lock.txt", "data/synthetic"):
        assert path in dockerfile and (REPO_ROOT / path).exists()
    assert "USER 10001" in dockerfile
    ignore = (REPO_ROOT / ".dockerignore").read_text()
    for needed in ("!src/", "!requirements/", "!data/synthetic/", "!pyproject.toml"):
        assert needed in ignore
