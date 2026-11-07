#!/usr/bin/env python3
"""Black-box smoke test against a running deployment (container, cluster, or cloud URL).

    python scripts/smoke_test.py http://localhost:8000 [--wait 60]

Standard library only, so it runs anywhere (CI, a jump box, a deploy pipeline step).
Exits non-zero on the first failed check.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
import uuid


def call(base: str, method: str, path: str, body: dict | None = None, headers=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base.rstrip("/") + path, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, json.loads(resp.read() or b"null")
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"null")


def check(cond: bool, msg: str) -> None:
    print(("PASS " if cond else "FAIL ") + msg, flush=True)
    if not cond:
        sys.exit(1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("base_url")
    ap.add_argument("--wait", type=float, default=60, help="seconds to wait for /readyz")
    ap.add_argument("--dataset", default="pair_01")
    args = ap.parse_args()
    base = args.base_url

    deadline = time.monotonic() + args.wait
    while True:
        try:
            status, body = call(base, "GET", "/readyz")
            if status == 200:
                break
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        if time.monotonic() > deadline:
            check(False, f"service ready within {args.wait}s")
        time.sleep(1)
    check(True, f"/readyz {body}")

    status, body = call(base, "GET", "/healthz")
    check(status == 200 and body["status"] == "ok", f"/healthz {body}")

    analyst = {"X-Analyst-Id": "smoke-test"}
    status, _ = call(base, "POST", "/runs", {"dataset": args.dataset})
    check(status == 401, "mutations require X-Analyst-Id")

    key = {"Idempotency-Key": f"smoke-{uuid.uuid4()}", **analyst}
    status, run = call(base, "POST", "/runs", {"dataset": args.dataset}, key)
    check(status == 201 and run["status"] == "awaiting_mapping_confirmation", "run started")
    rid = run["run_id"]
    status, again = call(base, "POST", "/runs", {"dataset": args.dataset}, key)
    check(status == 201 and again["run_id"] == rid, "idempotent retry returns the same run")

    status, mapping = call(base, "GET", f"/runs/{rid}/mapping")
    check(status == 200 and mapping["problems"] == [], "mapping proposed without problems")

    status, run = call(base, "POST", f"/runs/{rid}/mapping/confirm", {}, analyst)
    check(status == 200, f"mapping confirmed -> {run.get('status')}")

    status, log = call(base, "GET", f"/runs/{rid}/remediation-log")
    check(status == 200 and len(log) > 0, f"remediation log has {len(log)} entries")

    status, events = call(base, "GET", f"/runs/{rid}/events")
    check(
        status == 200
        and events[0]["event_type"] == "mapping_confirmed"
        and events[0]["actor"] == "smoke-test",
        "oversight event recorded with analyst identity",
    )
    print(f"OK run {rid}: status={run['status']} counts={run['exception_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
