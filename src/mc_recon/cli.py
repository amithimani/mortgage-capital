"""``mc-recon`` command line: generate fixtures, profile files, run the engine.

mc-recon generate [--out data/synthetic]
mc-recon profile  data/synthetic/pair_01
mc-recon reconcile data/synthetic/pair_01 [--mapping path.json] [--check]
mc-recon baseline data/synthetic/pair_0*   # Week 1 baseline summary
mc-recon serve                             # HTTP API (PORT / MC_* env vars)
mc-recon eval [--json]                     # every harness scenario, as a scorecard
mc-recon mock [--port 8100]                # mock Disseqt + LLM endpoints
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from .data import synthetic
from .data.synthetic import load_catalog, load_expected_mapping, pair_paths
from .engine import ColumnMapping, EngineInputError, load_inputs, profile_csv, reconcile


def _cmd_profile(args: argparse.Namespace) -> int:
    profiles = [profile_csv(p, s).model_dump(mode="json") for s, p in pair_paths(args.pair).items()]
    print(json.dumps(profiles, indent=2))
    return 0


def _cmd_reconcile(args: argparse.Namespace) -> int:
    mapping = (
        ColumnMapping.model_validate_json(Path(args.mapping).read_text())
        if args.mapping
        else load_expected_mapping(args.pair)
    )
    try:
        result = reconcile(load_inputs(pair_paths(args.pair), mapping))
    except EngineInputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(result.model_dump_json(indent=2))
    if args.check:
        got = Counter((d.kind.value, d.loan_id) for d in result.discrepancies)
        want = Counter(
            (e["expected_kind"], e["loan_id"]) for e in load_catalog(args.pair)["errors"]
        )
        if got != want:
            print(
                f"MISMATCH vs seeded errors: extra={dict(got - want)} missing={dict(want - got)}",
                file=sys.stderr,
            )
            return 1
        print(f"OK: {sum(want.values())} seeded errors found, no false positives", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mc-recon")
    sub = parser.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="generate synthetic pairs + oracle recordings")
    g.add_argument("rest", nargs=argparse.REMAINDER)

    p = sub.add_parser("profile", help="profile the three files of a pair")
    p.add_argument("pair", type=Path)

    r = sub.add_parser("reconcile", help="run the deterministic engine on a pair")
    r.add_argument("pair", type=Path)
    r.add_argument("--mapping", type=Path, help="ColumnMapping JSON (default: expected_mapping)")
    r.add_argument("--check", action="store_true", help="compare against seeded_errors.json")

    b = sub.add_parser("baseline", help="Week 1 baseline: manual notes through the same checks")
    b.add_argument("pairs", type=Path, nargs="+")

    sub.add_parser("serve", help="run the agent HTTP API (configured by environment variables)")

    e = sub.add_parser("eval", help="run every harness scenario and print a scorecard")
    e.add_argument("--data-root", type=Path, default=Path("data/synthetic"))
    e.add_argument("--json", action="store_true")

    m = sub.add_parser("mock", help="serve the mock Disseqt and LLM endpoints")
    m.add_argument("--data-root", type=Path, default=Path("data/synthetic"))
    m.add_argument("--host", default="0.0.0.0")
    m.add_argument("--port", type=int, default=8100)
    m.add_argument("--api-key", default="mock-key")

    args = parser.parse_args(argv)
    if args.cmd == "eval":
        from .evaluation import main as evaluate

        return evaluate(args.data_root, args.json)
    if args.cmd == "mock":
        import uvicorn

        from .evaluation import Env
        from .mocks import create_mock_app

        app = create_mock_app(Env(args.data_root).recordings, api_key=args.api_key)
        uvicorn.run(app, host=args.host, port=args.port)
        return 0
    if args.cmd == "baseline":
        from .baseline import run_baseline
        from .policy import FakePolicyGate

        print(json.dumps(run_baseline(args.pairs, FakePolicyGate())["summary"], indent=2))
        return 0
    if args.cmd == "serve":
        from .server import serve

        return serve()
    if args.cmd == "generate":
        return synthetic.main(args.rest)
    if args.cmd == "profile":
        return _cmd_profile(args)
    return _cmd_reconcile(args)


if __name__ == "__main__":
    raise SystemExit(main())
