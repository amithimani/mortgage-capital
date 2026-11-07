"""``mc-recon`` command line: generate fixtures, profile files, run the engine.

mc-recon generate [--out data/synthetic]
mc-recon profile  data/synthetic/pair_01
mc-recon reconcile data/synthetic/pair_01 [--mapping path.json] [--check]
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

    args = parser.parse_args(argv)
    if args.cmd == "generate":
        return synthetic.main(args.rest)
    if args.cmd == "profile":
        return _cmd_profile(args)
    return _cmd_reconcile(args)


if __name__ == "__main__":
    raise SystemExit(main())
