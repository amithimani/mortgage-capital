"""Loader: mapping validation and parsing must fail loudly, never guess."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from mc_recon.data.synthetic import load_expected_mapping, pair_paths
from mc_recon.engine import (
    ColumnMapping,
    EngineInputError,
    MappingError,
    ParseError,
    Source,
    SourceMapping,
    load_inputs,
    normalize_loan_id,
)
from mc_recon.engine.loader import read_csv


def _copy_pair(src: Path, dst: Path) -> dict[Source, Path]:
    out = {}
    for source, path in pair_paths(src).items():
        target = dst / path.name
        target.write_bytes(path.read_bytes())
        out[source] = target
    return out


def test_normalize_loan_id():
    assert normalize_loan_id(" mc-101 0001 ") == "MC-1010001"


def test_loads_pair_with_expected_mapping(pair_dir):
    loaded = load_inputs(pair_paths(pair_dir), load_expected_mapping(pair_dir))
    assert loaded.servicing and loaded.payments and loaded.investor
    assert all(isinstance(r.scheduled_amount, Decimal) for r in loaded.servicing)
    assert all(isinstance(p.value_date, date) for p in loaded.payments)
    assert all(len(h) == 64 for h in loaded.file_hashes.values())
    # loan_ref_formatting noise ("mc-... ") is normalised
    assert all(p.loan_id == p.loan_id.strip().upper() for p in loaded.payments)


def test_mapping_reports_every_problem_at_once(pair_dir):
    good = load_expected_mapping(pair_dir)
    cols = dict(good.servicing.columns)
    del cols["due_date"]
    cols["margin"] = "No Such Column"
    cols["bogus"] = cols["loan_id"]
    bad = good.model_copy(update={"servicing": SourceMapping(columns=cols)})
    with pytest.raises(MappingError) as exc:
        load_inputs(pair_paths(pair_dir), bad)
    text = "\n".join(exc.value.problems)
    assert "required field 'due_date' is not mapped" in text
    assert "'No Such Column', which is not a header" in text
    assert "unknown canonical field 'bogus'" in text


def test_column_mapped_twice_is_rejected(pair_dir):
    good = load_expected_mapping(pair_dir)
    cols = dict(good.investor.columns)
    cols["scheduled_amount"] = cols["closing_balance"]
    bad = good.model_copy(update={"investor": SourceMapping(columns=cols)})
    with pytest.raises(MappingError, match="mapped to both"):
        load_inputs(pair_paths(pair_dir), bad)


def test_wrong_date_format_fails_loudly(pair_dir):
    good = load_expected_mapping(pair_dir)
    bad = good.model_copy(
        update={"payments": good.payments.model_copy(update={"date_format": None})}
    )
    with pytest.raises(ParseError) as exc:
        load_inputs(pair_paths(pair_dir), bad)
    assert exc.value.source is Source.PAYMENTS


def test_unparseable_amount_fails_with_row_and_column(tmp_path):
    src = Path(__file__).resolve().parents[1] / "data" / "synthetic" / "pair_01"
    paths = _copy_pair(src, tmp_path)
    lines = paths[Source.PAYMENTS].read_text().splitlines()
    cells = lines[3].split(",")
    cells[3] = "12O.00"  # letter O, not zero
    lines[3] = ",".join(cells)
    paths[Source.PAYMENTS].write_text("\n".join(lines) + "\n")
    with pytest.raises(ParseError) as exc:
        load_inputs(paths, load_expected_mapping(src))
    assert (exc.value.row, exc.value.column) == (3, "AMOUNT_GBP")


def test_duplicate_servicing_loan_fails(tmp_path):
    src = Path(__file__).resolve().parents[1] / "data" / "synthetic" / "pair_01"
    paths = _copy_pair(src, tmp_path)
    text = paths[Source.SERVICING].read_text()
    paths[Source.SERVICING].write_text(text + text.splitlines()[1] + "\n")
    with pytest.raises(EngineInputError, match="appears on rows"):
        load_inputs(paths, load_expected_mapping(src))


def test_ragged_row_fails(tmp_path):
    f = tmp_path / "x.csv"
    f.write_text("a,b\n1,2\n3\n")
    with pytest.raises(EngineInputError, match="has 1 cells"):
        read_csv(f)


def test_missing_input_file_fails():
    mapping = ColumnMapping(
        servicing=SourceMapping(columns={}),
        payments=SourceMapping(columns={}),
        investor=SourceMapping(columns={}),
    )
    with pytest.raises(EngineInputError, match="missing input files"):
        load_inputs({}, mapping)
