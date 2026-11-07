"""Synthetic data generation and ground-truth fixtures."""

from .synthetic import (
    ERROR_TYPES,
    FILE_NAMES,
    NOISE_TYPES,
    PAIRS,
    PairSpec,
    build_pair,
    load_catalog,
    load_expected_mapping,
    pair_paths,
    write_pair,
)

__all__ = [
    "ERROR_TYPES",
    "FILE_NAMES",
    "NOISE_TYPES",
    "PAIRS",
    "PairSpec",
    "build_pair",
    "load_catalog",
    "load_expected_mapping",
    "pair_paths",
    "write_pair",
]
