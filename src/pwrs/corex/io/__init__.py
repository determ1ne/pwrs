# Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause
"""Unified, lossless IO for MATPOWER-compatible case data."""

from __future__ import annotations

from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import Any

from ..mpc import MatpowerCase
from ._excel import read_excel, write_excel
from ._json import read_json, write_json
from ._mat import read_mat, write_mat
from ._npz import read_npz, write_npz
from ._pypower import from_pypower, to_pypower

_SUFFIX_FORMATS = {".json": "json", ".mat": "mat", ".npz": "npz", ".xlsx": "excel"}
_FORMAT_ALIASES = {
    "json": "json",
    "mat": "mat",
    "npz": "npz",
    "excel": "excel",
    "xlsx": "excel",
}


def _format(path: Path, requested: str | None) -> str:
    suffix_format = _SUFFIX_FORMATS.get(path.suffix.lower())
    if requested is None:
        if suffix_format is None:
            if path.suffix.lower() == ".xls":
                raise ValueError("legacy .xls workbooks are not supported; use .xlsx")
            raise ValueError(f"cannot infer case format from path: {path}")
        return suffix_format
    try:
        explicit = _FORMAT_ALIASES[requested.lower()]
    except (AttributeError, KeyError) as exc:
        raise ValueError(f"unsupported case format: {requested!r}") from exc
    if path.suffix and suffix_format != explicit:
        raise ValueError(f"case format {requested!r} conflicts with path suffix {path.suffix!r}")
    return explicit


def load(
    source: str | PathLike[str] | Mapping[str, Any],
    *,
    format: str | None = None,
    variable_name: str = "mpc",
) -> MatpowerCase:
    """Load a case by path suffix, explicit format, or PYPOWER mapping."""
    if isinstance(source, Mapping):
        if format not in (None, "pypower", "dict"):
            raise ValueError("a mapping source is only valid for the PYPOWER format")
        return from_pypower(source)
    path = Path(source)
    selected = _format(path, format)
    if selected == "json":
        return read_json(path)
    if selected == "mat":
        return read_mat(path, variable_name=variable_name)
    if selected == "npz":
        return read_npz(path)
    return read_excel(path)


def save(
    case: MatpowerCase | Mapping[str, Any],
    destination: str | PathLike[str],
    *,
    format: str | None = None,
    variable_name: str = "mpc",
) -> Path:
    """Save a case using a suffix-selected or explicitly selected format."""
    path = Path(destination)
    selected = _format(path, format)
    if selected == "json":
        return write_json(case, path)
    if selected == "mat":
        return write_mat(case, path, variable_name=variable_name)
    if selected == "npz":
        return write_npz(case, path)
    return write_excel(case, path)


def read_case(
    source: str | PathLike[str] | Mapping[str, Any],
    *,
    format: str | None = None,
    variable_name: str = "mpc",
) -> MatpowerCase:
    """Alias for :func:`load`."""
    return load(source, format=format, variable_name=variable_name)


def write_case(
    case: MatpowerCase | Mapping[str, Any],
    destination: str | PathLike[str],
    *,
    format: str | None = None,
    variable_name: str = "mpc",
) -> Path:
    """Alias for :func:`save`."""
    return save(case, destination, format=format, variable_name=variable_name)


__all__ = [
    "from_pypower",
    "load",
    "read_case",
    "read_excel",
    "read_json",
    "read_mat",
    "read_npz",
    "to_pypower",
    "save",
    "write_case",
    "write_excel",
    "write_json",
    "write_mat",
    "write_npz",
]
