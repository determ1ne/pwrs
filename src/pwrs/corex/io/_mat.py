# Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause
"""MATLAB MAT-file case format."""

from __future__ import annotations

import re
from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat, savemat

from ..mpc import MatpowerCase
from ._schema import STRING_LIST_FIELDS, case_mapping, normalize_case

_MATLAB_NAME = re.compile(r"^[A-Za-z]\w*$")


def _validate_variable_name(variable_name: str) -> None:
    if not _MATLAB_NAME.fullmatch(variable_name):
        raise ValueError(f"invalid MATLAB variable name: {variable_name!r}")


def _path(value: str | PathLike[str]) -> Path:
    path = Path(value)
    if not path.suffix:
        return path.with_suffix(".mat")
    if path.suffix.lower() != ".mat":
        raise ValueError(f"expected a .mat path, got {path}")
    return path


def _mat_value(value: Any) -> Any:
    if hasattr(value, "_fieldnames"):
        return {name: _mat_value(getattr(value, name)) for name in value._fieldnames}
    if isinstance(value, np.ndarray) and value.dtype.kind in "US":
        strings = [str(item) for item in value.reshape(-1)]
        return strings[0] if len(strings) == 1 else strings
    if isinstance(value, np.ndarray) and value.dtype == object:
        if value.size == 1 and hasattr(value.item(), "_fieldnames"):
            return _mat_value(value.item())
        converted = [
            "" if isinstance(item, np.ndarray) and item.size == 0 else _mat_value(item)
            for item in value.reshape(-1)
        ]
        if all(isinstance(item, str) for item in converted):
            return converted
        return np.asarray(converted, dtype=object).reshape(value.shape)
    return value


def read_mat(source: str | PathLike[str], *, variable_name: str = "mpc") -> MatpowerCase:
    """Read a MATPOWER struct from a MATLAB MAT-file."""
    _validate_variable_name(variable_name)
    path = _path(source)
    try:
        payload = loadmat(path, struct_as_record=False, squeeze_me=False)
    except NotImplementedError as exc:
        raise ValueError("MATLAB v7.3/HDF5 MAT files are not supported") from exc
    if variable_name not in payload:
        raise ValueError(f"MAT variable {variable_name!r} not found in {path}")
    value = _mat_value(payload[variable_name])
    if not isinstance(value, Mapping):
        raise ValueError(f"MAT variable {variable_name!r} is not a case struct")
    return normalize_case(value)


def write_mat(
    case: MatpowerCase | Mapping[str, Any],
    destination: str | PathLike[str],
    *,
    variable_name: str = "mpc",
) -> Path:
    """Write a case as a MATLAB struct with a configurable variable name."""
    _validate_variable_name(variable_name)
    path = _path(destination)
    data = case_mapping(case)
    for name in STRING_LIST_FIELDS:
        if name in data:
            data[name] = np.asarray(data[name], dtype=object).reshape(-1, 1)
    savemat(path, {variable_name: data}, do_compression=False, long_field_names=True)
    return path


__all__ = ["read_mat", "write_mat"]
