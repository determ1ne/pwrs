# Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause
"""Canonical field and type boundary shared by all case IO formats."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from numbers import Integral, Real
from typing import Any

import numpy as np
from scipy import sparse

from ..mpc import MatpowerCase
from ..results import CaseResult

SCHEMA_NAME = "pwrs.matpower_case"
SCHEMA_VERSION = 1

REQUIRED_FIELDS = ("baseMVA", "bus", "gen", "branch")
DENSE_MATRIX_FIELDS = (
    "bus",
    "gen",
    "branch",
    "gencost",
    "dcline",
    "dclinecost",
    "areas",
    "fparm",
)
MATRIX_FIELDS = ("A", "N", "H")
VECTOR_FIELDS = ("l", "u", "Cw", "z0", "zl", "zu")
STRING_LIST_FIELDS = ("gentype", "genfuel", "bus_name", "branch_name", "comments")
SCALAR_FIELDS = ("version", "baseMVA", "success", "iterations", "et", "f")
ALIASES = {"busname": "bus_name", "branchname": "branch_name"}
SERIALIZABLE_FIELDS = frozenset(
    (*DENSE_MATRIX_FIELDS, *MATRIX_FIELDS, *VECTOR_FIELDS, *STRING_LIST_FIELDS, *SCALAR_FIELDS)
)


def _source_mapping(case: MatpowerCase | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(case, MatpowerCase):
        return dict(case.to_dict())
    if isinstance(case, CaseResult):
        # Solver state such as order/raw/om/mu is not case data and can contain
        # arbitrary Python objects. A solved result is persisted as a reusable
        # MATPOWER case, including its standard result scalars.
        return {name: value for name, value in case.to_dict().items() if name in SERIALIZABLE_FIELDS}
    if isinstance(case, Mapping):
        return dict(case)
    raise TypeError("case must be a MatpowerCase or PYPOWER-style mapping")


def _canonicalize_names(data: dict[str, Any]) -> dict[str, Any]:
    for legacy, canonical in ALIASES.items():
        if legacy not in data:
            continue
        if canonical in data:
            raise ValueError(f"case contains both {canonical!r} and deprecated alias {legacy!r}")
        data[canonical] = data.pop(legacy)
    return data


def _dense_matrix(name: str, value: Any) -> np.ndarray:
    if sparse.issparse(value):
        raise TypeError(f"case field {name!r} must be a dense numeric matrix")
    array = np.asarray(value)
    if array.ndim == 0:
        raise ValueError(f"case field {name!r} must be a two-dimensional matrix")
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"case field {name!r} must be a two-dimensional numeric matrix")
    return np.array(array, copy=True)


def _matrix(name: str, value: Any) -> Any:
    if sparse.issparse(value):
        if len(value.shape) != 2:
            raise ValueError(f"case field {name!r} must be a two-dimensional matrix")
        return value.copy()
    return _dense_matrix(name, value)


def _vector(name: str, value: Any) -> np.ndarray:
    if sparse.issparse(value):
        value = value.toarray()
    array = np.asarray(value)
    if array.ndim > 2 or not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"case field {name!r} must be a numeric vector")
    if array.ndim == 2 and min(array.shape, default=0) > 1:
        raise ValueError(f"case field {name!r} must be a vector")
    return np.array(array, copy=True).reshape(-1)


def _string_list(name: str, value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    array = np.asarray(value, dtype=object).reshape(-1)
    if any(not isinstance(item, str) for item in array):
        raise TypeError(f"case field {name!r} must contain strings")
    return [str(item) for item in array]


def _real_scalar(name: str, value: Any) -> int | float:
    if isinstance(value, np.ndarray):
        if value.size != 1:
            raise TypeError(f"case field {name!r} must be a real numeric scalar")
        value = value.reshape(-1)[0]
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"case field {name!r} must be a real numeric scalar")
    scalar = value.item() if isinstance(value, np.generic) else value
    return int(scalar) if isinstance(scalar, Integral) else float(scalar)


def _normalize_scalar(name: str, value: Any) -> Any:
    if name == "version":
        return str(np.asarray(value).reshape(-1)[0])
    if name == "baseMVA":
        return float(_real_scalar(name, value))
    if name == "success":
        array = np.asarray(value)
        if array.size != 1:
            raise TypeError("case field 'success' must be a boolean scalar")
        item = array.reshape(-1)[0]
        if isinstance(item, (bool, np.bool_)):
            return bool(item)
        scalar = _real_scalar(name, value)
        if scalar not in (0, 1):
            raise ValueError("case field 'success' must be boolean or numeric 0/1")
        return bool(scalar)
    if name == "iterations":
        scalar = _real_scalar(name, value)
        if not isinstance(scalar, Integral) and not float(scalar).is_integer():
            raise ValueError("case field 'iterations' must be integral")
        return int(scalar)
    if name in {"et", "f"}:
        return float(_real_scalar(name, value))
    raise AssertionError(f"unhandled scalar field: {name}")


def normalize_case(case: MatpowerCase | Mapping[str, Any]) -> MatpowerCase:
    """Validate and copy a case into the canonical language-level schema."""
    data = _canonicalize_names(_source_mapping(case))
    unknown = sorted(set(data).difference(SERIALIZABLE_FIELDS))
    if unknown:
        raise ValueError(f"unsupported case fields for serialization: {', '.join(unknown)}")
    missing = [name for name in REQUIRED_FIELDS if name not in data]
    if missing:
        raise ValueError(f"case is missing required fields: {', '.join(missing)}")

    normalized: dict[str, Any] = {}
    for name, value in data.items():
        if value is None:
            continue
        if name in DENSE_MATRIX_FIELDS:
            normalized[name] = _dense_matrix(name, value)
        elif name in MATRIX_FIELDS:
            normalized[name] = _matrix(name, value)
        elif name in VECTOR_FIELDS:
            normalized[name] = _vector(name, value)
        elif name in STRING_LIST_FIELDS:
            normalized[name] = _string_list(name, value)
        else:
            normalized[name] = _normalize_scalar(name, value)
    return MatpowerCase.from_dict(normalized)


def case_mapping(case: MatpowerCase | Mapping[str, Any]) -> dict[str, Any]:
    """Return a deep, canonical mapping suitable for a format encoder."""
    return copy.deepcopy(normalize_case(case).to_dict())


__all__ = [
    "DENSE_MATRIX_FIELDS",
    "MATRIX_FIELDS",
    "SCALAR_FIELDS",
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "SERIALIZABLE_FIELDS",
    "STRING_LIST_FIELDS",
    "VECTOR_FIELDS",
    "case_mapping",
    "normalize_case",
]
