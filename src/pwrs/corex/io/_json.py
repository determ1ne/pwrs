# Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause
"""Readable, dtype-preserving JSON case format."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

from ..mpc import MatpowerCase
from ._schema import (
    DENSE_MATRIX_FIELDS,
    MATRIX_FIELDS,
    SCHEMA_NAME,
    SCHEMA_VERSION,
    VECTOR_FIELDS,
    case_mapping,
    normalize_case,
)


def _path(value: str | PathLike[str], suffix: str) -> Path:
    path = Path(value)
    if not path.suffix:
        return path.with_suffix(suffix)
    if path.suffix.lower() != suffix:
        raise ValueError(f"expected a {suffix} path, got {path}")
    return path


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _format_json(value: Any, level: int = 0) -> str:
    indent = "  " * level
    child_indent = "  " * (level + 1)
    if isinstance(value, dict):
        if not value:
            return "{}"
        items = [
            f"{child_indent}{json.dumps(str(key), ensure_ascii=False)}: {_format_json(item, level + 1)}"
            for key, item in value.items()
        ]
        return "{\n" + ",\n".join(items) + f"\n{indent}}}"
    if isinstance(value, list):
        if not value:
            return "[]"
        if all(isinstance(item, list) for item in value):
            rows = [json.dumps(item, ensure_ascii=False, allow_nan=False) for item in value]
            return "[\n" + ",\n".join(f"{child_indent}{row}" for row in rows) + f"\n{indent}]"
        return "[\n" + ",\n".join(
            f"{child_indent}{_format_json(item, level + 1)}" for item in value
        ) + f"\n{indent}]"
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def _encode_mapping(case: MatpowerCase | Mapping[str, Any]) -> dict[str, Any]:
    data = case_mapping(case)
    encoded: dict[str, Any] = {
        "__schema__": {"name": SCHEMA_NAME, "version": SCHEMA_VERSION},
    }
    dtypes: dict[str, str] = {}
    for name, value in data.items():
        if sparse.issparse(value):
            matrix = value.tocoo()
            encoded[name] = {
                "format": "coo",
                "shape": list(matrix.shape),
                "row": matrix.row.tolist(),
                "col": matrix.col.tolist(),
                "data": _json_safe(matrix.data.tolist()),
                "dtype": str(matrix.dtype),
            }
        elif isinstance(value, np.ndarray):
            encoded[name] = _json_safe(value.tolist())
            dtypes[name] = str(value.dtype)
        else:
            encoded[name] = _json_safe(value)
    if dtypes:
        encoded["__dtypes__"] = dtypes
    return encoded


def _decode_sparse(name: str, value: Mapping[str, Any]) -> Any:
    required = {"shape", "row", "col", "data", "dtype"}
    missing = sorted(required.difference(value))
    if missing:
        raise ValueError(f"JSON sparse field {name!r} is missing: {', '.join(missing)}")
    shape = tuple(int(item) for item in value["shape"])
    if len(shape) != 2:
        raise ValueError(f"JSON sparse field {name!r} has invalid shape")
    row = np.asarray(value["row"], dtype=np.int64)
    col = np.asarray(value["col"], dtype=np.int64)
    data = np.asarray(value["data"], dtype=np.dtype(str(value["dtype"])))
    if not (len(row) == len(col) == len(data)):
        raise ValueError(f"JSON sparse field {name!r} has inconsistent COO arrays")
    return sparse.coo_matrix((data, (row, col)), shape=shape).tocsc()


def _decode_mapping(raw: Mapping[str, Any]) -> dict[str, Any]:
    schema = raw.get("__schema__")
    if schema is not None and (
        not isinstance(schema, Mapping)
        or schema.get("name") != SCHEMA_NAME
        or schema.get("version") != SCHEMA_VERSION
    ):
        raise ValueError(f"unsupported JSON case schema: {schema!r}")
    dtypes = raw.get("__dtypes__", {})
    if not isinstance(dtypes, Mapping):
        raise ValueError("JSON __dtypes__ must be an object")
    decoded: dict[str, Any] = {}
    for name, value in raw.items():
        if name in {"__schema__", "__dtypes__"}:
            continue
        if isinstance(value, Mapping) and value.get("format") == "coo":
            decoded[name] = _decode_sparse(name, value)
        elif name in dtypes:
            decoded[name] = np.asarray(value, dtype=np.dtype(str(dtypes[name])))
        elif name in {*DENSE_MATRIX_FIELDS, *MATRIX_FIELDS, *VECTOR_FIELDS}:
            # Legacy packaged JSON has no dtype table and represents non-finite
            # values as strings. NumPy's float conversion restores both cases.
            decoded[name] = np.asarray(value, dtype=float)
        else:
            decoded[name] = value
    return decoded


def decode_json_case(raw: Mapping[str, Any]) -> MatpowerCase:
    """Decode an already parsed JSON object using the shared case schema."""
    return normalize_case(_decode_mapping(raw))


def read_json(source: str | PathLike[str]) -> MatpowerCase:
    """Read a case from the canonical or legacy packaged JSON format."""
    path = _path(source, ".json")
    with path.open("r", encoding="utf-8") as stream:
        raw = json.load(stream)
    if not isinstance(raw, Mapping):
        raise ValueError(f"JSON case root must be an object: {path}")
    return decode_json_case(raw)


def write_json(case: MatpowerCase | Mapping[str, Any], destination: str | PathLike[str]) -> Path:
    """Write a case as stable, readable, standards-compliant JSON."""
    path = _path(destination, ".json")
    path.write_text(_format_json(_encode_mapping(case)) + "\n", encoding="utf-8")
    return path


__all__ = ["read_json", "write_json"]
