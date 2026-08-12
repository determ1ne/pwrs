# Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause
"""Fast, pickle-free NumPy archive case format."""

from __future__ import annotations

from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

from ..mpc import MatpowerCase
from ._schema import MATRIX_FIELDS, SCHEMA_NAME, SCHEMA_VERSION, case_mapping, normalize_case

_SCHEMA_NAME_KEY = "__schema_name__"
_SCHEMA_VERSION_KEY = "__schema_version__"
_METADATA_KEYS = frozenset({_SCHEMA_NAME_KEY, _SCHEMA_VERSION_KEY})
_SPARSE_PARTS = ("format", "data", "indices", "indptr", "shape")


def _path(value: str | PathLike[str]) -> Path:
    path = Path(value)
    if not path.suffix:
        return path.with_suffix(".npz")
    if path.suffix.lower() != ".npz":
        raise ValueError(f"expected a .npz path, got {path}")
    return path


def _sparse_key(field: str, part: str) -> str:
    return f"{field}.__sparse__.{part}"


def _scalar(archive: Any, key: str) -> Any:
    value = archive[key]
    if value.size != 1:
        raise ValueError(f"NPZ metadata {key!r} must be scalar")
    return value.reshape(-1)[0].item()


def write_npz(
    case: MatpowerCase | Mapping[str, Any], destination: str | PathLike[str]
) -> Path:
    """Write an uncompressed NPZ archive optimized for fast local loading."""
    path = _path(destination)
    payload: dict[str, Any] = {
        _SCHEMA_NAME_KEY: np.asarray(SCHEMA_NAME),
        _SCHEMA_VERSION_KEY: np.asarray(SCHEMA_VERSION, dtype=np.int64),
    }
    for field, value in case_mapping(case).items():
        if sparse.issparse(value):
            matrix = sparse.csc_matrix(value)
            payload[_sparse_key(field, "format")] = np.asarray("csc")
            payload[_sparse_key(field, "data")] = matrix.data
            payload[_sparse_key(field, "indices")] = matrix.indices
            payload[_sparse_key(field, "indptr")] = matrix.indptr
            payload[_sparse_key(field, "shape")] = np.asarray(matrix.shape, dtype=np.int64)
        elif isinstance(value, list):
            payload[field] = np.asarray(value, dtype=np.str_)
        else:
            payload[field] = np.asarray(value)
    with path.open("wb") as stream:
        np.savez(stream, allow_pickle=False, **payload)
    return path


def _read_sparse(archive: Any, field: str, files: set[str]) -> Any:
    keys = {_sparse_key(field, part) for part in _SPARSE_PARTS}
    missing = sorted(keys.difference(files))
    if missing:
        raise ValueError(f"NPZ sparse field {field!r} is missing components: {', '.join(missing)}")
    if _scalar(archive, _sparse_key(field, "format")) != "csc":
        raise ValueError(f"NPZ sparse field {field!r} has an unsupported format")
    shape_array = np.asarray(archive[_sparse_key(field, "shape")], dtype=np.int64).reshape(-1)
    if shape_array.size != 2 or np.any(shape_array < 0):
        raise ValueError(f"NPZ sparse field {field!r} has an invalid shape")
    shape = (int(shape_array[0]), int(shape_array[1]))
    data = archive[_sparse_key(field, "data")]
    indices = archive[_sparse_key(field, "indices")]
    indptr = archive[_sparse_key(field, "indptr")]
    try:
        matrix = sparse.csc_matrix((data, indices, indptr), shape=shape)
        matrix.check_format(full_check=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"NPZ sparse field {field!r} has invalid CSC components") from exc
    return matrix


def read_npz(source: str | PathLike[str]) -> MatpowerCase:
    """Read a case archive without permitting pickled object arrays."""
    path = _path(source)
    data: dict[str, Any] = {}
    try:
        with np.load(path, allow_pickle=False) as archive:
            files = set(archive.files)
            missing_metadata = sorted(_METADATA_KEYS.difference(files))
            if missing_metadata:
                raise ValueError(f"NPZ case is missing metadata: {', '.join(missing_metadata)}")
            if (
                _scalar(archive, _SCHEMA_NAME_KEY) != SCHEMA_NAME
                or _scalar(archive, _SCHEMA_VERSION_KEY) != SCHEMA_VERSION
            ):
                raise ValueError("unsupported NPZ case schema")

            sparse_fields = {
                key.removesuffix(".__sparse__.format")
                for key in files
                if key.endswith(".__sparse__.format")
            }
            component_keys = {
                _sparse_key(field, part) for field in sparse_fields for part in _SPARSE_PARTS
            }
            unexpected_components = sorted(
                key
                for key in files.difference(_METADATA_KEYS, component_keys)
                if ".__sparse__." in key
            )
            if unexpected_components:
                raise ValueError(
                    f"NPZ case has unregistered sparse components: {', '.join(unexpected_components)}"
                )
            for field in sparse_fields:
                if field not in MATRIX_FIELDS or field in files:
                    raise ValueError(f"NPZ case has an invalid sparse field {field!r}")
                data[field] = _read_sparse(archive, field, files)
            for field in files.difference(_METADATA_KEYS, component_keys):
                data[field] = archive[field]
    except ValueError as exc:
        if "Object arrays cannot be loaded" in str(exc):
            raise ValueError("NPZ case contains a pickled object array, which is not supported") from exc
        raise
    return normalize_case(data)


__all__ = ["read_npz", "write_npz"]
