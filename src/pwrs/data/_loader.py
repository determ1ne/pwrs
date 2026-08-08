# Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause
"""Shared loader and registration helpers for packaged case data."""

from __future__ import annotations

import json
import sys
import types
from importlib import resources
from typing import Any

import numpy as np

from ..corex import MatpowerCase

_ARRAY_FIELDS = {"bus", "gen", "branch", "gencost", "areas", "dcline"}


def _restore_value(name: str, value: Any) -> Any:
    if name in _ARRAY_FIELDS and isinstance(value, dict) and "data" in value:
        dtype = np.int64 if value.get("dtype") == "int64" else float
        return np.array(value["data"], dtype=dtype)
    return value


def case_resource(case_name: str, package_name: str):
    """Return the preferred packaged resource for a case.

    JSON is the canonical format.
    """
    package = resources.files(package_name)
    resource = package.joinpath(f"{case_name}.json")
    if resource.is_file():
        return resource
    raise FileNotFoundError(f"case data not found: {package_name}/{case_name}")


def discover_case_names(package_name: str, prefix: str) -> list[str]:
    """Discover case names from JSON files."""
    names = {
        path.name.removesuffix(".json")
        for path in resources.files(package_name).iterdir()
        if path.name.startswith(prefix) and path.name.endswith(".json")
    }
    return sorted(names)


def _restore_json_value(name: str, value: Any, dtypes: dict[str, Any]) -> Any:
    if name in _ARRAY_FIELDS and isinstance(value, list):
        dtype_name = dtypes.get(name, "float")
        dtype = np.dtype(dtype_name)
        return np.array(value, dtype=dtype)
    return value


def load_case_data(case_name: str, package_name: str) -> dict[str, Any]:
    resource = case_resource(case_name, package_name)
    with resource.open("r", encoding="utf-8") as stream:
        if resource.name.endswith(".json"):
            raw_data = json.load(stream)
            dtypes = raw_data.get("__dtypes__", {})
            return {
                key: _restore_json_value(key, value, dtypes)
                for key, value in raw_data.items()
                if key not in {"comments", "__dtypes__"}
            }
        raise ValueError(f"Unsupported case data format: {resource.name}")


def load_case(case_name: str, package_name: str):
    def f():
        return MatpowerCase.from_dict(load_case_data(case_name, package_name))

    f.__name__ = case_name
    return f


def _raw_case_function(case_name: str, package_name: str):
    def f():
        return load_case_data(case_name, package_name)

    f.__name__ = case_name
    return f


def register_case_modules(package_name: str, case_names: list[str], data_package_name: str | None = None) -> None:
    if data_package_name is None:
        data_package_name = package_name
    for case_name in case_names:
        module_name = f"{package_name}.{case_name}"
        module = types.ModuleType(module_name)
        module.__dict__[case_name] = _raw_case_function(case_name, data_package_name)
        module.__all__ = [case_name]
        module.__package__ = package_name
        sys.modules.setdefault(module_name, module)
