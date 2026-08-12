# Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause
"""Shared loader and registration helpers for packaged case data."""

from __future__ import annotations

import json
import sys
import types
from collections.abc import Callable
from importlib import resources
from typing import Any

from ..corex import MatpowerCase
from ..corex.io._json import decode_json_case


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


def load_case_data(case_name: str, package_name: str) -> dict[str, Any]:
    resource = case_resource(case_name, package_name)
    with resource.open("r", encoding="utf-8") as stream:
        if resource.name.endswith(".json"):
            raw_data = json.load(stream)
            return decode_json_case(raw_data).to_dict()
        raise ValueError(f"Unsupported case data format: {resource.name}")


def load_case(case_name: str, package_name: str) -> Callable[[], MatpowerCase]:
    def f() -> MatpowerCase:
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
        module.__dict__["__all__"] = [case_name]
        module.__package__ = package_name
        sys.modules.setdefault(module_name, module)
