# Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause
"""Backward-compatible exports for the shared case-data loader."""

from .._loader import (
    case_resource,
    discover_case_names,
    load_case,
    load_case_data,
    register_case_modules,
)

__all__ = [
    "case_resource",
    "discover_case_names",
    "load_case",
    "load_case_data",
    "register_case_modules",
]
