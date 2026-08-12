# Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause
"""PYPOWER mapping conversion."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..mpc import MatpowerCase
from ._schema import case_mapping, normalize_case


def from_pypower(ppc: Mapping[str, Any]) -> MatpowerCase:
    """Convert and validate a PYPOWER-style case dictionary."""
    return normalize_case(ppc)


def to_pypower(case: MatpowerCase | Mapping[str, Any]) -> dict[str, Any]:
    """Return an independent PYPOWER-style case dictionary."""
    return case_mapping(case)


__all__ = ["from_pypower", "to_pypower"]
