# Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Mapping

from .optimization_types import SolverOptions


def merge_nested_options(destination: SolverOptions, source: Mapping[str, object]) -> None:
    """Merge a nested solver-option mapping after runtime shape checks."""
    for key, value in source.items():
        current = destination.get(key)
        if isinstance(value, Mapping) and isinstance(current, dict):
            merge_nested_options(current, value)
        else:
            destination[key] = value
