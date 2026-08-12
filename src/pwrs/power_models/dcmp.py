# Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause
"""Public build and solve API for the DCMP formulation."""

from __future__ import annotations

from typing import Any

from .dispatch import solve_built_power_model, solve_formulation_opf
from .pyomo import PyomoPowerModel, build_dcmp_power_model, import_pyomo

DCMPModel = PyomoPowerModel


def build_dcmp_model(mpc: Any) -> DCMPModel:
    """Build a DCMP model without solving it."""
    return build_dcmp_power_model(mpc, import_pyomo())


def solve_dcmp_model(problem: DCMPModel, mpopt: Any, nargout: int = 1):
    """Solve a previously constructed and optionally extended DCMP model."""
    return solve_built_power_model(problem, mpopt, nargout, expected_formulation="DCMP")


def solve_dcmp_opf(mpc: Any, mpopt: Any, nargout: int = 1):
    """Build and solve a MATPOWER case with DCMP."""
    return solve_formulation_opf(mpc, mpopt, "DCMP", nargout)


__all__ = ["DCMPModel", "build_dcmp_model", "solve_dcmp_opf", "solve_dcmp_model"]
