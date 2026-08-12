# Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause
"""Public build and solve API for the SOCWR formulation."""

from __future__ import annotations

from typing import Any

from .dispatch import solve_built_power_model, solve_formulation_opf
from .pyomo import PyomoPowerModel, build_socwr_power_model, import_pyomo

SOCWRModel = PyomoPowerModel


def build_socwr_model(mpc: Any) -> SOCWRModel:
    """Build a non-conic SOCWR model without solving it."""
    return build_socwr_power_model(mpc, import_pyomo())


def solve_socwr_model(problem: SOCWRModel, mpopt: Any, nargout: int = 1):
    """Solve a previously constructed and optionally extended SOCWR model."""
    return solve_built_power_model(problem, mpopt, nargout, expected_formulation="SOCWR")


def solve_socwr_opf(mpc: Any, mpopt: Any, nargout: int = 1):
    """Build and solve a MATPOWER case with non-conic SOCWR."""
    return solve_formulation_opf(mpc, mpopt, "SOCWR", nargout)


__all__ = [
    "SOCWRModel",
    "build_socwr_model",
    "solve_socwr_model",
    "solve_socwr_opf",
]
