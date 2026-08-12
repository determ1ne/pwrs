# Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause
"""Public build and solve API for the ACR formulation."""

from __future__ import annotations

from typing import Any

from .dispatch import solve_built_power_model, solve_formulation_opf
from .pyomo import PyomoPowerModel, build_acr_power_model, import_pyomo

ACRModel = PyomoPowerModel


def build_acr_model(mpc: Any) -> ACRModel:
    """Build an ACR model without solving it."""
    return build_acr_power_model(mpc, import_pyomo())


def solve_acr_model(problem: ACRModel, mpopt: Any, nargout: int = 1):
    """Solve a previously constructed and optionally extended ACR model."""
    return solve_built_power_model(problem, mpopt, nargout, expected_formulation="ACR")


def solve_acr_opf(mpc: Any, mpopt: Any, nargout: int = 1):
    """Build and solve a MATPOWER case with ACR."""
    return solve_formulation_opf(mpc, mpopt, "ACR", nargout)


__all__ = ["ACRModel", "build_acr_model", "solve_acr_opf", "solve_acr_model"]
