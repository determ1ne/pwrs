# Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause
"""Public build and solve API for the ACP formulation."""

from __future__ import annotations

from typing import Any

from .dispatch import solve_built_power_model, solve_formulation_opf
from .pyomo import PyomoPowerModel, build_acp_power_model, import_pyomo

ACPModel = PyomoPowerModel


def build_acp_model(mpc: Any) -> ACPModel:
    """Build an ACP model without solving it."""
    return build_acp_power_model(mpc, import_pyomo())


def solve_acp_model(problem: ACPModel, mpopt: Any, nargout: int = 1):
    """Solve a previously constructed and optionally extended ACP model."""
    return solve_built_power_model(problem, mpopt, nargout, expected_formulation="ACP")


def solve_acp_opf(mpc: Any, mpopt: Any, nargout: int = 1):
    """Build and solve a MATPOWER case with ACP."""
    return solve_formulation_opf(mpc, mpopt, "ACP", nargout)


__all__ = ["ACPModel", "build_acp_model", "solve_acp_opf", "solve_acp_model"]
