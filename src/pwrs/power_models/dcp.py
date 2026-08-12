# Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause
"""Public APIs for active-power formulations."""

from __future__ import annotations

from typing import Any

from .dispatch import solve_built_power_model, solve_formulation_opf
from .pyomo import (
    PyomoPowerModel,
    build_dcp_power_model,
    build_dcpll_power_model,
    build_nfa_power_model,
    import_pyomo,
)

DCPModel = PyomoPowerModel
NFAModel = PyomoPowerModel
DCPLLModel = PyomoPowerModel


def build_dcp_model(mpc: Any) -> DCPModel:
    """Build a DCP model without solving it."""
    return build_dcp_power_model(mpc, import_pyomo())


def solve_dcp_model(problem: DCPModel, mpopt: Any, nargout: int = 1):
    """Solve a previously constructed and optionally extended DCP model."""
    return solve_built_power_model(problem, mpopt, nargout, expected_formulation="DCP")


def solve_dcp_opf(mpc: Any, mpopt: Any, nargout: int = 1):
    """Build and solve a MATPOWER case with DCP."""
    return solve_formulation_opf(mpc, mpopt, "DCP", nargout)


def build_nfa_model(mpc: Any) -> NFAModel:
    """Build an NFA model without solving it."""
    return build_nfa_power_model(mpc, import_pyomo())


def solve_nfa_model(problem: NFAModel, mpopt: Any, nargout: int = 1):
    """Solve a previously constructed and optionally extended NFA model."""
    return solve_built_power_model(problem, mpopt, nargout, expected_formulation="NFA")


def solve_nfa_opf(mpc: Any, mpopt: Any, nargout: int = 1):
    """Build and solve a MATPOWER case with NFA."""
    return solve_formulation_opf(mpc, mpopt, "NFA", nargout)


def build_dcpll_model(mpc: Any) -> DCPLLModel:
    """Build a DCPLL model without solving it."""
    return build_dcpll_power_model(mpc, import_pyomo())


def solve_dcpll_model(problem: DCPLLModel, mpopt: Any, nargout: int = 1):
    """Solve a previously constructed and optionally extended DCPLL model."""
    return solve_built_power_model(problem, mpopt, nargout, expected_formulation="DCPLL")


def solve_dcpll_opf(mpc: Any, mpopt: Any, nargout: int = 1):
    """Build and solve a MATPOWER case with DCPLL."""
    return solve_formulation_opf(mpc, mpopt, "DCPLL", nargout)


__all__ = [
    "DCPLLModel",
    "DCPModel",
    "NFAModel",
    "build_dcpll_model",
    "build_dcp_model",
    "build_nfa_model",
    "solve_dcpll_opf",
    "solve_dcp_opf",
    "solve_nfa_opf",
    "solve_dcpll_model",
    "solve_dcp_model",
    "solve_nfa_model",
]
