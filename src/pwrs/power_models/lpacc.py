# Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause
"""Public build and solve API for the LPACC formulation."""

from __future__ import annotations

from typing import Any

from .dispatch import solve_built_power_model, solve_formulation_opf
from .pyomo import PyomoPowerModel, build_lpacc_power_model, import_pyomo

LPACCModel = PyomoPowerModel


def build_lpacc_model(mpc: Any) -> LPACCModel:
    """Build an LPACC model without solving it."""
    return build_lpacc_power_model(mpc, import_pyomo())


def solve_lpacc_model(problem: LPACCModel, mpopt: Any, nargout: int = 1):
    """Solve a previously constructed and optionally extended LPACC model."""
    return solve_built_power_model(problem, mpopt, nargout, expected_formulation="LPACC")


def solve_lpacc_opf(mpc: Any, mpopt: Any, nargout: int = 1):
    """Build and solve a MATPOWER case with LPACC."""
    return solve_formulation_opf(mpc, mpopt, "LPACC", nargout)


__all__ = [
    "LPACCModel",
    "build_lpacc_model",
    "solve_lpacc_opf",
    "solve_lpacc_model",
]
