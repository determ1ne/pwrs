# Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause
"""Public build and solve API for the ACT formulation."""

from __future__ import annotations

from typing import Any

from .dispatch import solve_built_power_model, solve_formulation_opf
from .pyomo import PyomoPowerModel, build_act_power_model, import_pyomo

ACTModel = PyomoPowerModel


def build_act_model(mpc: Any) -> ACTModel:
    """Build an ACT model without solving it."""
    return build_act_power_model(mpc, import_pyomo())


def solve_act_model(problem: ACTModel, mpopt: Any, nargout: int = 1):
    """Solve a previously constructed and optionally extended ACT model."""
    return solve_built_power_model(problem, mpopt, nargout, expected_formulation="ACT")


def solve_act_opf(mpc: Any, mpopt: Any, nargout: int = 1):
    """Build and solve a MATPOWER case with ACT."""
    return solve_formulation_opf(mpc, mpopt, "ACT", nargout)


__all__ = ["ACTModel", "build_act_model", "solve_act_opf", "solve_act_model"]
