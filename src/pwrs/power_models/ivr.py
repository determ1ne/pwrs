# Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause
"""Public API for the IVR formulation."""

from __future__ import annotations

from typing import Any

from .dispatch import solve_built_power_model, solve_formulation_opf
from .pyomo import PyomoPowerModel, build_ivr_power_model, import_pyomo

IVRModel = PyomoPowerModel


def build_ivr_model(mpc: Any) -> IVRModel:
    """Build an IVR model without solving it."""
    return build_ivr_power_model(mpc, import_pyomo())


def solve_ivr_model(problem: IVRModel, mpopt: Any, nargout: int = 1):
    """Solve a previously constructed and optionally extended IVR model."""
    return solve_built_power_model(problem, mpopt, nargout, expected_formulation="IVR")


def solve_ivr_opf(mpc: Any, mpopt: Any, nargout: int = 1):
    """Build and solve a MATPOWER case with IVR."""
    return solve_formulation_opf(mpc, mpopt, "IVR", nargout)


__all__ = ["IVRModel", "build_ivr_model", "solve_ivr_opf", "solve_ivr_model"]
