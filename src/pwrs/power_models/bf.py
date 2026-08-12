# Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause
"""Public APIs for non-conic branch-flow formulations."""

from __future__ import annotations

from typing import Any

from .dispatch import solve_built_power_model, solve_formulation_opf
from .pyomo import (
    PyomoPowerModel,
    build_bfa_power_model,
    build_socbf_power_model,
    import_pyomo,
)

BFAModel = PyomoPowerModel
SOCBFModel = PyomoPowerModel


def build_bfa_model(mpc: Any) -> BFAModel:
    """Build a BFA model without solving it."""
    return build_bfa_power_model(mpc, import_pyomo())


def solve_bfa_model(problem: BFAModel, mpopt: Any, nargout: int = 1):
    """Solve a previously constructed and optionally extended BFA model."""
    return solve_built_power_model(problem, mpopt, nargout, expected_formulation="BFA")


def solve_bfa_opf(mpc: Any, mpopt: Any, nargout: int = 1):
    """Build and solve a MATPOWER case with BFA."""
    return solve_formulation_opf(mpc, mpopt, "BFA", nargout)


def build_socbf_model(mpc: Any) -> SOCBFModel:
    """Build a non-conic SOCBF model without solving it."""
    return build_socbf_power_model(mpc, import_pyomo())


def solve_socbf_model(problem: SOCBFModel, mpopt: Any, nargout: int = 1):
    """Solve a previously constructed and optionally extended SOCBF model."""
    return solve_built_power_model(problem, mpopt, nargout, expected_formulation="SOCBF")


def solve_socbf_opf(mpc: Any, mpopt: Any, nargout: int = 1):
    """Build and solve a MATPOWER case with non-conic SOCBF."""
    return solve_formulation_opf(mpc, mpopt, "SOCBF", nargout)


__all__ = [
    "BFAModel",
    "SOCBFModel",
    "build_bfa_model",
    "build_socbf_model",
    "solve_bfa_opf",
    "solve_bfa_model",
    "solve_socbf_model",
    "solve_socbf_opf",
]
