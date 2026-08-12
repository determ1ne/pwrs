# Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause
"""Public APIs for the four conic PowerModels formulations."""

from __future__ import annotations

from typing import Any

from .cvxpy import (
    CvxpyPowerModel,
    build_sdpwrm_power_model,
    build_socbf_conic_power_model,
    build_socwr_conic_power_model,
    build_sparse_sdpwrm_power_model,
)
from .dispatch import solve_built_power_model, solve_formulation_opf

SOCWRConicModel = CvxpyPowerModel
SOCBFConicModel = CvxpyPowerModel
SDPWRMModel = CvxpyPowerModel
SparseSDPWRMModel = CvxpyPowerModel


def build_socwr_conic_model(mpc: Any) -> SOCWRConicModel:
    """Build SOCWRConic without solving it."""
    return build_socwr_conic_power_model(mpc)


def build_socbf_conic_model(mpc: Any) -> SOCBFConicModel:
    """Build SOCBFConic without solving it."""
    return build_socbf_conic_power_model(mpc)


def build_sdpwrm_model(mpc: Any) -> SDPWRMModel:
    """Build the dense SDPWRM relaxation without solving it."""
    return build_sdpwrm_power_model(mpc)


def build_sparse_sdpwrm_model(mpc: Any) -> SparseSDPWRMModel:
    """Build the chordal SparseSDPWRM relaxation without solving it."""
    return build_sparse_sdpwrm_power_model(mpc)


def solve_socwr_conic_model(problem: SOCWRConicModel, mpopt: Any, nargout: int = 1):
    return solve_built_power_model(problem, mpopt, nargout, expected_formulation="SOCWRCONIC")


def solve_socbf_conic_model(problem: SOCBFConicModel, mpopt: Any, nargout: int = 1):
    return solve_built_power_model(problem, mpopt, nargout, expected_formulation="SOCBFCONIC")


def solve_sdpwrm_model(problem: SDPWRMModel, mpopt: Any, nargout: int = 1):
    return solve_built_power_model(problem, mpopt, nargout, expected_formulation="SDPWRM")


def solve_sparse_sdpwrm_model(problem: SparseSDPWRMModel, mpopt: Any, nargout: int = 1):
    return solve_built_power_model(problem, mpopt, nargout, expected_formulation="SPARSESDPWRM")


def solve_socwr_conic_opf(mpc: Any, mpopt: Any, nargout: int = 1):
    return solve_formulation_opf(mpc, mpopt, "SOCWRCONIC", nargout)


def solve_socbf_conic_opf(mpc: Any, mpopt: Any, nargout: int = 1):
    return solve_formulation_opf(mpc, mpopt, "SOCBFCONIC", nargout)


def solve_sdpwrm_opf(mpc: Any, mpopt: Any, nargout: int = 1):
    return solve_formulation_opf(mpc, mpopt, "SDPWRM", nargout)


def solve_sparse_sdpwrm_opf(mpc: Any, mpopt: Any, nargout: int = 1):
    return solve_formulation_opf(mpc, mpopt, "SPARSESDPWRM", nargout)


__all__ = [
    "SDPWRMModel",
    "SOCBFConicModel",
    "SOCWRConicModel",
    "SparseSDPWRMModel",
    "build_sdpwrm_model",
    "build_socbf_conic_model",
    "build_socwr_conic_model",
    "build_sparse_sdpwrm_model",
    "solve_sdpwrm_model",
    "solve_socbf_conic_model",
    "solve_socwr_conic_model",
    "solve_sparse_sdpwrm_model",
    "solve_sdpwrm_opf",
    "solve_socbf_conic_opf",
    "solve_socwr_conic_opf",
    "solve_sparse_sdpwrm_opf",
]
