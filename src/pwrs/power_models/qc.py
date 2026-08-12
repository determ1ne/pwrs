# Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause
"""Public APIs for the QCRM and QCLS formulations."""

from __future__ import annotations

from typing import Any

from .dispatch import solve_built_power_model, solve_formulation_opf
from .pyomo import (
    PyomoPowerModel,
    build_qcls_power_model,
    build_qcrm_power_model,
    import_pyomo,
)

QCRMModel = PyomoPowerModel
QCLSModel = PyomoPowerModel


def build_qcrm_model(mpc: Any) -> QCRMModel:
    """Build a QCRM model without solving it."""
    return build_qcrm_power_model(mpc, import_pyomo())


def solve_qcrm_model(problem: QCRMModel, mpopt: Any, nargout: int = 1):
    """Solve a previously constructed and optionally extended QCRM model."""
    return solve_built_power_model(problem, mpopt, nargout, expected_formulation="QCRM")


def solve_qcrm_opf(mpc: Any, mpopt: Any, nargout: int = 1):
    """Build and solve a MATPOWER case with QCRM."""
    return solve_formulation_opf(mpc, mpopt, "QCRM", nargout)


def build_qcls_model(mpc: Any) -> QCLSModel:
    """Build a QCLS model without solving it."""
    return build_qcls_power_model(mpc, import_pyomo())


def solve_qcls_model(problem: QCLSModel, mpopt: Any, nargout: int = 1):
    """Solve a previously constructed and optionally extended QCLS model."""
    return solve_built_power_model(problem, mpopt, nargout, expected_formulation="QCLS")


def solve_qcls_opf(mpc: Any, mpopt: Any, nargout: int = 1):
    """Build and solve a MATPOWER case with QCLS."""
    return solve_formulation_opf(mpc, mpopt, "QCLS", nargout)


__all__ = [
    "QCLSModel",
    "QCRMModel",
    "build_qcls_model",
    "build_qcrm_model",
    "solve_qcls_model",
    "solve_qcrm_model",
    "solve_qcls_opf",
    "solve_qcrm_opf",
]
