# Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause
"""Option validation shared by PowerModels OPF formulations."""

from __future__ import annotations

from typing import Any

_ALGEBRAIC_SOLVERS = ("DEFAULT", "HIGHS", "IPOPT", "GLPK")
_CONIC_SOLVERS = ("DEFAULT", "CLARABEL", "SCS", "MOSEK")


def _validate_solver_options(mpopt: Any, prefix: str, legacy_solver: str) -> None:
    solver = str(mpopt.opf.power_models.solver).upper()
    if solver not in _ALGEBRAIC_SOLVERS:
        raise ValueError(f"{prefix} has unsupported opf.power_models.solver={solver!r}")
    legacy_solver = legacy_solver.upper()
    if legacy_solver not in _ALGEBRAIC_SOLVERS:
        raise ValueError(
            f"{prefix} requires a PowerModels-compatible solver; "
            f"set opf.power_models.solver instead of {legacy_solver!r}"
        )


def validate_ac_opf_options(mpopt: Any, formulation: str) -> None:
    """Validate options common to AC formulations."""
    prefix = f"POWER_MODELS/{formulation}"
    if str(mpopt.model).upper() != "AC":
        raise ValueError(f"{prefix} requires model='AC'")
    _validate_solver_options(mpopt, prefix, str(mpopt.opf.ac.solver))
    if str(mpopt.opf.flow_lim).upper() != "S":
        raise ValueError(f"{prefix} currently supports apparent-power branch limits only")


def validate_ac_power_opf_options(mpopt: Any, formulation: str) -> None:
    """Validate options common to AC power-balance formulations."""
    validate_ac_opf_options(mpopt, formulation)
    if float(mpopt.opf.current_balance):
        raise ValueError(f"POWER_MODELS/{formulation} uses power-balance equations")


def validate_dc_power_opf_options(mpopt: Any, formulation: str) -> None:
    """Validate options common to active-power-only formulations."""
    prefix = f"POWER_MODELS/{formulation}"
    if str(mpopt.model).upper() != "DC":
        raise ValueError(f"{prefix} requires model='DC'")
    _validate_solver_options(mpopt, prefix, str(mpopt.opf.dc.solver))
    if float(mpopt.opf.current_balance):
        raise ValueError(f"{prefix} uses power-balance equations")


def validate_conic_ac_power_opf_options(mpopt: Any, formulation: str) -> None:
    """Validate AC power-balance options for CVXPY conic formulations."""
    prefix = f"POWER_MODELS/{formulation}"
    if str(mpopt.model).upper() != "AC":
        raise ValueError(f"{prefix} requires model='AC'")
    solver = str(mpopt.opf.power_models.solver).upper()
    legacy_solver = str(mpopt.opf.ac.solver).upper()
    if solver not in _CONIC_SOLVERS or legacy_solver not in _CONIC_SOLVERS:
        selected = solver if solver != "DEFAULT" else legacy_solver
        raise ValueError(f"{prefix}/CVXPY requires a conic solver; got {selected!r}")
    if str(mpopt.opf.flow_lim).upper() != "S":
        raise ValueError(f"{prefix} currently supports apparent-power branch limits only")
    if float(mpopt.opf.current_balance):
        raise ValueError(f"{prefix} uses power-balance equations")


__all__ = [
    "validate_ac_opf_options",
    "validate_ac_power_opf_options",
    "validate_conic_ac_power_opf_options",
    "validate_dc_power_opf_options",
]
