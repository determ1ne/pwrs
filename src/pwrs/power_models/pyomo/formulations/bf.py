# Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause
"""Non-conic branch-flow formulations for Pyomo OPF."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...network import PowerNetwork
from ...options import validate_ac_power_opf_options
from ...results import PowerModelSolution, build_matpower_result
from ..angle_space import fit_voltage_angles
from ..branch_flow import (
    add_bfa_branch_constraints,
    add_branch_current_variables,
    add_branch_flow_angle_constraints,
    add_socbf_branch_constraints,
    add_socbf_current_constraints,
    branch_voltage_product_values,
)
from ..common import (
    add_apparent_power_limits,
    add_branch_power_variables,
    add_dcline_power_variables,
    add_generator_power_variables,
    add_power_balance_constraints,
)
from ..context import PyomoPowerModel
from ..opf import PyomoOpfSpec, build_opf_problem
from ..w_space import add_squared_voltage_variables, map_w_solution


def _validate_bfa_options(mpopt: Any) -> None:
    validate_ac_power_opf_options(mpopt, "BFA")


def _validate_socbf_options(mpopt: Any) -> None:
    validate_ac_power_opf_options(mpopt, "SOCBF")


def _build_bf_result(
    network: PowerNetwork,
    solution: PowerModelSolution,
    objective: float,
    success: int,
    info: dict[str, Any],
    formulation: str,
):
    real, imaginary = branch_voltage_product_values(
        network,
        solution.variables["w"],
        solution.variables["pf"],
        solution.variables["qf"],
    )
    branch_angles = np.arctan2(imaginary, real)
    edges = np.column_stack((network.f_bus, network.t_bus))
    va, angle_residual = fit_voltage_angles(network, edges, branch_angles)
    mapped = map_w_solution(solution, va)
    result, raw = build_matpower_result(
        network,
        mapped,
        objective,
        success,
        info,
        formulation=formulation,
        implementation="PYOMO",
        angle_limit="branch",
    )
    raw["output"].update(
        {
            "voltage_angle_source": "least_squares_from_branch_flow",
            "voltage_angle_max_branch_residual": angle_residual,
        }
    )
    return result, raw


def _build_bfa_result(network, solution, objective, success, info):
    return _build_bf_result(network, solution, objective, success, info, "BFA")


def _build_socbf_result(network, solution, objective, success, info):
    return _build_bf_result(network, solution, objective, success, info, "SOCBF")


def _populate_branch_flow(problem: PyomoPowerModel, pyo: Any, formulation: str) -> None:
    add_squared_voltage_variables(problem, pyo)
    add_generator_power_variables(problem, pyo)
    add_dcline_power_variables(problem, pyo, reactive=True)
    add_branch_power_variables(problem, pyo)
    if formulation == "SOCBF":
        add_branch_current_variables(problem, pyo)
    if formulation == "SOCBF":
        add_socbf_current_constraints(problem, pyo)
    add_power_balance_constraints(problem, pyo)
    if formulation == "BFA":
        add_bfa_branch_constraints(problem, pyo)
    else:
        add_socbf_branch_constraints(problem, pyo)
    add_branch_flow_angle_constraints(problem, pyo)
    add_apparent_power_limits(problem, pyo)


def _populate_bfa(problem: PyomoPowerModel, pyo: Any) -> None:
    _populate_branch_flow(problem, pyo, "BFA")


def _populate_socbf(problem: PyomoPowerModel, pyo: Any) -> None:
    _populate_branch_flow(problem, pyo, "SOCBF")


_BRANCH_FLOW_CONSTRAINTS = (
    "active_balance",
    "reactive_balance",
    "branch_active_loss",
    "branch_reactive_loss",
    "voltage_drop",
    "angle_upper",
    "angle_lower",
    "thermal_from",
    "thermal_to",
)
_BFA_OPF = PyomoOpfSpec(
    name="BFA",
    populate=_populate_bfa,
    result_builder=_build_bfa_result,
    option_validator=_validate_bfa_options,
    variable_order=("w", "pg", "qg", "pf", "qf", "pt", "qt"),
    constraint_order=_BRANCH_FLOW_CONSTRAINTS,
)
_SOCBF_OPF = PyomoOpfSpec(
    name="SOCBF",
    populate=_populate_socbf,
    result_builder=_build_socbf_result,
    option_validator=_validate_socbf_options,
    variable_order=("w", "pg", "qg", "pf", "qf", "pt", "qt", "ccm"),
    constraint_order=("current_model", *_BRANCH_FLOW_CONSTRAINTS),
)


def build_bfa_power_model(mpc: Any, pyo: Any) -> PyomoPowerModel:
    """Build the PowerModels linearized branch-flow approximation."""
    return build_opf_problem(mpc, pyo, _BFA_OPF)


def build_socbf_power_model(mpc: Any, pyo: Any) -> PyomoPowerModel:
    """Build the non-conic PowerModels SOC branch-flow relaxation."""
    return build_opf_problem(mpc, pyo, _SOCBF_OPF)


__all__ = ["build_bfa_power_model", "build_socbf_power_model"]
