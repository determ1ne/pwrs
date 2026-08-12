# Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause
"""AC-rectangular voltage variables and network equations for Pyomo OPF."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...network import PowerNetwork
from ...options import validate_ac_power_opf_options
from ...results import PowerModelSolution, build_matpower_result
from ..common import (
    add_apparent_power_limits,
    add_branch_power_equations,
    add_branch_power_variables,
    add_dcline_power_variables,
    add_generator_power_variables,
    add_power_balance_constraints,
)
from ..context import PyomoPowerModel
from ..opf import PyomoOpfSpec, build_opf_problem
from ..rectangular import (
    add_rectangular_angle_constraints,
    add_rectangular_branch_voltage_expressions,
    add_rectangular_reference_constraints,
    add_rectangular_voltage_magnitude_constraints,
    add_rectangular_voltage_variables,
)


def _validate_acr_options(mpopt: Any) -> None:
    validate_ac_power_opf_options(mpopt, "ACR")


def _build_acr_result(
    network: PowerNetwork,
    solution: PowerModelSolution,
    objective: float,
    success: int,
    info: dict[str, Any],
):
    vr = solution.variables["vr"]
    vi = solution.variables["vi"]
    vm = np.hypot(vr, vi)
    variables = dict(solution.variables)
    variables.update({"va": np.arctan2(vi, vr), "vm": vm})
    constraint_multipliers = dict(solution.constraint_multipliers)
    pair_real = np.asarray([vr[f_bus] * vr[t_bus] + vi[f_bus] * vi[t_bus] for f_bus, t_bus in network.angle_pairs])
    if "angle_upper" in constraint_multipliers:
        constraint_multipliers["angle_upper"] = (
            constraint_multipliers["angle_upper"] * pair_real / np.cos(network.angle_max) ** 2
        )
    if "angle_lower" in constraint_multipliers:
        constraint_multipliers["angle_lower"] = (
            -constraint_multipliers["angle_lower"] * pair_real / np.cos(network.angle_min) ** 2
        )
    lower_bounds = dict(solution.lower_bound_multipliers)
    upper_bounds = dict(solution.upper_bound_multipliers)
    voltage_lower = constraint_multipliers.get("voltage_lower", np.zeros(len(vm)))
    voltage_upper = constraint_multipliers.get("voltage_upper", np.zeros(len(vm)))
    lower_bounds["vm"] = -2 * vm * voltage_lower
    upper_bounds["vm"] = (
        2 * vm * voltage_upper
        + solution.lower_bound_multipliers["vr"]
        + solution.upper_bound_multipliers["vr"]
        + solution.lower_bound_multipliers["vi"]
        + solution.upper_bound_multipliers["vi"]
    )
    mapped = PowerModelSolution(
        vector=solution.vector,
        variables=variables,
        constraint_multipliers=constraint_multipliers,
        lower_bound_multipliers=lower_bounds,
        upper_bound_multipliers=upper_bounds,
    )
    return build_matpower_result(
        network,
        mapped,
        objective,
        success,
        info,
        formulation="ACR",
        implementation="PYOMO",
    )


def _populate_acr(problem: PyomoPowerModel, pyo: Any) -> None:
    """Add ACR variables and network constraints to an OPF problem."""
    add_rectangular_voltage_variables(problem, pyo)
    add_rectangular_voltage_magnitude_constraints(problem, pyo)
    add_generator_power_variables(problem, pyo)
    add_dcline_power_variables(problem, pyo, reactive=True)
    add_branch_power_variables(problem, pyo)
    add_rectangular_reference_constraints(problem, pyo)
    add_power_balance_constraints(problem, pyo)
    add_rectangular_branch_voltage_expressions(problem, pyo)
    add_branch_power_equations(problem, pyo)
    add_rectangular_angle_constraints(problem, pyo)
    add_apparent_power_limits(problem, pyo)


_ACR_OPF = PyomoOpfSpec(
    name="ACR",
    populate=_populate_acr,
    result_builder=_build_acr_result,
    option_validator=_validate_acr_options,
    variable_order=("vr", "vi", "pg", "qg", "pf", "qf", "pt", "qt"),
    constraint_order=(
        "voltage_lower",
        "voltage_upper",
        "reference_angle",
        "active_balance",
        "reactive_balance",
        "branch_flow",
        "angle_upper",
        "angle_lower",
        "thermal_from",
        "thermal_to",
    ),
)


def build_acr_power_model(mpc: Any, pyo: Any) -> PyomoPowerModel:
    """Build the ACR formulation as a standard single-network OPF."""
    return build_opf_problem(mpc, pyo, _ACR_OPF)


__all__ = ["build_acr_power_model"]
