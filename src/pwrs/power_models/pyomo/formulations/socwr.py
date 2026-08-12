# Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause
"""SOC relaxation in rectangular W-space for Pyomo OPF."""

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
from ..w_space import (
    add_w_angle_constraints,
    add_w_branch_voltage_expressions,
    add_w_voltage_variables,
    map_w_solution,
    reconstruct_voltage_angles,
)


def _validate_socwr_options(mpopt: Any) -> None:
    validate_ac_power_opf_options(mpopt, "SOCWR")


def _build_socwr_result(
    network: PowerNetwork,
    solution: PowerModelSolution,
    objective: float,
    success: int,
    info: dict[str, Any],
):
    multipliers = dict(solution.constraint_multipliers)
    wr = solution.variables["wr"]
    if "angle_upper" in multipliers:
        multipliers["angle_upper"] = multipliers["angle_upper"] * wr / np.cos(network.angle_max) ** 2
    if "angle_lower" in multipliers:
        multipliers["angle_lower"] = -multipliers["angle_lower"] * wr / np.cos(network.angle_min) ** 2
    native = PowerModelSolution(
        vector=solution.vector,
        variables=solution.variables,
        constraint_multipliers=multipliers,
        lower_bound_multipliers=solution.lower_bound_multipliers,
        upper_bound_multipliers=solution.upper_bound_multipliers,
    )
    va, angle_residual = reconstruct_voltage_angles(network, wr, solution.variables["wi"])
    mapped = map_w_solution(native, va)
    result, raw = build_matpower_result(
        network,
        mapped,
        objective,
        success,
        info,
        formulation="SOCWR",
        implementation="PYOMO",
    )
    raw["output"].update(
        {
            "voltage_angle_source": "least_squares_from_wr_wi",
            "voltage_angle_max_pair_residual": angle_residual,
        }
    )
    return result, raw


def _add_soc_voltage_constraints(problem: PyomoPowerModel, pyo: Any) -> None:
    model, network = problem.model, problem.network

    def soc_rule(m: Any, i: int) -> Any:
        f_bus, t_bus = (int(value) for value in network.angle_pairs[i])
        return m.wr[i] ** 2 + m.wi[i] ** 2 <= m.w[f_bus] * m.w[t_bus]

    model.soc_voltage_product = pyo.Constraint(model.ANGLE_PAIR, rule=soc_rule)
    problem.register_constraints(
        "soc_voltage_product",
        tuple(model.soc_voltage_product[i] for i in model.ANGLE_PAIR),
    )


def _populate_socwr(problem: PyomoPowerModel, pyo: Any) -> None:
    """Add SOCWR variables and network constraints to an OPF problem."""
    add_w_voltage_variables(problem, pyo)
    add_generator_power_variables(problem, pyo)
    add_dcline_power_variables(problem, pyo, reactive=True)
    add_branch_power_variables(problem, pyo)
    _add_soc_voltage_constraints(problem, pyo)
    add_power_balance_constraints(problem, pyo)
    add_w_branch_voltage_expressions(problem, pyo)
    add_branch_power_equations(problem, pyo)
    add_w_angle_constraints(problem, pyo)
    add_apparent_power_limits(problem, pyo)


_SOCWR_OPF = PyomoOpfSpec(
    name="SOCWR",
    populate=_populate_socwr,
    result_builder=_build_socwr_result,
    option_validator=_validate_socwr_options,
    variable_order=("w", "wr", "wi", "pg", "qg", "pf", "qf", "pt", "qt"),
    constraint_order=(
        "soc_voltage_product",
        "active_balance",
        "reactive_balance",
        "branch_flow",
        "angle_upper",
        "angle_lower",
        "lifted_cut_upper_voltage",
        "lifted_cut_lower_voltage",
        "thermal_from",
        "thermal_to",
    ),
)


def build_socwr_power_model(mpc: Any, pyo: Any) -> PyomoPowerModel:
    """Build the SOCWR formulation as a standard single-network OPF."""
    return build_opf_problem(mpc, pyo, _SOCWR_OPF)


__all__ = ["build_socwr_power_model"]
