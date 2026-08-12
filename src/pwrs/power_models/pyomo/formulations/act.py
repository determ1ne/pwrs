# Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause
"""AC W-theta voltage variables and network equations for Pyomo OPF."""

from __future__ import annotations

from typing import Any

from ...network import PowerNetwork
from ...options import validate_ac_power_opf_options
from ...results import PowerModelSolution, build_matpower_result
from ..angle_space import (
    add_linear_angle_difference_constraints,
    add_reference_angle_constraints,
    add_voltage_angle_variables,
)
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
from ..w_space import add_w_branch_voltage_expressions, add_w_voltage_variables, map_w_solution


def _validate_act_options(mpopt: Any) -> None:
    validate_ac_power_opf_options(mpopt, "ACT")


def _build_act_result(
    network: PowerNetwork,
    solution: PowerModelSolution,
    objective: float,
    success: int,
    info: dict[str, Any],
):
    mapped = map_w_solution(solution, solution.variables["va"])
    return build_matpower_result(
        network,
        mapped,
        objective,
        success,
        info,
        formulation="ACT",
        implementation="PYOMO",
    )


def _add_voltage_variables(problem: PyomoPowerModel, pyo: Any) -> None:
    add_voltage_angle_variables(problem, pyo)
    add_w_voltage_variables(problem, pyo)


def _add_voltage_model_constraints(problem: PyomoPowerModel, pyo: Any) -> None:
    model, network = problem.model, problem.network

    def product_rule(m: Any, i: int) -> Any:
        f_bus, t_bus = (int(value) for value in network.angle_pairs[i])
        return m.wr[i] ** 2 + m.wi[i] ** 2 == m.w[f_bus] * m.w[t_bus]

    def angle_link_rule(m: Any, i: int) -> Any:
        f_bus, t_bus = (int(value) for value in network.angle_pairs[i])
        return m.wi[i] == pyo.tan(m.va[f_bus] - m.va[t_bus]) * m.wr[i]

    model.voltage_product = pyo.Constraint(model.ANGLE_PAIR, rule=product_rule)
    model.angle_link = pyo.Constraint(model.ANGLE_PAIR, rule=angle_link_rule)
    problem.register_constraints("voltage_product", tuple(model.voltage_product[i] for i in model.ANGLE_PAIR))
    problem.register_constraints("angle_link", tuple(model.angle_link[i] for i in model.ANGLE_PAIR))


def _populate_act(problem: PyomoPowerModel, pyo: Any) -> None:
    """Add ACT variables and network constraints to an OPF problem."""
    _add_voltage_variables(problem, pyo)
    add_generator_power_variables(problem, pyo)
    add_dcline_power_variables(problem, pyo, reactive=True)
    add_branch_power_variables(problem, pyo)
    _add_voltage_model_constraints(problem, pyo)
    add_reference_angle_constraints(problem, pyo)
    add_power_balance_constraints(problem, pyo)
    add_w_branch_voltage_expressions(problem, pyo)
    add_branch_power_equations(problem, pyo)
    add_linear_angle_difference_constraints(problem, pyo)
    add_apparent_power_limits(problem, pyo)


_ACT_OPF = PyomoOpfSpec(
    name="ACT",
    populate=_populate_act,
    result_builder=_build_act_result,
    option_validator=_validate_act_options,
    variable_order=("va", "w", "wr", "wi", "pg", "qg", "pf", "qf", "pt", "qt"),
    constraint_order=(
        "voltage_product",
        "angle_link",
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


def build_act_power_model(mpc: Any, pyo: Any) -> PyomoPowerModel:
    """Build the ACT formulation as a standard single-network OPF."""
    return build_opf_problem(mpc, pyo, _ACT_OPF)


__all__ = ["build_act_power_model"]
