# Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause
"""AC-polar voltage variables and network equations for Pyomo OPF."""

from __future__ import annotations

from typing import Any

from ....core.idx_bus import VMAX, VMIN
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


def _validate_acp_options(mpopt: Any) -> None:
    validate_ac_power_opf_options(mpopt, "ACP")
    if float(mpopt.opf.v_cartesian):
        raise ValueError("POWER_MODELS/ACP uses polar voltage variables")


def _build_acp_result(network, solution: PowerModelSolution, objective, success, info):
    return build_matpower_result(
        network,
        solution,
        objective,
        success,
        info,
        formulation="ACP",
        implementation="PYOMO",
    )


def _add_voltage_variables(problem: PyomoPowerModel, pyo: Any) -> None:
    model, network = problem.model, problem.network
    add_voltage_angle_variables(problem, pyo)
    model.vm = pyo.Var(
        model.BUS,
        bounds=lambda _, i: (float(network.bus[i, VMIN - 1]), float(network.bus[i, VMAX - 1])),
    )
    for i in model.BUS:
        model.vm[i].set_value(1.0, skip_validation=True)
    problem.register_variables("vm", tuple(model.vm[i] for i in model.BUS))
    model.voltage_magnitude_squared = pyo.Expression(model.BUS, rule=lambda m, i: m.vm[i] ** 2)
    problem.register_expression("voltage_magnitude_squared", model.voltage_magnitude_squared)


def _add_branch_voltage_expressions(problem: PyomoPowerModel, pyo: Any) -> None:
    model, network = problem.model, problem.network
    model.branch_angle = pyo.Expression(
        model.BRANCH,
        rule=lambda m, i: m.va[int(network.f_bus[i])] - m.va[int(network.t_bus[i])],
    )
    model.branch_vm_product = pyo.Expression(
        model.BRANCH,
        rule=lambda m, i: m.vm[int(network.f_bus[i])] * m.vm[int(network.t_bus[i])],
    )
    model.branch_cos_angle = pyo.Expression(model.BRANCH, rule=lambda m, i: pyo.cos(m.branch_angle[i]))
    model.branch_sin_angle = pyo.Expression(model.BRANCH, rule=lambda m, i: pyo.sin(m.branch_angle[i]))
    for name in ("branch_angle", "branch_vm_product", "branch_cos_angle", "branch_sin_angle"):
        problem.register_expression(name, getattr(model, name))
    model.branch_voltage_product_real = pyo.Expression(
        model.BRANCH,
        rule=lambda m, i: m.branch_vm_product[i] * m.branch_cos_angle[i],
    )
    model.branch_voltage_product_imaginary = pyo.Expression(
        model.BRANCH,
        rule=lambda m, i: m.branch_vm_product[i] * m.branch_sin_angle[i],
    )
    problem.register_expression("branch_voltage_product_real", model.branch_voltage_product_real)
    problem.register_expression("branch_voltage_product_imaginary", model.branch_voltage_product_imaginary)


def _populate_acp(problem: PyomoPowerModel, pyo: Any) -> None:
    """Add ACP variables and network constraints to an OPF problem."""
    _add_voltage_variables(problem, pyo)
    add_generator_power_variables(problem, pyo)
    add_dcline_power_variables(problem, pyo, reactive=True)
    add_branch_power_variables(problem, pyo)
    add_reference_angle_constraints(problem, pyo)
    add_power_balance_constraints(problem, pyo)
    _add_branch_voltage_expressions(problem, pyo)
    add_branch_power_equations(problem, pyo)
    add_linear_angle_difference_constraints(problem, pyo)
    add_apparent_power_limits(problem, pyo)


_ACP_OPF = PyomoOpfSpec(
    name="ACP",
    populate=_populate_acp,
    result_builder=_build_acp_result,
    option_validator=_validate_acp_options,
    variable_order=("va", "vm", "pg", "qg", "pf", "qf", "pt", "qt"),
    constraint_order=(
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


def build_acp_power_model(mpc: Any, pyo: Any) -> PyomoPowerModel:
    """Build the ACP formulation as a standard single-network OPF."""
    return build_opf_problem(mpc, pyo, _ACP_OPF)


__all__ = ["build_acp_power_model"]
