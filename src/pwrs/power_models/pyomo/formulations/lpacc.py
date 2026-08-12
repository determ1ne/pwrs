# Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause
"""LPAC cold-start approximation for Pyomo OPF."""

from __future__ import annotations

from typing import Any

import numpy as np

from ....core.idx_bus import VMAX, VMIN
from ...network import PowerNetwork
from ...options import validate_ac_power_opf_options
from ...results import PowerModelSolution, build_matpower_result
from ..angle_space import (
    add_linear_angle_difference_constraints,
    add_reference_angle_constraints,
    add_voltage_angle_variables,
    branch_pair_indices,
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


def _validate_lpacc_options(mpopt: Any) -> None:
    validate_ac_power_opf_options(mpopt, "LPACC")


def _build_lpacc_result(
    network: PowerNetwork,
    solution: PowerModelSolution,
    objective: float,
    success: int,
    info: dict[str, Any],
):
    variables = dict(solution.variables)
    variables["vm"] = 1.0 + variables["phi"]
    lower = dict(solution.lower_bound_multipliers)
    upper = dict(solution.upper_bound_multipliers)
    lower["vm"] = lower["phi"]
    upper["vm"] = upper["phi"]
    mapped = PowerModelSolution(
        vector=solution.vector,
        variables=variables,
        constraint_multipliers=solution.constraint_multipliers,
        lower_bound_multipliers=lower,
        upper_bound_multipliers=upper,
    )
    return build_matpower_result(
        network,
        mapped,
        objective,
        success,
        info,
        formulation="LPACC",
        implementation="PYOMO",
    )


def _cosine_bounds(network: PowerNetwork) -> tuple[np.ndarray, np.ndarray]:
    angle_min = network.angle_min
    angle_max = network.angle_max
    cosine_min = np.empty(len(angle_min))
    cosine_max = np.empty(len(angle_min))
    positive = angle_min >= 0
    negative = angle_max <= 0
    crossing = ~(positive | negative)
    cosine_min[positive] = np.cos(angle_max[positive])
    cosine_max[positive] = np.cos(angle_min[positive])
    cosine_min[negative] = np.cos(angle_min[negative])
    cosine_max[negative] = np.cos(angle_max[negative])
    cosine_min[crossing] = np.minimum(np.cos(angle_min[crossing]), np.cos(angle_max[crossing]))
    cosine_max[crossing] = 1.0
    return cosine_min, cosine_max


def _add_lpacc_voltage_variables(problem: PyomoPowerModel, pyo: Any) -> None:
    model, network = problem.model, problem.network
    add_voltage_angle_variables(problem, pyo)
    model.phi = pyo.Var(
        model.BUS,
        bounds=lambda _, i: (
            float(network.bus[i, VMIN - 1] - 1.0),
            float(network.bus[i, VMAX - 1] - 1.0),
        ),
        initialize=0.0,
    )
    cosine_min, cosine_max = _cosine_bounds(network)
    pair_indices = tuple(range(len(network.angle_pairs)))
    model.ANGLE_PAIR = pyo.Set(initialize=pair_indices, ordered=True)
    model.cs = pyo.Var(
        model.ANGLE_PAIR,
        bounds=lambda _, i: (float(cosine_min[i]), float(cosine_max[i])),
        initialize=1.0,
    )
    problem.register_variables("phi", tuple(model.phi[i] for i in model.BUS))
    problem.register_variables("cs", tuple(model.cs[i] for i in model.ANGLE_PAIR))
    model.voltage_magnitude_squared = pyo.Expression(model.BUS, rule=lambda m, i: 1.0 + 2.0 * m.phi[i])
    problem.register_expression("voltage_magnitude_squared", model.voltage_magnitude_squared)


def _add_lpacc_voltage_model(problem: PyomoPowerModel, pyo: Any) -> None:
    model, network = problem.model, problem.network
    max_angle = np.maximum(np.abs(network.angle_min), np.abs(network.angle_max))
    cosine_coefficient = (1.0 - np.cos(max_angle)) / max_angle**2
    model.cosine_envelope = pyo.Constraint(
        model.ANGLE_PAIR,
        rule=lambda m, i: (
            m.cs[i]
            <= 1.0
            - float(cosine_coefficient[i])
            * (m.va[int(network.angle_pairs[i, 0])] - m.va[int(network.angle_pairs[i, 1])]) ** 2
        ),
    )
    problem.register_constraints("cosine_envelope", tuple(model.cosine_envelope[i] for i in model.ANGLE_PAIR))


def _add_lpacc_branch_voltage_expressions(problem: PyomoPowerModel, pyo: Any) -> None:
    model, network = problem.model, problem.network
    branch_pairs = branch_pair_indices(network)
    model.branch_voltage_product_real = pyo.Expression(
        model.BRANCH,
        rule=lambda m, i: m.cs[int(branch_pairs[i])] + m.phi[int(network.f_bus[i])] + m.phi[int(network.t_bus[i])],
    )
    model.branch_voltage_product_imaginary = pyo.Expression(
        model.BRANCH,
        rule=lambda m, i: m.va[int(network.f_bus[i])] - m.va[int(network.t_bus[i])],
    )
    problem.register_expression("branch_voltage_product_real", model.branch_voltage_product_real)
    problem.register_expression("branch_voltage_product_imaginary", model.branch_voltage_product_imaginary)


def _populate_lpacc(problem: PyomoPowerModel, pyo: Any) -> None:
    """Add LPACC variables and network constraints to an OPF problem."""
    _add_lpacc_voltage_variables(problem, pyo)
    add_generator_power_variables(problem, pyo)
    add_dcline_power_variables(problem, pyo, reactive=True)
    add_branch_power_variables(problem, pyo)
    _add_lpacc_voltage_model(problem, pyo)
    add_reference_angle_constraints(problem, pyo)
    add_power_balance_constraints(problem, pyo)
    _add_lpacc_branch_voltage_expressions(problem, pyo)
    add_branch_power_equations(problem, pyo)
    add_linear_angle_difference_constraints(problem, pyo)
    add_apparent_power_limits(problem, pyo)


_LPACC_OPF = PyomoOpfSpec(
    name="LPACC",
    populate=_populate_lpacc,
    result_builder=_build_lpacc_result,
    option_validator=_validate_lpacc_options,
    variable_order=("va", "phi", "cs", "pg", "qg", "pf", "qf", "pt", "qt"),
    constraint_order=(
        "cosine_envelope",
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


def build_lpacc_power_model(mpc: Any, pyo: Any) -> PyomoPowerModel:
    """Build the LPACC formulation as a standard single-network OPF."""
    return build_opf_problem(mpc, pyo, _LPACC_OPF)


__all__ = ["build_lpacc_power_model"]
