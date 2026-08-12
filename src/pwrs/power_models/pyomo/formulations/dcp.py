# Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause
"""Active-power injection formulations for Pyomo OPF."""

from __future__ import annotations

from typing import Any

import numpy as np

from ....core.idx_brch import BR_R, BR_X
from ...network import PowerNetwork
from ...options import validate_dc_power_opf_options
from ...results import PowerModelSolution, build_matpower_result
from ..active_power import (
    add_active_power_balance_constraints,
    add_directed_active_power_balance_constraints,
    add_directed_branch_power_variables,
    add_directed_dc_branch_constraints,
    add_linear_branch_power_equations,
    add_lossless_branch_power_variables,
)
from ..angle_space import (
    add_linear_angle_difference_constraints,
    add_reference_angle_constraints,
    add_voltage_angle_variables,
)
from ..common import (
    add_active_generator_power_variables,
    add_dcline_power_variables,
)
from ..context import PyomoPowerModel
from ..opf import PyomoOpfSpec, build_opf_problem


def _validate_dcp_options(mpopt: Any) -> None:
    validate_dc_power_opf_options(mpopt, "DCP")


def _validate_dcmp_options(mpopt: Any) -> None:
    validate_dc_power_opf_options(mpopt, "DCMP")


def _validate_nfa_options(mpopt: Any) -> None:
    validate_dc_power_opf_options(mpopt, "NFA")


def _validate_dcpll_options(mpopt: Any) -> None:
    validate_dc_power_opf_options(mpopt, "DCPLL")


def _map_active_power_solution(
    solution: PowerModelSolution,
    network: PowerNetwork,
) -> PowerModelSolution:
    nb, ng, nl = len(network.bus), len(network.gen), len(network.branch)
    variables = dict(solution.variables)
    if "p" in variables:
        p = variables["p"]
        pf = p.copy()
        pt = -p
    else:
        pf = variables["pf"]
        pt = variables["pt"]
    variables.update(
        {
            "va": variables.get("va", np.zeros(nb)),
            "vm": np.ones(nb),
            "qg": np.zeros(ng),
            "pf": pf,
            "qf": np.zeros(nl),
            "pt": pt,
            "qt": np.zeros(nl),
        }
    )
    lower = dict(solution.lower_bound_multipliers)
    upper = dict(solution.upper_bound_multipliers)
    lower.update({"vm": np.zeros(nb), "qg": np.zeros(ng)})
    upper.update({"vm": np.zeros(nb), "qg": np.zeros(ng)})
    return PowerModelSolution(
        vector=solution.vector,
        variables=variables,
        constraint_multipliers=solution.constraint_multipliers,
        lower_bound_multipliers=lower,
        upper_bound_multipliers=upper,
    )


def _build_dc_result(
    network: PowerNetwork,
    solution: PowerModelSolution,
    objective: float,
    success: int,
    info: dict[str, Any],
    formulation: str,
):
    mapped = _map_active_power_solution(solution, network)
    branch_limit = "directed_active_power" if formulation == "DCPLL" else "active_power"
    result, raw = build_matpower_result(
        network,
        mapped,
        objective,
        success,
        info,
        formulation=formulation,
        implementation="PYOMO",
        branch_limit=branch_limit,
    )
    raw["output"].update(
        {
            "voltage_magnitude_source": "fixed_one",
            "reactive_power_source": "not_modeled_zero",
            "voltage_angle_source": "not_modeled_zero" if formulation == "NFA" else "optimized",
        }
    )
    return result, raw


def _build_dcp_result(network, solution, objective, success, info):
    return _build_dc_result(network, solution, objective, success, info, "DCP")


def _build_dcmp_result(network, solution, objective, success, info):
    return _build_dc_result(network, solution, objective, success, info, "DCMP")


def _build_nfa_result(network, solution, objective, success, info):
    return _build_dc_result(network, solution, objective, success, info, "NFA")


def _build_dcpll_result(network, solution, objective, success, info):
    return _build_dc_result(network, solution, objective, success, info, "DCPLL")


def _branch_parameters(network: PowerNetwork, formulation: str) -> tuple[np.ndarray, np.ndarray]:
    resistance = network.branch[:, BR_R]
    reactance = network.branch[:, BR_X]
    if formulation == "DCP":
        denominator = resistance**2 + reactance**2
        return reactance / denominator, np.zeros(len(network.branch))
    if np.any(reactance == 0):
        raise ValueError("POWER_MODELS/DCMP does not support branches with zero reactance")
    return 1.0 / (reactance * network.tap), network.shift


def _populate_dc(problem: PyomoPowerModel, pyo: Any, formulation: str) -> None:
    network = problem.network
    if formulation != "NFA":
        add_voltage_angle_variables(problem, pyo)
    add_active_generator_power_variables(problem, pyo)
    add_dcline_power_variables(problem, pyo, reactive=False)
    if formulation == "DCPLL":
        add_directed_branch_power_variables(problem, pyo)
    else:
        add_lossless_branch_power_variables(problem, pyo)
    if formulation == "DCPLL":
        add_reference_angle_constraints(problem, pyo)
        add_directed_active_power_balance_constraints(problem, pyo)
        coefficient, _ = _branch_parameters(network, "DCP")
        add_directed_dc_branch_constraints(problem, pyo, coefficient)
        add_linear_angle_difference_constraints(problem, pyo)
    elif formulation == "NFA":
        add_active_power_balance_constraints(problem, pyo)
    else:
        add_reference_angle_constraints(problem, pyo)
        add_active_power_balance_constraints(problem, pyo)
        coefficient, angle_shift = _branch_parameters(network, formulation)
        add_linear_branch_power_equations(problem, pyo, coefficient, angle_shift)
        add_linear_angle_difference_constraints(problem, pyo)


def _populate_dcp(problem: PyomoPowerModel, pyo: Any) -> None:
    _populate_dc(problem, pyo, "DCP")


def _populate_dcmp(problem: PyomoPowerModel, pyo: Any) -> None:
    _populate_dc(problem, pyo, "DCMP")


def _populate_nfa(problem: PyomoPowerModel, pyo: Any) -> None:
    _populate_dc(problem, pyo, "NFA")


def _populate_dcpll(problem: PyomoPowerModel, pyo: Any) -> None:
    _populate_dc(problem, pyo, "DCPLL")


_DC_ANGLE_CONSTRAINTS = (
    "reference_angle",
    "active_balance",
    "branch_flow",
    "angle_upper",
    "angle_lower",
)
_DCP_OPF = PyomoOpfSpec(
    name="DCP",
    populate=_populate_dcp,
    result_builder=_build_dcp_result,
    option_validator=_validate_dcp_options,
    variable_order=("va", "pg", "p"),
    constraint_order=_DC_ANGLE_CONSTRAINTS,
)
_DCMP_OPF = PyomoOpfSpec(
    name="DCMP",
    populate=_populate_dcmp,
    result_builder=_build_dcmp_result,
    option_validator=_validate_dcmp_options,
    variable_order=("va", "pg", "p"),
    constraint_order=_DC_ANGLE_CONSTRAINTS,
)
_NFA_OPF = PyomoOpfSpec(
    name="NFA",
    populate=_populate_nfa,
    result_builder=_build_nfa_result,
    option_validator=_validate_nfa_options,
    variable_order=("pg", "p"),
    constraint_order=("active_balance",),
)
_DCPLL_OPF = PyomoOpfSpec(
    name="DCPLL",
    populate=_populate_dcpll,
    result_builder=_build_dcpll_result,
    option_validator=_validate_dcpll_options,
    variable_order=("va", "pg", "pf", "pt"),
    constraint_order=(
        "reference_angle",
        "active_balance",
        "branch_flow",
        "branch_loss",
        "angle_upper",
        "angle_lower",
    ),
)


def build_dcp_power_model(mpc: Any, pyo: Any) -> PyomoPowerModel:
    """Build the PowerModels DCP active-power-only formulation."""
    return build_opf_problem(mpc, pyo, _DCP_OPF)


def build_dcmp_power_model(mpc: Any, pyo: Any) -> PyomoPowerModel:
    """Build the MATPOWER-compatible PowerModels DCMP formulation."""
    return build_opf_problem(mpc, pyo, _DCMP_OPF)


def build_nfa_power_model(mpc: Any, pyo: Any) -> PyomoPowerModel:
    """Build the PowerModels network-flow active-power approximation."""
    return build_opf_problem(mpc, pyo, _NFA_OPF)


def build_dcpll_power_model(mpc: Any, pyo: Any) -> PyomoPowerModel:
    """Build the PowerModels DC approximation with quadratic line losses."""
    return build_opf_problem(mpc, pyo, _DCPLL_OPF)


__all__ = [
    "build_dcmp_power_model",
    "build_dcp_power_model",
    "build_dcpll_power_model",
    "build_nfa_power_model",
]
