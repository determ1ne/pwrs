# Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause
"""Second-order-cone W-space and branch-flow OPF formulations."""

from __future__ import annotations

from typing import Any

import numpy as np

from ....core.idx_brch import BR_R, BR_X
from ....core.idx_bus import VMIN
from ...options import validate_conic_ac_power_opf_options
from ...voltage import branch_angle_coefficients
from ..common import (
    add_branch_power_equations,
    add_power_balance_constraints,
    add_thermal_cones,
    add_w_angle_constraints,
    add_w_power_variables,
)
from ..context import CvxpyPowerModel
from ..opf import CvxpyOpfSpec, build_opf_problem
from ..results import build_branch_flow_result, build_w_result


def _validate_socwr_conic_options(mpopt: Any) -> None:
    validate_conic_ac_power_opf_options(mpopt, "SOCWRCONIC")


def _validate_socbf_conic_options(mpopt: Any) -> None:
    validate_conic_ac_power_opf_options(mpopt, "SOCBFCONIC")


def _build_socwr_result(network, solution, objective, success, info):
    return build_w_result(network, solution, objective, success, info, "SOCWRCONIC")


def _build_socbf_result(network, solution, objective, success, info):
    return build_branch_flow_result(network, solution, objective, success, info, "SOCBFCONIC")


def _populate_socwr_conic(problem: CvxpyPowerModel, cp: Any) -> None:
    """Add SOCWRConic variables, cones, and network constraints."""
    network = problem.network
    add_w_power_variables(problem, cp)
    w, wr, wi = (problem.expressions[name] for name in ("w", "wr", "wi"))
    f_bus, t_bus = network.angle_pairs[:, 0], network.angle_pairs[:, 1]
    voltage_cones = tuple(
        cp.SOC(
            w[int(f_bus[i])] + w[int(t_bus[i])],
            cp.hstack(
                [
                    2 * wr[i],
                    2 * wi[i],
                    w[int(f_bus[i])] - w[int(t_bus[i])],
                ]
            ),
        )
        for i in range(len(network.angle_pairs))
    )
    problem.register_constraints("soc_voltage_product", voltage_cones)
    add_power_balance_constraints(problem, cp)
    add_branch_power_equations(problem, cp)
    add_w_angle_constraints(problem, cp)
    add_thermal_cones(problem, cp)


_SOCWR_CONIC_OPF = CvxpyOpfSpec(
    name="SOCWRCONIC",
    cone_kind="SOCP",
    populate=_populate_socwr_conic,
    result_builder=_build_socwr_result,
    option_validator=_validate_socwr_conic_options,
    variable_order=("w", "wr", "wi", "pg", "qg", "pf", "qf", "pt", "qt"),
)


def build_socwr_conic_power_model(mpc: Any) -> CvxpyPowerModel:
    """Build PowerModels' explicitly conic SOC bus-injection relaxation."""
    return build_opf_problem(mpc, _SOCWR_CONIC_OPF)


def _add_branch_flow_equations(problem: CvxpyPowerModel, cp: Any) -> None:
    network = problem.network
    variables = problem.variables
    w = problem.expressions["w"]
    pf, qf = variables["pf"], variables["qf"]
    pt, qt, ccm = variables["pt"], variables["qt"], variables["ccm"]
    resistance = network.branch[:, BR_R]
    reactance = network.branch[:, BR_X]
    w_from = w[network.f_bus] / network.tap**2
    w_to = w[network.t_bus]
    series_current = ccm + cp.multiply(network.b_fr**2, w_from) + cp.multiply(2 * network.b_fr, qf)
    problem.register_constraints(
        "branch_active_loss",
        pf + pt == cp.multiply(resistance, series_current),
    )
    problem.register_constraints(
        "branch_reactive_loss",
        qf + qt
        == cp.multiply(reactance, series_current) - cp.multiply(network.b_fr, w_from) - cp.multiply(network.b_to, w_to),
    )
    problem.register_constraints(
        "voltage_drop",
        cp.multiply(1 - 2 * reactance * network.b_fr, w_from) - w_to
        == 2 * (cp.multiply(resistance, pf) + cp.multiply(reactance, qf))
        - cp.multiply(resistance**2 + reactance**2, series_current),
    )


def _add_branch_angle_constraints(problem: CvxpyPowerModel, cp: Any) -> None:
    network = problem.network
    variables = problem.variables
    coefficients = branch_angle_coefficients(network)
    real_w, real_p, real_q, imag_w, imag_p, imag_q = coefficients
    w_from = problem.expressions["w"][network.f_bus]
    real = cp.multiply(real_w, w_from) + cp.multiply(real_p, variables["pf"]) + cp.multiply(real_q, variables["qf"])
    imaginary = (
        cp.multiply(imag_w, w_from) + cp.multiply(imag_p, variables["pf"]) + cp.multiply(imag_q, variables["qf"])
    )
    problem.register_expression("branch_voltage_product_real", real)
    problem.register_expression("branch_voltage_product_imaginary", imaginary)
    problem.register_constraints(
        "angle_upper",
        imaginary <= cp.multiply(np.tan(network.branch_angle_max), real),
    )
    problem.register_constraints(
        "angle_lower",
        imaginary >= cp.multiply(np.tan(network.branch_angle_min), real),
    )


def _populate_socbf_conic(problem: CvxpyPowerModel, cp: Any) -> None:
    """Add SOCBFConic variables, cones, and network constraints."""
    network = problem.network
    add_w_power_variables(problem, cp, add_voltage_products=False)
    ccm = cp.Variable(len(network.branch), name="ccm")
    problem.register_variable("ccm", ccm)
    lower = ccm >= 0
    problem.register_bound("ccm", lower, len(network.branch), np.arange(len(network.branch)), upper=False)
    finite = np.flatnonzero(np.isfinite(network.rate))
    if finite.size:
        upper_values = (network.rate[finite] * network.tap[finite] / network.bus[network.f_bus[finite], VMIN]) ** 2
        upper = ccm[finite] <= upper_values
        problem.register_bound("ccm", upper, len(network.branch), finite, upper=True)
    w_from = problem.expressions["w"][network.f_bus] / network.tap**2
    current_cones = tuple(
        cp.SOC(
            w_from[i] + ccm[i],
            cp.hstack([2 * problem.variables["pf"][i], 2 * problem.variables["qf"][i], w_from[i] - ccm[i]]),
        )
        for i in range(len(network.branch))
    )
    problem.register_constraints("current_model", current_cones)
    add_power_balance_constraints(problem, cp)
    _add_branch_flow_equations(problem, cp)
    _add_branch_angle_constraints(problem, cp)
    add_thermal_cones(problem, cp)


_SOCBF_CONIC_OPF = CvxpyOpfSpec(
    name="SOCBFCONIC",
    cone_kind="SOCP",
    populate=_populate_socbf_conic,
    result_builder=_build_socbf_result,
    option_validator=_validate_socbf_conic_options,
    variable_order=("w", "pg", "qg", "pf", "qf", "pt", "qt", "ccm"),
)


def build_socbf_conic_power_model(mpc: Any) -> CvxpyPowerModel:
    """Build PowerModels' explicitly conic SOC branch-flow relaxation."""
    return build_opf_problem(mpc, _SOCBF_CONIC_OPF)


__all__ = ["build_socbf_conic_power_model", "build_socwr_conic_power_model"]
