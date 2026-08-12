# Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause
"""Quadratic-convex QCRM and QCLS Pyomo OPF formulations."""

from __future__ import annotations

from typing import Any

import numpy as np

from ....core.idx_brch import BR_R, BR_X
from ....core.idx_bus import VMAX, VMIN
from ...network import PowerNetwork
from ...options import validate_ac_power_opf_options
from ...results import PowerModelSolution, build_matpower_result
from ..angle_space import add_reference_angle_constraints, add_voltage_angle_variables, branch_pair_indices
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
from ..w_space import add_w_angle_constraints, add_w_branch_voltage_expressions, add_w_voltage_variables


def _validate_qcrm_options(mpopt: Any) -> None:
    validate_ac_power_opf_options(mpopt, "QCRM")


def _validate_qcls_options(mpopt: Any) -> None:
    validate_ac_power_opf_options(mpopt, "QCLS")


def _build_qc_result(
    network: PowerNetwork,
    solution: PowerModelSolution,
    objective: float,
    success: int,
    info: dict[str, Any],
    formulation: str,
):
    multipliers = dict(solution.constraint_multipliers)
    wr = solution.variables["wr"]
    multipliers["angle_upper"] = multipliers["angle_upper"] * wr / np.cos(network.angle_max) ** 2
    multipliers["angle_lower"] = -multipliers["angle_lower"] * wr / np.cos(network.angle_min) ** 2
    mapped = PowerModelSolution(
        vector=solution.vector,
        variables=solution.variables,
        constraint_multipliers=multipliers,
        lower_bound_multipliers=solution.lower_bound_multipliers,
        upper_bound_multipliers=solution.upper_bound_multipliers,
    )
    return build_matpower_result(
        network,
        mapped,
        objective,
        success,
        info,
        formulation=formulation,
        implementation="PYOMO",
    )


def _build_qcrm_result(network, solution, objective, success, info):
    return _build_qc_result(network, solution, objective, success, info, "QCRM")


def _build_qcls_result(network, solution, objective, success, info):
    return _build_qc_result(network, solution, objective, success, info, "QCLS")


def _representative_branches(network: PowerNetwork) -> np.ndarray:
    pair_for_branch = branch_pair_indices(network)
    representatives = np.full(len(network.angle_pairs), len(network.branch), dtype=int)
    np.minimum.at(representatives, pair_for_branch, np.arange(len(network.branch)))
    return representatives


def _add_qc_voltage_variables(problem: PyomoPowerModel, pyo: Any, formulation: str) -> None:
    model, network = problem.model, problem.network
    add_voltage_angle_variables(problem, pyo)
    model.vm = pyo.Var(
        model.BUS,
        bounds=lambda _, i: (
            float(network.bus[i, VMIN - 1]),
            float(network.bus[i, VMAX - 1]),
        ),
        initialize=1.0,
    )
    problem.register_variables("vm", tuple(model.vm[i] for i in model.BUS))
    add_w_voltage_variables(problem, pyo)
    model.td = pyo.Var(
        model.ANGLE_PAIR,
        bounds=lambda _, i: (float(network.angle_min[i]), float(network.angle_max[i])),
        initialize=0.0,
    )
    problem.register_variables("td", tuple(model.td[i] for i in model.ANGLE_PAIR))

    f_bus, t_bus = network.angle_pairs[:, 0], network.angle_pairs[:, 1]
    vv_lower = network.bus[f_bus, VMIN - 1] * network.bus[t_bus, VMIN - 1]
    vv_upper = network.bus[f_bus, VMAX - 1] * network.bus[t_bus, VMAX - 1]
    if formulation == "QCRM":
        model.vv = pyo.Var(
            model.ANGLE_PAIR,
            bounds=lambda _, i: (float(vv_lower[i]), float(vv_upper[i])),
            initialize=1.0,
        )
        problem.register_variables("vv", tuple(model.vv[i] for i in model.ANGLE_PAIR))
    else:
        model.QC_CORNER = pyo.RangeSet(0, 7)
        model.lambda_wr = pyo.Var(model.ANGLE_PAIR, model.QC_CORNER, bounds=(0.0, 1.0), initialize=0.0)
        model.lambda_wi = pyo.Var(model.ANGLE_PAIR, model.QC_CORNER, bounds=(0.0, 1.0), initialize=0.0)
        problem.register_variables(
            "lambda_wr",
            tuple(model.lambda_wr[i, corner] for i in model.ANGLE_PAIR for corner in model.QC_CORNER),
        )
        problem.register_variables(
            "lambda_wi",
            tuple(model.lambda_wi[i, corner] for i in model.ANGLE_PAIR for corner in model.QC_CORNER),
        )

    cosine_lower = np.minimum(np.cos(network.angle_min), np.cos(network.angle_max))
    cosine_upper = np.where(
        (network.angle_min < 0) & (network.angle_max > 0),
        1.0,
        np.maximum(np.cos(network.angle_min), np.cos(network.angle_max)),
    )
    model.cs = pyo.Var(
        model.ANGLE_PAIR,
        bounds=lambda _, i: (float(cosine_lower[i]), float(cosine_upper[i])),
        initialize=1.0,
    )
    model.si = pyo.Var(
        model.ANGLE_PAIR,
        bounds=lambda _, i: (float(np.sin(network.angle_min[i])), float(np.sin(network.angle_max[i]))),
        initialize=0.0,
    )
    problem.register_variables("cs", tuple(model.cs[i] for i in model.ANGLE_PAIR))
    problem.register_variables("si", tuple(model.si[i] for i in model.ANGLE_PAIR))

    representatives = _representative_branches(network)

    def current_bounds(_: Any, i: int) -> tuple[float, float | None]:
        branch = int(representatives[i])
        rating = float(network.rate[branch])
        if not np.isfinite(rating):
            return 0.0, None
        bus = int(network.f_bus[branch])
        upper = (rating * network.tap[branch] / network.bus[bus, VMIN - 1]) ** 2
        return 0.0, float(upper)

    model.ccm = pyo.Var(model.ANGLE_PAIR, bounds=current_bounds, initialize=0.0)
    problem.register_variables("ccm", tuple(model.ccm[i] for i in model.ANGLE_PAIR))


def _add_square_relaxation(problem: PyomoPowerModel, pyo: Any) -> None:
    model, network = problem.model, problem.network
    model.voltage_square_lower = pyo.Constraint(
        model.BUS,
        rule=lambda m, i: m.w[i] >= m.vm[i] ** 2,
    )
    model.voltage_square_upper = pyo.Constraint(
        model.BUS,
        rule=lambda m, i: (
            m.w[i]
            <= float(network.bus[i, VMIN - 1] + network.bus[i, VMAX - 1]) * m.vm[i]
            - float(network.bus[i, VMIN - 1] * network.bus[i, VMAX - 1])
        ),
    )
    problem.register_constraints(
        "voltage_square_lower",
        tuple(model.voltage_square_lower[i] for i in model.BUS),
    )
    problem.register_constraints(
        "voltage_square_upper",
        tuple(model.voltage_square_upper[i] for i in model.BUS),
    )


def _add_angle_relaxations(problem: PyomoPowerModel, pyo: Any) -> None:
    model, network = problem.model, problem.network
    model.angle_link = pyo.Constraint(
        model.ANGLE_PAIR,
        rule=lambda m, i: m.va[int(network.angle_pairs[i, 0])] - m.va[int(network.angle_pairs[i, 1])] == m.td[i],
    )
    problem.register_constraints("angle_link", tuple(model.angle_link[i] for i in model.ANGLE_PAIR))

    lower, upper = network.angle_min, network.angle_max
    maximum = np.maximum(np.abs(lower), np.abs(upper))
    tangent_slope = np.cos(maximum / 2)
    chord_slope = (np.sin(lower) - np.sin(upper)) / (lower - upper)
    chord_intercept = np.sin(lower) - chord_slope * lower
    tangent_upper_intercept = np.sin(maximum / 2) - tangent_slope * maximum / 2
    tangent_lower_intercept = tangent_slope * maximum / 2 - np.sin(maximum / 2)
    sin_upper_slope = np.where(upper <= 0, chord_slope, tangent_slope)
    sin_upper_intercept = np.where(upper <= 0, chord_intercept, tangent_upper_intercept)
    sin_lower_slope = np.where(lower >= 0, chord_slope, tangent_slope)
    sin_lower_intercept = np.where(lower >= 0, chord_intercept, tangent_lower_intercept)
    model.sine_upper = pyo.Constraint(
        model.ANGLE_PAIR,
        rule=lambda m, i: m.si[i] <= float(sin_upper_slope[i]) * m.td[i] + float(sin_upper_intercept[i]),
    )
    model.sine_lower = pyo.Constraint(
        model.ANGLE_PAIR,
        rule=lambda m, i: m.si[i] >= float(sin_lower_slope[i]) * m.td[i] + float(sin_lower_intercept[i]),
    )
    cosine_coefficient = (1.0 - np.cos(maximum)) / maximum**2
    cosine_chord_slope = (np.cos(lower) - np.cos(upper)) / (lower - upper)
    cosine_chord_intercept = np.cos(lower) - cosine_chord_slope * lower
    model.cosine_upper = pyo.Constraint(
        model.ANGLE_PAIR,
        rule=lambda m, i: m.cs[i] <= 1.0 - float(cosine_coefficient[i]) * m.td[i] ** 2,
    )
    model.cosine_lower = pyo.Constraint(
        model.ANGLE_PAIR,
        rule=lambda m, i: m.cs[i] >= float(cosine_chord_slope[i]) * m.td[i] + float(cosine_chord_intercept[i]),
    )
    for name in ("sine_upper", "sine_lower", "cosine_upper", "cosine_lower"):
        component = getattr(model, name)
        problem.register_constraints(name, tuple(component[i] for i in model.ANGLE_PAIR))


def _add_mccormick_constraints(
    problem: PyomoPowerModel,
    pyo: Any,
    name: str,
    x: Any,
    y: Any,
    z: Any,
    x_lower: np.ndarray,
    x_upper: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
) -> None:
    model = problem.model
    if not hasattr(model, "QC_ENVELOPE"):
        model.QC_ENVELOPE = pyo.RangeSet(0, 3)

    def rule(_: Any, i: int, side: int) -> Any:
        xl, xu = float(x_lower[i]), float(x_upper[i])
        yl, yu = float(y_lower[i]), float(y_upper[i])
        if side == 0:
            return z[i] >= xl * y[i] + yl * x[i] - xl * yl
        if side == 1:
            return z[i] >= xu * y[i] + yu * x[i] - xu * yu
        if side == 2:
            return z[i] <= xl * y[i] + yu * x[i] - xl * yu
        return z[i] <= xu * y[i] + yl * x[i] - xu * yl

    component = pyo.Constraint(model.ANGLE_PAIR, model.QC_ENVELOPE, rule=rule)
    setattr(model, name, component)
    problem.register_constraints(
        name,
        tuple(component[i, side] for i in model.ANGLE_PAIR for side in model.QC_ENVELOPE),
    )


def _add_qcrm_product_relaxations(problem: PyomoPowerModel, pyo: Any) -> None:
    model, network = problem.model, problem.network
    f_bus, t_bus = network.angle_pairs[:, 0], network.angle_pairs[:, 1]
    vf_lower, vf_upper = network.bus[f_bus, VMIN - 1], network.bus[f_bus, VMAX - 1]
    vt_lower, vt_upper = network.bus[t_bus, VMIN - 1], network.bus[t_bus, VMAX - 1]
    vv_lower, vv_upper = vf_lower * vt_lower, vf_upper * vt_upper
    cs_lower = np.asarray([model.cs[i].lb for i in model.ANGLE_PAIR])
    cs_upper = np.asarray([model.cs[i].ub for i in model.ANGLE_PAIR])
    si_lower = np.asarray([model.si[i].lb for i in model.ANGLE_PAIR])
    si_upper = np.asarray([model.si[i].ub for i in model.ANGLE_PAIR])
    vm_from = {i: model.vm[int(f_bus[i])] for i in model.ANGLE_PAIR}
    vm_to = {i: model.vm[int(t_bus[i])] for i in model.ANGLE_PAIR}
    _add_mccormick_constraints(
        problem,
        pyo,
        "mccormick_voltage_product",
        vm_from,
        vm_to,
        model.vv,
        vf_lower,
        vf_upper,
        vt_lower,
        vt_upper,
    )
    _add_mccormick_constraints(
        problem,
        pyo,
        "mccormick_real_product",
        model.vv,
        model.cs,
        model.wr,
        vv_lower,
        vv_upper,
        cs_lower,
        cs_upper,
    )
    _add_mccormick_constraints(
        problem,
        pyo,
        "mccormick_imaginary_product",
        model.vv,
        model.si,
        model.wi,
        vv_lower,
        vv_upper,
        si_lower,
        si_upper,
    )


def _add_trilinear_constraints(problem: PyomoPowerModel, pyo: Any, suffix: str, trig: Any, target: Any) -> None:
    model, network = problem.model, problem.network
    lambdas = getattr(model, f"lambda_{suffix}")
    f_bus, t_bus = network.angle_pairs[:, 0], network.angle_pairs[:, 1]

    def corners(i: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x_bounds = network.bus[int(f_bus[i]), [VMIN - 1, VMAX - 1]]
        y_bounds = network.bus[int(t_bus[i]), [VMIN - 1, VMAX - 1]]
        z_bounds = np.asarray([trig[i].lb, trig[i].ub])
        x_values = np.repeat(x_bounds, 4)
        y_values = np.tile(np.repeat(y_bounds, 2), 2)
        z_values = np.tile(z_bounds, 4)
        return x_values, y_values, z_values, x_values * y_values * z_values

    if not hasattr(model, "QC_TRILINEAR"):
        model.QC_TRILINEAR = pyo.RangeSet(0, 4)

    def rule(_: Any, i: int, equation: int) -> Any:
        x_values, y_values, z_values, product_values = corners(i)
        weighted = lambda values: pyo.quicksum(float(values[k]) * lambdas[i, k] for k in model.QC_CORNER)
        if equation == 0:
            return target[i] == weighted(product_values)
        if equation == 1:
            return model.vm[int(f_bus[i])] == weighted(x_values)
        if equation == 2:
            return model.vm[int(t_bus[i])] == weighted(y_values)
        if equation == 3:
            return trig[i] == weighted(z_values)
        return pyo.quicksum(lambdas[i, k] for k in model.QC_CORNER) == 1.0

    component = pyo.Constraint(model.ANGLE_PAIR, model.QC_TRILINEAR, rule=rule)
    name = f"trilinear_{suffix}"
    setattr(model, name, component)
    problem.register_constraints(
        name,
        tuple(component[i, equation] for i in model.ANGLE_PAIR for equation in model.QC_TRILINEAR),
    )


def _add_qcls_product_relaxations(problem: PyomoPowerModel, pyo: Any) -> None:
    model, network = problem.model, problem.network
    _add_trilinear_constraints(problem, pyo, "wr", model.cs, model.wr)
    _add_trilinear_constraints(problem, pyo, "wi", model.si, model.wi)
    f_bus, t_bus = network.angle_pairs[:, 0], network.angle_pairs[:, 1]

    def replicate_rule(m: Any, i: int) -> Any:
        vf_lower, vf_upper = network.bus[int(f_bus[i]), [VMIN - 1, VMAX - 1]]
        vt_lower, vt_upper = network.bus[int(t_bus[i]), [VMIN - 1, VMAX - 1]]
        products = np.repeat([vf_lower, vf_upper], 4) * np.tile(np.repeat([vt_lower, vt_upper], 2), 2)
        return pyo.quicksum(float(products[k]) * (m.lambda_wr[i, k] - m.lambda_wi[i, k]) for k in m.QC_CORNER) == 0.0

    model.product_replicates = pyo.Constraint(model.ANGLE_PAIR, rule=replicate_rule)
    problem.register_constraints(
        "product_replicates",
        tuple(model.product_replicates[i] for i in model.ANGLE_PAIR),
    )


def _add_power_magnitude_strengthening(problem: PyomoPowerModel, pyo: Any) -> None:
    model, network = problem.model, problem.network
    representatives = _representative_branches(network)
    resistance = network.branch[:, BR_R - 1]
    reactance = network.branch[:, BR_X - 1]
    conductance = resistance / (resistance**2 + reactance**2)
    susceptance = -reactance / (resistance**2 + reactance**2)
    tr = network.tap * np.cos(network.shift)
    ti = network.tap * np.sin(network.shift)

    model.power_magnitude_sqr = pyo.Constraint(
        model.ANGLE_PAIR,
        rule=lambda m, i: (
            m.pf[int(representatives[i])] ** 2 + m.qf[int(representatives[i])] ** 2
            <= m.w[int(network.f_bus[int(representatives[i])])]
            / float(network.tap[int(representatives[i])] ** 2)
            * m.ccm[i]
        ),
    )

    def link_rule(m: Any, i: int) -> Any:
        branch = int(representatives[i])
        f_bus, t_bus = int(network.f_bus[branch]), int(network.t_bus[branch])
        tap_squared = network.tap[branch] ** 2
        admittance_squared = conductance[branch] ** 2 + susceptance[branch] ** 2
        return (
            m.ccm[i]
            == admittance_squared
            * (m.w[f_bus] / tap_squared + m.w[t_bus] - 2 * (tr[branch] * m.wr[i] + ti[branch] * m.wi[i]) / tap_squared)
            - network.b_fr[branch] ** 2 * m.w[f_bus] / tap_squared
            - 2 * network.b_fr[branch] * m.qf[branch]
        )

    model.power_magnitude_link = pyo.Constraint(model.ANGLE_PAIR, rule=link_rule)
    problem.register_constraints(
        "power_magnitude_sqr",
        tuple(model.power_magnitude_sqr[i] for i in model.ANGLE_PAIR),
    )
    problem.register_constraints(
        "power_magnitude_link",
        tuple(model.power_magnitude_link[i] for i in model.ANGLE_PAIR),
    )


def _populate_qc(problem: PyomoPowerModel, pyo: Any, formulation: str) -> None:
    _add_qc_voltage_variables(problem, pyo, formulation)
    add_generator_power_variables(problem, pyo)
    add_dcline_power_variables(problem, pyo, reactive=True)
    add_branch_power_variables(problem, pyo)
    _add_square_relaxation(problem, pyo)
    _add_angle_relaxations(problem, pyo)
    if formulation == "QCRM":
        _add_qcrm_product_relaxations(problem, pyo)
    else:
        _add_qcls_product_relaxations(problem, pyo)
    _add_power_magnitude_strengthening(problem, pyo)
    add_reference_angle_constraints(problem, pyo)
    add_power_balance_constraints(problem, pyo)
    add_w_branch_voltage_expressions(problem, pyo)
    add_branch_power_equations(problem, pyo)
    add_w_angle_constraints(problem, pyo)
    add_apparent_power_limits(problem, pyo)


def _populate_qcrm(problem: PyomoPowerModel, pyo: Any) -> None:
    _populate_qc(problem, pyo, "QCRM")


def _populate_qcls(problem: PyomoPowerModel, pyo: Any) -> None:
    _populate_qc(problem, pyo, "QCLS")


_QC_CONSTRAINT_PREFIX = (
    "voltage_square_lower",
    "voltage_square_upper",
    "angle_link",
    "sine_upper",
    "sine_lower",
    "cosine_upper",
    "cosine_lower",
)
_QC_CONSTRAINT_SUFFIX = (
    "power_magnitude_sqr",
    "power_magnitude_link",
    "reference_angle",
    "active_balance",
    "reactive_balance",
    "branch_flow",
    "angle_upper",
    "angle_lower",
    "lifted_cut_upper_voltage",
    "lifted_cut_lower_voltage",
    "thermal_from",
    "thermal_to",
)
_QCRM_OPF = PyomoOpfSpec(
    name="QCRM",
    populate=_populate_qcrm,
    result_builder=_build_qcrm_result,
    option_validator=_validate_qcrm_options,
    variable_order=("va", "vm", "w", "wr", "wi", "td", "vv", "cs", "si", "ccm", "pg", "qg", "pf", "qf", "pt", "qt"),
    constraint_order=(
        *_QC_CONSTRAINT_PREFIX,
        "mccormick_voltage_product",
        "mccormick_real_product",
        "mccormick_imaginary_product",
        *_QC_CONSTRAINT_SUFFIX,
    ),
)
_QCLS_OPF = PyomoOpfSpec(
    name="QCLS",
    populate=_populate_qcls,
    result_builder=_build_qcls_result,
    option_validator=_validate_qcls_options,
    variable_order=(
        "va",
        "vm",
        "w",
        "wr",
        "wi",
        "td",
        "lambda_wr",
        "lambda_wi",
        "cs",
        "si",
        "ccm",
        "pg",
        "qg",
        "pf",
        "qf",
        "pt",
        "qt",
    ),
    constraint_order=(
        *_QC_CONSTRAINT_PREFIX,
        "trilinear_wr",
        "trilinear_wi",
        "product_replicates",
        *_QC_CONSTRAINT_SUFFIX,
    ),
)


def build_qcrm_power_model(mpc: Any, pyo: Any) -> PyomoPowerModel:
    """Build the recursive-McCormick quadratic-convex relaxation."""
    return build_opf_problem(mpc, pyo, _QCRM_OPF)


def build_qcls_power_model(mpc: Any, pyo: Any) -> PyomoPowerModel:
    """Build the lambda-strengthened quadratic-convex relaxation."""
    return build_opf_problem(mpc, pyo, _QCLS_OPF)


__all__ = ["build_qcls_power_model", "build_qcrm_power_model"]
