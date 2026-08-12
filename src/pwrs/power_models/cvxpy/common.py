# Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause
"""CVXPY components shared by conic and semidefinite OPF formulations."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse

from ...core.idx_bus import BS, GS, PD, QD, VMAX, VMIN
from ...core.idx_gen import PMAX, PMIN, QMAX, QMIN
from ..costs import (
    PiecewiseLinearGeneratorCost,
    PolynomialGeneratorCost,
    dcline_costs,
    generator_costs,
)
from ..voltage import branch_pair_indices, voltage_product_bounds
from .context import CvxpyPowerModel


def import_cvxpy() -> Any:
    try:
        import cvxpy as cp
    except ImportError as exc:
        raise ImportError("conic PowerModels formulations require cvxpy; install pwrs group 'all'") from exc
    return cp


def _register_full_bounds(
    problem: CvxpyPowerModel,
    name: str,
    variable: Any,
    lower: np.ndarray,
    upper: np.ndarray,
) -> None:
    size = int(np.size(lower))
    indices = np.arange(size)
    lower_constraint = variable >= np.asarray(lower, dtype=float)
    upper_constraint = variable <= np.asarray(upper, dtype=float)
    problem.register_bound(name, lower_constraint, size, indices, upper=False)
    problem.register_bound(name, upper_constraint, size, indices, upper=True)


def _register_finite_bounds(
    problem: CvxpyPowerModel,
    name: str,
    variable: Any,
    lower: np.ndarray,
    upper: np.ndarray,
) -> None:
    finite = np.flatnonzero(np.isfinite(lower) & np.isfinite(upper))
    if not finite.size:
        return
    lower_constraint = variable[finite] >= lower[finite]
    upper_constraint = variable[finite] <= upper[finite]
    problem.register_bound(name, lower_constraint, len(lower), finite, upper=False)
    problem.register_bound(name, upper_constraint, len(upper), finite, upper=True)


def _register_partial_bounds(
    problem: CvxpyPowerModel,
    name: str,
    variable: Any,
    lower: np.ndarray,
    upper: np.ndarray,
) -> None:
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    finite_lower = np.flatnonzero(np.isfinite(lower))
    finite_upper = np.flatnonzero(np.isfinite(upper))
    if finite_lower.size:
        problem.register_bound(
            name,
            variable[finite_lower] >= lower[finite_lower],
            len(lower),
            finite_lower,
            upper=False,
        )
    if finite_upper.size:
        problem.register_bound(
            name,
            variable[finite_upper] <= upper[finite_upper],
            len(upper),
            finite_upper,
            upper=True,
        )


def add_power_variables(problem: CvxpyPowerModel, cp: Any) -> None:
    """Add generation and directed branch-power variables."""
    network = problem.network
    ng, nl = len(network.gen), len(network.branch)
    pg = cp.Variable(ng, name="pg")
    qg = cp.Variable(ng, name="qg")
    pf = cp.Variable(nl, name="pf")
    qf = cp.Variable(nl, name="qf")
    pt = cp.Variable(nl, name="pt")
    qt = cp.Variable(nl, name="qt")
    for name, variable in (("pg", pg), ("qg", qg), ("pf", pf), ("qf", qf), ("pt", pt), ("qt", qt)):
        problem.register_variable(name, variable)
    _register_full_bounds(
        problem,
        "pg",
        pg,
        network.gen[:, PMIN] / network.base_mva,
        network.gen[:, PMAX] / network.base_mva,
    )
    _register_full_bounds(
        problem,
        "qg",
        qg,
        network.gen[:, QMIN] / network.base_mva,
        network.gen[:, QMAX] / network.base_mva,
    )
    flow_lower = np.where(np.isfinite(network.rate), -network.rate, -np.inf)
    flow_upper = np.where(np.isfinite(network.rate), network.rate, np.inf)
    for name, variable in (("pf", pf), ("qf", qf), ("pt", pt), ("qt", qt)):
        _register_finite_bounds(problem, name, variable, flow_lower, flow_upper)
    if len(network.dcline):
        count = len(network.dcline)
        pdcf = cp.Variable(count, name="pdcf")
        pdct = cp.Variable(count, name="pdct")
        qdcf = cp.Variable(count, name="qdcf")
        qdct = cp.Variable(count, name="qdct")
        for name, variable in (("pdcf", pdcf), ("pdct", pdct), ("qdcf", qdcf), ("qdct", qdct)):
            problem.register_variable(name, variable)
        _register_partial_bounds(problem, "pdcf", pdcf, network.dc_pmin_from, network.dc_pmax_from)
        _register_partial_bounds(problem, "pdct", pdct, network.dc_pmin_to, network.dc_pmax_to)
        _register_partial_bounds(problem, "qdcf", qdcf, network.dc_qmin_from, network.dc_qmax_from)
        _register_partial_bounds(problem, "qdct", qdct, network.dc_qmin_to, network.dc_qmax_to)
        problem.register_constraints(
            "dcline_loss",
            cp.multiply(1.0 - network.dc_loss1, pdcf) + pdct == network.dc_loss0,
        )


def register_w_bounds(problem: CvxpyPowerModel, w: Any, wr: Any, wi: Any) -> None:
    """Register voltage-product bounds for either variables or affine expressions."""
    network = problem.network
    _register_full_bounds(
        problem,
        "w",
        w,
        network.bus[:, VMIN] ** 2,
        network.bus[:, VMAX] ** 2,
    )
    wr_min, wr_max, wi_min, wi_max = voltage_product_bounds(network)
    _register_full_bounds(problem, "wr", wr, wr_min, wr_max)
    _register_full_bounds(problem, "wi", wi, wi_min, wi_max)


def add_w_power_variables(problem: CvxpyPowerModel, cp: Any, *, add_voltage_products: bool = True) -> None:
    """Add common W-space voltage, generation, and branch-power variables."""
    network = problem.network
    w = cp.Variable(len(network.bus), name="w")
    problem.register_variable("w", w)
    problem.register_expression("w", w)
    if add_voltage_products:
        pair_count = len(network.angle_pairs)
        wr = cp.Variable(pair_count, name="wr")
        wi = cp.Variable(pair_count, name="wi")
        problem.register_variable("wr", wr)
        problem.register_variable("wi", wi)
        problem.register_expression("wr", wr)
        problem.register_expression("wi", wi)
        register_w_bounds(problem, w, wr, wi)
    else:
        _register_full_bounds(
            problem,
            "w",
            w,
            network.bus[:, VMIN] ** 2,
            network.bus[:, VMAX] ** 2,
        )
    add_power_variables(problem, cp)


def _component_cost_terms(
    problem: CvxpyPowerModel,
    cp: Any,
    costs: tuple[PolynomialGeneratorCost | PiecewiseLinearGeneratorCost, ...],
    power: Any,
    *,
    prefix: str,
) -> list[Any]:
    network = problem.network
    pwl_components = tuple(i for i, cost in enumerate(costs) if isinstance(cost, PiecewiseLinearGeneratorCost))
    point_offsets: dict[int, np.ndarray] = {}
    cost_lambda = None
    if pwl_components:
        point = 0
        for i in pwl_components:
            cost = costs[i]
            assert isinstance(cost, PiecewiseLinearGeneratorCost)
            point_offsets[i] = np.arange(point, point + len(cost.points))
            point += len(cost.points)
        lambda_name = f"{prefix}_cost_lambda"
        cost_lambda = cp.Variable(point, name=lambda_name)
        problem.register_variable(lambda_name, cost_lambda)
        _register_full_bounds(problem, lambda_name, cost_lambda, np.zeros(point), np.ones(point))
        sum_constraints = []
        link_constraints = []
        for i in pwl_components:
            cost = costs[i]
            assert isinstance(cost, PiecewiseLinearGeneratorCost)
            indices = point_offsets[i]
            sum_constraints.append(cp.sum(cost_lambda[indices]) == 1.0)
            link_constraints.append(cost.points[:, 0] @ cost_lambda[indices] == power[i])
        problem.register_constraints(f"{prefix}_cost_lambda_sum", tuple(sum_constraints))
        problem.register_constraints(f"{prefix}_cost_link", tuple(link_constraints))

    terms = []
    for i, cost in enumerate(costs):
        if isinstance(cost, PiecewiseLinearGeneratorCost):
            assert cost_lambda is not None
            terms.append(cost.points[:, 1] @ cost_lambda[point_offsets[i]])
            continue
        assert isinstance(cost, PolynomialGeneratorCost)
        coefficients = cost.coefficients
        order = len(coefficients)
        power_mw = network.base_mva * power[i]
        if order == 1:
            terms.append(float(coefficients[0]))
        elif order == 2:
            terms.append(float(coefficients[0]) * power_mw + float(coefficients[1]))
        elif order == 3:
            if coefficients[0] < 0:
                raise ValueError("conic PowerModels formulations require convex quadratic costs")
            terms.append(
                float(coefficients[0]) * cp.square(power_mw)
                + float(coefficients[1]) * power_mw
                + float(coefficients[2])
            )
        else:
            raise NotImplementedError("conic PowerModels formulations support costs up to quadratic order")
    if pwl_components:
        expression_name = "generator_cost" if prefix == "pg" else "dcline_cost"
        problem.register_expression(expression_name, cp.hstack(terms))
    return terms


def generator_cost_expression(problem: CvxpyPowerModel, cp: Any) -> Any:
    """Build convex polynomial and PWL generator and DC-line costs."""
    network = problem.network
    pg = problem.expressions.get("pg", problem.variables["pg"])
    terms = _component_cost_terms(
        problem,
        cp,
        generator_costs(network),
        pg,
        prefix="pg",
    )
    if len(network.dcline):
        terms.extend(
            _component_cost_terms(
                problem,
                cp,
                dcline_costs(network),
                problem.variables["pdcf"],
                prefix="dc_p",
            )
        )
    return cp.sum(terms)


def add_power_balance_constraints(problem: CvxpyPowerModel, cp: Any) -> None:
    """Add sparse active and reactive nodal balances."""
    network = problem.network
    nb, ng, nl = len(network.bus), len(network.gen), len(network.branch)
    cg = sparse.coo_matrix((np.ones(ng), (network.gen_bus, np.arange(ng))), shape=(nb, ng)).tocsr()
    cf = sparse.coo_matrix((np.ones(nl), (network.f_bus, np.arange(nl))), shape=(nb, nl)).tocsr()
    ct = sparse.coo_matrix((np.ones(nl), (network.t_bus, np.arange(nl))), shape=(nb, nl)).tocsr()
    variables = problem.variables
    if len(network.dcline):
        ndc = len(network.dcline)
        cdf = sparse.coo_matrix(
            (np.ones(ndc), (network.dc_f_bus, np.arange(ndc))),
            shape=(nb, ndc),
        ).tocsr()
        cdt = sparse.coo_matrix(
            (np.ones(ndc), (network.dc_t_bus, np.arange(ndc))),
            shape=(nb, ndc),
        ).tocsr()
        dc_active = cdf @ variables["pdcf"] + cdt @ variables["pdct"]
        dc_reactive = cdf @ variables["qdcf"] + cdt @ variables["qdct"]
    else:
        dc_active = np.zeros(nb)
        dc_reactive = np.zeros(nb)
    active = (
        network.bus[:, PD] / network.base_mva
        + cp.multiply(network.bus[:, GS] / network.base_mva, problem.expressions["w"])
        + cf @ variables["pf"]
        + ct @ variables["pt"]
        + dc_active
        - cg @ variables["pg"]
        == 0
    )
    reactive = (
        network.bus[:, QD] / network.base_mva
        - cp.multiply(network.bus[:, BS] / network.base_mva, problem.expressions["w"])
        + cf @ variables["qf"]
        + ct @ variables["qt"]
        + dc_reactive
        - cg @ variables["qg"]
        == 0
    )
    problem.register_constraints("active_balance", active)
    problem.register_constraints("reactive_balance", reactive)


def add_branch_power_equations(problem: CvxpyPowerModel, cp: Any) -> None:
    """Add vectorized W-space branch Ohm equations."""
    network = problem.network
    variables = problem.variables
    w = problem.expressions["w"]
    wr = problem.expressions["wr"]
    wi = problem.expressions["wi"]
    pair = branch_pair_indices(network)
    admittance = network.admittance
    equations = (
        variables["pf"]
        == cp.multiply(admittance.g_ff, w[network.f_bus])
        + cp.multiply(admittance.g_ft, wr[pair])
        + cp.multiply(admittance.b_ft, wi[pair]),
        variables["qf"]
        == -cp.multiply(admittance.b_ff, w[network.f_bus])
        - cp.multiply(admittance.b_ft, wr[pair])
        + cp.multiply(admittance.g_ft, wi[pair]),
        variables["pt"]
        == cp.multiply(admittance.g_tt, w[network.t_bus])
        + cp.multiply(admittance.g_tf, wr[pair])
        - cp.multiply(admittance.b_tf, wi[pair]),
        variables["qt"]
        == -cp.multiply(admittance.b_tt, w[network.t_bus])
        - cp.multiply(admittance.b_tf, wr[pair])
        - cp.multiply(admittance.g_tf, wi[pair]),
    )
    problem.register_constraints("branch_flow", equations)


def add_w_angle_constraints(problem: CvxpyPowerModel, cp: Any) -> None:
    """Add PowerModels tangent angle bounds and lifted nonlinear cuts."""
    network = problem.network
    w, wr, wi = (problem.expressions[name] for name in ("w", "wr", "wi"))
    upper = wi <= cp.multiply(np.tan(network.angle_max), wr)
    lower = wi >= cp.multiply(np.tan(network.angle_min), wr)
    problem.register_constraints("angle_upper", upper)
    problem.register_constraints("angle_lower", lower)

    f_bus, t_bus = network.angle_pairs[:, 0], network.angle_pairs[:, 1]
    vf_min, vf_max = network.bus[f_bus, VMIN], network.bus[f_bus, VMAX]
    vt_min, vt_max = network.bus[t_bus, VMIN], network.bus[t_bus, VMAX]
    phi = (network.angle_max + network.angle_min) / 2
    cosine = np.cos((network.angle_max - network.angle_min) / 2)
    sf, st = vf_min + vf_max, vt_min + vt_max
    spread = vf_min * vt_min - vf_max * vt_max
    projection = cp.multiply(np.cos(phi), wr) + cp.multiply(np.sin(phi), wi)
    lifted_upper = (
        cp.multiply(sf * st, projection)
        - cp.multiply(vt_max * cosine * st, w[f_bus])
        - cp.multiply(vf_max * cosine * sf, w[t_bus])
        >= vf_max * vt_max * cosine * spread
    )
    lifted_lower = (
        cp.multiply(sf * st, projection)
        - cp.multiply(vt_min * cosine * st, w[f_bus])
        - cp.multiply(vf_min * cosine * sf, w[t_bus])
        >= -vf_min * vt_min * cosine * spread
    )
    problem.register_constraints("lifted_cut_upper_voltage", lifted_upper)
    problem.register_constraints("lifted_cut_lower_voltage", lifted_lower)


def add_thermal_cones(problem: CvxpyPowerModel, cp: Any) -> None:
    """Add one SOC apparent-power limit at each rated branch endpoint."""
    network = problem.network
    variables = problem.variables
    finite = np.flatnonzero(np.isfinite(network.rate))
    from_cones = tuple(
        cp.SOC(float(network.rate[i]), cp.hstack([variables["pf"][i], variables["qf"][i]])) for i in finite
    )
    to_cones = tuple(
        cp.SOC(float(network.rate[i]), cp.hstack([variables["pt"][i], variables["qt"][i]])) for i in finite
    )
    problem.register_constraints("thermal_from", from_cones)
    problem.register_constraints("thermal_to", to_cones)


__all__ = [
    "add_branch_power_equations",
    "add_power_variables",
    "add_power_balance_constraints",
    "add_thermal_cones",
    "add_w_angle_constraints",
    "add_w_power_variables",
    "generator_cost_expression",
    "import_cvxpy",
    "register_w_bounds",
]
