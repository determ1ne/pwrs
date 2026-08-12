# Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause
"""Pyomo components shared by bus-injection OPF formulations."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...core.idx_bus import BS, GS, PD, QD
from ...core.idx_dcline import PF as DC_PF
from ...core.idx_dcline import PT as DC_PT
from ...core.idx_dcline import QF as DC_QF
from ...core.idx_dcline import QT as DC_QT
from ...core.idx_gen import PMAX, PMIN, QMAX, QMIN
from ..costs import (
    PiecewiseLinearGeneratorCost,
    PolynomialGeneratorCost,
    dcline_costs,
    generator_costs,
)
from .context import PyomoPowerModel


def add_component_sets(problem: PyomoPowerModel, pyo: Any) -> None:
    """Add deterministic indexes for active network components."""
    model, network = problem.model, problem.network
    model.BUS = pyo.Set(initialize=range(len(network.bus)), ordered=True)
    model.GEN = pyo.Set(initialize=range(len(network.gen)), ordered=True)
    model.BRANCH = pyo.Set(initialize=range(len(network.branch)), ordered=True)
    model.DCLINE = pyo.Set(initialize=range(len(network.dcline)), ordered=True)


def add_dcline_power_variables(
    problem: PyomoPowerModel,
    pyo: Any,
    *,
    reactive: bool,
) -> None:
    """Add directed DC-line power and linear loss constraints."""
    model, network = problem.model, problem.network
    if not len(network.dcline):
        return

    def bounds(lower: float, upper: float) -> tuple[float | None, float | None]:
        return (
            float(lower) if np.isfinite(lower) else None,
            float(upper) if np.isfinite(upper) else None,
        )

    model.pdcf = pyo.Var(
        model.DCLINE,
        bounds=lambda _, i: bounds(float(network.dc_pmin_from[i]), float(network.dc_pmax_from[i])),
    )
    model.pdct = pyo.Var(
        model.DCLINE,
        bounds=lambda _, i: bounds(float(network.dc_pmin_to[i]), float(network.dc_pmax_to[i])),
    )
    for i in model.DCLINE:
        model.pdcf[i].set_value(float(network.dcline[i, DC_PF - 1] / network.base_mva), skip_validation=True)
        model.pdct[i].set_value(float(-network.dcline[i, DC_PT - 1] / network.base_mva), skip_validation=True)
    problem.register_variables("pdcf", tuple(model.pdcf[i] for i in model.DCLINE))
    problem.register_variables("pdct", tuple(model.pdct[i] for i in model.DCLINE))
    if reactive:
        model.qdcf = pyo.Var(
            model.DCLINE,
            bounds=lambda _, i: bounds(float(network.dc_qmin_from[i]), float(network.dc_qmax_from[i])),
        )
        model.qdct = pyo.Var(
            model.DCLINE,
            bounds=lambda _, i: bounds(float(network.dc_qmin_to[i]), float(network.dc_qmax_to[i])),
        )
        for i in model.DCLINE:
            model.qdcf[i].set_value(float(-network.dcline[i, DC_QF - 1] / network.base_mva), skip_validation=True)
            model.qdct[i].set_value(float(-network.dcline[i, DC_QT - 1] / network.base_mva), skip_validation=True)
        problem.register_variables("qdcf", tuple(model.qdcf[i] for i in model.DCLINE))
        problem.register_variables("qdct", tuple(model.qdct[i] for i in model.DCLINE))
    model.dcline_loss = pyo.Constraint(
        model.DCLINE,
        rule=lambda m, i: (
            (1.0 - float(network.dc_loss1[i])) * m.pdcf[i]
            + m.pdct[i]
            - float(network.dc_loss0[i])
            == 0.0
        ),
    )
    problem.register_constraints("dcline_loss", tuple(model.dcline_loss[i] for i in model.DCLINE))


def add_active_generator_power_variables(problem: PyomoPowerModel, pyo: Any) -> None:
    """Add bounded active generator-power variables."""
    model, network = problem.model, problem.network
    model.pg = pyo.Var(
        model.GEN,
        bounds=lambda _, i: (
            float(network.gen[i, PMIN - 1] / network.base_mva),
            float(network.gen[i, PMAX - 1] / network.base_mva),
        ),
    )
    for i in model.GEN:
        model.pg[i].set_value(0.0, skip_validation=True)
    problem.register_variables("pg", tuple(model.pg[i] for i in model.GEN))


def add_reactive_generator_power_variables(problem: PyomoPowerModel, pyo: Any) -> None:
    """Add bounded reactive generator-power variables."""
    model, network = problem.model, problem.network
    model.qg = pyo.Var(
        model.GEN,
        bounds=lambda _, i: (
            float(network.gen[i, QMIN - 1] / network.base_mva),
            float(network.gen[i, QMAX - 1] / network.base_mva),
        ),
    )
    for i in model.GEN:
        model.qg[i].set_value(0.0, skip_validation=True)
    problem.register_variables("qg", tuple(model.qg[i] for i in model.GEN))


def add_generator_power_variables(problem: PyomoPowerModel, pyo: Any) -> None:
    """Add bounded active and reactive generator-power variables."""
    add_active_generator_power_variables(problem, pyo)
    add_reactive_generator_power_variables(problem, pyo)


def add_branch_power_variables(problem: PyomoPowerModel, pyo: Any) -> None:
    """Add explicit directed active and reactive branch-flow variables."""
    model, network = problem.model, problem.network

    def flow_bounds(_: Any, i: int) -> tuple[float | None, float | None]:
        rating = float(network.rate[i])
        return (-rating, rating) if np.isfinite(rating) else (None, None)

    model.pf = pyo.Var(model.BRANCH, initialize=0.0, bounds=flow_bounds)
    model.qf = pyo.Var(model.BRANCH, initialize=0.0, bounds=flow_bounds)
    model.pt = pyo.Var(model.BRANCH, initialize=0.0, bounds=flow_bounds)
    model.qt = pyo.Var(model.BRANCH, initialize=0.0, bounds=flow_bounds)
    for name in ("pf", "qf", "pt", "qt"):
        component = getattr(model, name)
        problem.register_variables(name, tuple(component[i] for i in model.BRANCH))


def add_power_balance_constraints(problem: PyomoPowerModel, pyo: Any) -> None:
    """Add nodal power balances using the formulation's squared voltage expression."""
    model, network = problem.model, problem.network
    voltage_squared = model.voltage_magnitude_squared

    def active_balance_rule(m: Any, i: int) -> Any:
        return (
            float(network.bus[i, PD - 1] / network.base_mva)
            + float(network.bus[i, GS - 1] / network.base_mva) * voltage_squared[i]
            + pyo.quicksum(m.pf[j] for j in network.from_branches_at_bus[i])
            + pyo.quicksum(m.pt[j] for j in network.to_branches_at_bus[i])
            + (pyo.quicksum(m.pdcf[j] for j in network.from_dclines_at_bus[i]) if len(network.dcline) else 0.0)
            + (pyo.quicksum(m.pdct[j] for j in network.to_dclines_at_bus[i]) if len(network.dcline) else 0.0)
            - pyo.quicksum(m.pg[j] for j in network.generators_at_bus[i])
            == 0.0
        )

    def reactive_balance_rule(m: Any, i: int) -> Any:
        return (
            float(network.bus[i, QD - 1] / network.base_mva)
            - float(network.bus[i, BS - 1] / network.base_mva) * voltage_squared[i]
            + pyo.quicksum(m.qf[j] for j in network.from_branches_at_bus[i])
            + pyo.quicksum(m.qt[j] for j in network.to_branches_at_bus[i])
            + (pyo.quicksum(m.qdcf[j] for j in network.from_dclines_at_bus[i]) if len(network.dcline) else 0.0)
            + (pyo.quicksum(m.qdct[j] for j in network.to_dclines_at_bus[i]) if len(network.dcline) else 0.0)
            - pyo.quicksum(m.qg[j] for j in network.generators_at_bus[i])
            == 0.0
        )

    model.active_balance = pyo.Constraint(model.BUS, rule=active_balance_rule)
    model.reactive_balance = pyo.Constraint(model.BUS, rule=reactive_balance_rule)
    problem.register_constraints("active_balance", tuple(model.active_balance[i] for i in model.BUS))
    problem.register_constraints("reactive_balance", tuple(model.reactive_balance[i] for i in model.BUS))


def add_branch_power_equations(problem: PyomoPowerModel, pyo: Any) -> None:
    """Add common Ohm-law equations from formulation-defined voltage products."""
    model, network = problem.model, problem.network
    admittance = network.admittance
    g_ff, b_ff, g_ft, b_ft, g_tf, b_tf, g_tt, b_tt = (
        values.tolist()
        for values in (
            admittance.g_ff,
            admittance.b_ff,
            admittance.g_ft,
            admittance.b_ft,
            admittance.g_tf,
            admittance.b_tf,
            admittance.g_tt,
            admittance.b_tt,
        )
    )
    voltage_squared = model.voltage_magnitude_squared
    product_real = model.branch_voltage_product_real
    product_imaginary = model.branch_voltage_product_imaginary
    model.pf_equation = pyo.Constraint(
        model.BRANCH,
        rule=lambda m, i: (
            m.pf[i]
            == g_ff[i] * voltage_squared[int(network.f_bus[i])]
            + g_ft[i] * product_real[i]
            + b_ft[i] * product_imaginary[i]
        ),
    )
    model.qf_equation = pyo.Constraint(
        model.BRANCH,
        rule=lambda m, i: (
            m.qf[i]
            == -b_ff[i] * voltage_squared[int(network.f_bus[i])]
            - b_ft[i] * product_real[i]
            + g_ft[i] * product_imaginary[i]
        ),
    )
    model.pt_equation = pyo.Constraint(
        model.BRANCH,
        rule=lambda m, i: (
            m.pt[i]
            == g_tt[i] * voltage_squared[int(network.t_bus[i])]
            + g_tf[i] * product_real[i]
            - b_tf[i] * product_imaginary[i]
        ),
    )
    model.qt_equation = pyo.Constraint(
        model.BRANCH,
        rule=lambda m, i: (
            m.qt[i]
            == -b_tt[i] * voltage_squared[int(network.t_bus[i])]
            - b_tf[i] * product_real[i]
            - g_tf[i] * product_imaginary[i]
        ),
    )
    branch_flow = []
    for i in model.BRANCH:
        branch_flow.extend((model.pf_equation[i], model.qf_equation[i], model.pt_equation[i], model.qt_equation[i]))
    problem.register_constraints("branch_flow", tuple(branch_flow))


def _polynomial_expression(variable: Any, coefficients: np.ndarray) -> Any:
    coefficients = np.trim_zeros(np.asarray(coefficients, dtype=float), trim="f")
    if not coefficients.size:
        return 0.0
    expression: Any = float(coefficients[0])
    for coefficient in coefficients[1:]:
        expression = expression * variable + float(coefficient)
    return expression


def _add_component_cost_expression(
    problem: PyomoPowerModel,
    pyo: Any,
    *,
    costs: tuple[PolynomialGeneratorCost | PiecewiseLinearGeneratorCost, ...],
    index: Any,
    power: Any,
    component: str,
) -> Any:
    model, network = problem.model, problem.network
    pwl_components = tuple(i for i, cost in enumerate(costs) if isinstance(cost, PiecewiseLinearGeneratorCost))
    point_offsets: dict[int, tuple[int, ...]] = {}
    if not pwl_components:
        return pyo.quicksum(
            _polynomial_expression(network.base_mva * power[i], cost.coefficients)
            for i, cost in enumerate(costs)
            if isinstance(cost, PolynomialGeneratorCost)
        )
    if pwl_components:
        point = 0
        for i in pwl_components:
            cost = costs[i]
            assert isinstance(cost, PiecewiseLinearGeneratorCost)
            point_offsets[i] = tuple(range(point, point + len(cost.points)))
            point += len(cost.points)

        stem = "pg_cost" if component == "generator" else "dc_p_cost"
        component_set_name = "PWL_COST_GEN" if component == "generator" else "PWL_COST_DCLINE"
        point_set_name = "PWL_COST_POINT" if component == "generator" else "PWL_DCLINE_COST_POINT"
        setattr(model, component_set_name, pyo.Set(initialize=pwl_components, ordered=True))
        setattr(model, point_set_name, pyo.Set(initialize=range(point), ordered=True))
        component_set = getattr(model, component_set_name)
        point_set = getattr(model, point_set_name)
        lambda_name = f"{stem}_lambda"
        setattr(model, lambda_name, pyo.Var(point_set, bounds=(0.0, 1.0), initialize=0.0))
        cost_lambda = getattr(model, lambda_name)
        for i in pwl_components:
            cost_lambda[point_offsets[i][0]].set_value(1.0)
        problem.register_variables(lambda_name, tuple(cost_lambda[i] for i in point_set))

        lambda_sum_name = f"{stem}_lambda_sum"
        setattr(
            model,
            lambda_sum_name,
            pyo.Constraint(
                component_set,
                rule=lambda _, i: pyo.quicksum(cost_lambda[j] for j in point_offsets[i]) == 1.0,
            ),
        )

        def power_link_rule(_: Any, i: int) -> Any:
            cost = costs[i]
            assert isinstance(cost, PiecewiseLinearGeneratorCost)
            return pyo.quicksum(
                float(cost.points[k, 0]) * cost_lambda[j]
                for k, j in enumerate(point_offsets[i])
            ) == power[i]

        link_name = f"{stem}_link"
        setattr(model, link_name, pyo.Constraint(component_set, rule=power_link_rule))
        lambda_sum = getattr(model, lambda_sum_name)
        link = getattr(model, link_name)
        problem.register_constraints(lambda_sum_name, tuple(lambda_sum[i] for i in pwl_components))
        problem.register_constraints(link_name, tuple(link[i] for i in pwl_components))

    def cost_rule(_: Any, i: int) -> Any:
        cost = costs[i]
        if isinstance(cost, PolynomialGeneratorCost):
            return _polynomial_expression(network.base_mva * power[i], cost.coefficients)
        lambda_name = "pg_cost_lambda" if component == "generator" else "dc_p_cost_lambda"
        cost_lambda = getattr(model, lambda_name)
        return pyo.quicksum(
            float(cost.points[k, 1]) * cost_lambda[j]
            for k, j in enumerate(point_offsets[i])
        )

    expression_name = "generator_cost" if component == "generator" else "dcline_cost"
    setattr(model, expression_name, pyo.Expression(index, rule=cost_rule))
    expression = getattr(model, expression_name)
    problem.register_expression(expression_name, expression)
    return pyo.quicksum(expression[i] for i in index)


def add_generator_cost_objective(problem: PyomoPowerModel, pyo: Any) -> None:
    """Add polynomial and convex PWL generator and DC-line costs."""
    model, network = problem.model, problem.network
    objective = _add_component_cost_expression(
        problem,
        pyo,
        costs=generator_costs(network),
        index=model.GEN,
        power=model.pg,
        component="generator",
    )
    if len(network.dcline):
        objective += _add_component_cost_expression(
            problem,
            pyo,
            costs=dcline_costs(network),
            index=model.DCLINE,
            power=model.pdcf,
            component="dcline",
        )
    model.objective = pyo.Objective(expr=objective, sense=pyo.minimize)


def add_apparent_power_limits(problem: PyomoPowerModel, pyo: Any) -> None:
    """Add apparent-power thermal limits at both branch ends."""
    model, network = problem.model, problem.network
    thermal_branches = tuple(int(i) for i in np.flatnonzero(np.isfinite(network.rate)))
    model.THERMAL_BRANCH = pyo.Set(initialize=thermal_branches, ordered=True)
    model.thermal_from = pyo.Constraint(
        model.THERMAL_BRANCH,
        rule=lambda m, i: m.pf[i] ** 2 + m.qf[i] ** 2 <= float(network.rate[i] ** 2),
    )
    model.thermal_to = pyo.Constraint(
        model.THERMAL_BRANCH,
        rule=lambda m, i: m.pt[i] ** 2 + m.qt[i] ** 2 <= float(network.rate[i] ** 2),
    )
    problem.register_constraints("thermal_from", tuple(model.thermal_from[i] for i in thermal_branches))
    problem.register_constraints("thermal_to", tuple(model.thermal_to[i] for i in thermal_branches))


def add_solver_suffixes(problem: PyomoPowerModel, pyo: Any) -> None:
    """Add suffixes used to normalize duals across supported solvers."""
    model = problem.model
    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    model.rc = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    model.ipopt_zL_out = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    model.ipopt_zU_out = pyo.Suffix(direction=pyo.Suffix.IMPORT)


add_ipopt_suffixes = add_solver_suffixes


__all__ = [
    "add_active_generator_power_variables",
    "add_apparent_power_limits",
    "add_branch_power_variables",
    "add_branch_power_equations",
    "add_component_sets",
    "add_dcline_power_variables",
    "add_generator_cost_objective",
    "add_generator_power_variables",
    "add_ipopt_suffixes",
    "add_solver_suffixes",
    "add_power_balance_constraints",
    "add_reactive_generator_power_variables",
]
