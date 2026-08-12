# Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause
"""Shared branch-flow components for BFA and SOCBF formulations."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...core.idx_brch import BR_R, BR_X
from ...core.idx_bus import VMIN
from ..voltage import branch_angle_coefficients, branch_voltage_product_values
from .context import PyomoPowerModel


def add_branch_current_variables(problem: PyomoPowerModel, pyo: Any) -> None:
    """Add bounded squared series-current variables for SOC branch flow."""
    model, network = problem.model, problem.network

    def current_bounds(_: Any, i: int) -> tuple[float, float | None]:
        rating = float(network.rate[i])
        if not np.isfinite(rating):
            return 0.0, None
        f_bus = int(network.f_bus[i])
        upper = (rating * network.tap[i] / network.bus[f_bus, VMIN - 1]) ** 2
        return 0.0, float(upper)

    model.ccm = pyo.Var(model.BRANCH, bounds=current_bounds, initialize=0.0)
    problem.register_variables("ccm", tuple(model.ccm[i] for i in model.BRANCH))


def add_socbf_current_constraints(problem: PyomoPowerModel, pyo: Any) -> None:
    """Add the non-conic rotated-SOC current relationship."""
    model, network = problem.model, problem.network
    model.current_model = pyo.Constraint(
        model.BRANCH,
        rule=lambda m, i: (
            m.pf[i] ** 2 + m.qf[i] ** 2 <= m.w[int(network.f_bus[i])] / float(network.tap[i] ** 2) * m.ccm[i]
        ),
    )
    problem.register_constraints("current_model", tuple(model.current_model[i] for i in model.BRANCH))


def add_bfa_branch_constraints(problem: PyomoPowerModel, pyo: Any) -> None:
    """Add linearized branch losses and voltage drop for BFA."""
    model, network = problem.model, problem.network
    resistance = network.branch[:, BR_R - 1]
    reactance = network.branch[:, BR_X - 1]
    model.branch_active_loss = pyo.Constraint(
        model.BRANCH,
        rule=lambda m, i: m.pf[i] + m.pt[i] == 0.0,
    )
    model.branch_reactive_loss = pyo.Constraint(
        model.BRANCH,
        rule=lambda m, i: (
            m.qf[i] + m.qt[i]
            == -float(network.b_fr[i] / network.tap[i] ** 2) * m.w[int(network.f_bus[i])]
            - float(network.b_to[i]) * m.w[int(network.t_bus[i])]
        ),
    )
    model.voltage_drop = pyo.Constraint(
        model.BRANCH,
        rule=lambda m, i: (
            m.w[int(network.f_bus[i])] / float(network.tap[i] ** 2) - m.w[int(network.t_bus[i])]
            == 2.0 * (float(resistance[i]) * m.pf[i] + float(reactance[i]) * m.qf[i])
        ),
    )
    _register_branch_equations(problem)


def add_socbf_branch_constraints(problem: PyomoPowerModel, pyo: Any) -> None:
    """Add SOCBF branch losses and voltage-drop equations."""
    model, network = problem.model, problem.network
    resistance = network.branch[:, BR_R - 1]
    reactance = network.branch[:, BR_X - 1]

    def series_current_expression(m: Any, i: int) -> Any:
        f_bus = int(network.f_bus[i])
        tap_squared = float(network.tap[i] ** 2)
        b_fr = float(network.b_fr[i])
        return m.ccm[i] + b_fr**2 * m.w[f_bus] / tap_squared + 2.0 * b_fr * m.qf[i]

    model.branch_active_loss = pyo.Constraint(
        model.BRANCH,
        rule=lambda m, i: m.pf[i] + m.pt[i] == float(resistance[i]) * series_current_expression(m, i),
    )
    model.branch_reactive_loss = pyo.Constraint(
        model.BRANCH,
        rule=lambda m, i: (
            m.qf[i] + m.qt[i]
            == float(reactance[i]) * series_current_expression(m, i)
            - float(network.b_fr[i] / network.tap[i] ** 2) * m.w[int(network.f_bus[i])]
            - float(network.b_to[i]) * m.w[int(network.t_bus[i])]
        ),
    )
    model.voltage_drop = pyo.Constraint(
        model.BRANCH,
        rule=lambda m, i: (
            (1.0 - 2.0 * float(reactance[i] * network.b_fr[i]))
            * m.w[int(network.f_bus[i])]
            / float(network.tap[i] ** 2)
            - m.w[int(network.t_bus[i])]
            == 2.0 * (float(resistance[i]) * m.pf[i] + float(reactance[i]) * m.qf[i])
            - float(resistance[i] ** 2 + reactance[i] ** 2) * series_current_expression(m, i)
        ),
    )
    _register_branch_equations(problem)


def _register_branch_equations(problem: PyomoPowerModel) -> None:
    model = problem.model
    problem.register_constraints("branch_active_loss", tuple(model.branch_active_loss[i] for i in model.BRANCH))
    problem.register_constraints(
        "branch_reactive_loss",
        tuple(model.branch_reactive_loss[i] for i in model.BRANCH),
    )
    problem.register_constraints("voltage_drop", tuple(model.voltage_drop[i] for i in model.BRANCH))


def add_branch_flow_angle_constraints(problem: PyomoPowerModel, pyo: Any) -> None:
    """Add PowerModels' linear branch-flow voltage-angle bounds."""
    model, network = problem.model, problem.network
    real_w, real_p, real_q, imag_w, imag_p, imag_q = branch_angle_coefficients(network)
    model.branch_voltage_product_real = pyo.Expression(
        model.BRANCH,
        rule=lambda m, i: (
            float(real_w[i]) * m.w[int(network.f_bus[i])] + float(real_p[i]) * m.pf[i] + float(real_q[i]) * m.qf[i]
        ),
    )
    model.branch_voltage_product_imaginary = pyo.Expression(
        model.BRANCH,
        rule=lambda m, i: (
            float(imag_w[i]) * m.w[int(network.f_bus[i])] + float(imag_p[i]) * m.pf[i] + float(imag_q[i]) * m.qf[i]
        ),
    )
    problem.register_expression("branch_voltage_product_real", model.branch_voltage_product_real)
    problem.register_expression("branch_voltage_product_imaginary", model.branch_voltage_product_imaginary)
    model.angle_upper = pyo.Constraint(
        model.BRANCH,
        rule=lambda m, i: (
            m.branch_voltage_product_imaginary[i]
            <= float(np.tan(network.branch_angle_max[i])) * m.branch_voltage_product_real[i]
        ),
    )
    model.angle_lower = pyo.Constraint(
        model.BRANCH,
        rule=lambda m, i: (
            m.branch_voltage_product_imaginary[i]
            >= float(np.tan(network.branch_angle_min[i])) * m.branch_voltage_product_real[i]
        ),
    )
    problem.register_constraints("angle_upper", tuple(model.angle_upper[i] for i in model.BRANCH))
    problem.register_constraints("angle_lower", tuple(model.angle_lower[i] for i in model.BRANCH))


__all__ = [
    "add_bfa_branch_constraints",
    "add_branch_current_variables",
    "add_branch_flow_angle_constraints",
    "add_socbf_branch_constraints",
    "add_socbf_current_constraints",
    "branch_angle_coefficients",
    "branch_voltage_product_values",
]
