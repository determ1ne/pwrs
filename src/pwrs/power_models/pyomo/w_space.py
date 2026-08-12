# Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause
"""Shared W-space voltage components for Pyomo power models."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...core.idx_bus import VMAX, VMIN
from ..voltage import (
    branch_pair_indices,
    map_w_solution,
    reconstruct_voltage_angles,
    voltage_product_bounds,
)
from .context import PyomoPowerModel


def add_squared_voltage_variables(problem: PyomoPowerModel, pyo: Any) -> None:
    """Add bounded squared-voltage variables with PowerModels starts."""
    model, network = problem.model, problem.network
    model.w = pyo.Var(
        model.BUS,
        bounds=lambda _, i: (
            float(network.bus[i, VMIN] ** 2),
            float(network.bus[i, VMAX] ** 2),
        ),
    )
    for i in model.BUS:
        model.w[i].set_value(1.001, skip_validation=True)
    problem.register_variables("w", tuple(model.w[i] for i in model.BUS))
    model.voltage_magnitude_squared = pyo.Expression(model.BUS, rule=lambda m, i: m.w[i])
    problem.register_expression("voltage_magnitude_squared", model.voltage_magnitude_squared)


def add_w_voltage_variables(problem: PyomoPowerModel, pyo: Any) -> None:
    """Add bounded ``w``, ``wr``, and ``wi`` variables with PowerModels starts."""
    model, network = problem.model, problem.network
    add_squared_voltage_variables(problem, pyo)
    pair_indices = tuple(range(len(network.angle_pairs)))
    if not hasattr(model, "ANGLE_PAIR"):
        model.ANGLE_PAIR = pyo.Set(initialize=pair_indices, ordered=True)
    wr_min, wr_max, wi_min, wi_max = voltage_product_bounds(network)
    model.wr = pyo.Var(
        model.ANGLE_PAIR,
        bounds=lambda _, i: (float(wr_min[i]), float(wr_max[i])),
    )
    model.wi = pyo.Var(
        model.ANGLE_PAIR,
        bounds=lambda _, i: (float(wi_min[i]), float(wi_max[i])),
    )
    for i in model.ANGLE_PAIR:
        model.wr[i].set_value(1.0, skip_validation=True)
        model.wi[i].set_value(0.0, skip_validation=True)
    problem.register_variables("wr", tuple(model.wr[i] for i in model.ANGLE_PAIR))
    problem.register_variables("wi", tuple(model.wi[i] for i in model.ANGLE_PAIR))


def add_w_branch_voltage_expressions(problem: PyomoPowerModel, pyo: Any) -> None:
    """Map shared bus-pair voltage products onto each physical branch."""
    model, network = problem.model, problem.network
    branch_pairs = branch_pair_indices(network)
    model.branch_voltage_product_real = pyo.Expression(
        model.BRANCH,
        rule=lambda m, i: m.wr[int(branch_pairs[i])],
    )
    model.branch_voltage_product_imaginary = pyo.Expression(
        model.BRANCH,
        rule=lambda m, i: m.wi[int(branch_pairs[i])],
    )
    problem.register_expression("branch_voltage_product_real", model.branch_voltage_product_real)
    problem.register_expression("branch_voltage_product_imaginary", model.branch_voltage_product_imaginary)


def add_w_angle_constraints(problem: PyomoPowerModel, pyo: Any) -> None:
    """Add tangent angle bounds and lifted nonlinear cuts in W-space."""
    model, network = problem.model, problem.network
    model.angle_upper = pyo.Constraint(
        model.ANGLE_PAIR,
        rule=lambda m, i: m.wi[i] <= float(np.tan(network.angle_max[i])) * m.wr[i],
    )
    model.angle_lower = pyo.Constraint(
        model.ANGLE_PAIR,
        rule=lambda m, i: m.wi[i] >= float(np.tan(network.angle_min[i])) * m.wr[i],
    )
    problem.register_constraints("angle_upper", tuple(model.angle_upper[i] for i in model.ANGLE_PAIR))
    problem.register_constraints("angle_lower", tuple(model.angle_lower[i] for i in model.ANGLE_PAIR))

    f_bus = network.angle_pairs[:, 0]
    t_bus = network.angle_pairs[:, 1]
    vf_min = network.bus[f_bus, VMIN]
    vf_max = network.bus[f_bus, VMAX]
    vt_min = network.bus[t_bus, VMIN]
    vt_max = network.bus[t_bus, VMAX]
    phi = (network.angle_max + network.angle_min) / 2
    cosine = np.cos((network.angle_max - network.angle_min) / 2)
    sf = vf_min + vf_max
    st = vt_min + vt_max
    spread = vf_min * vt_min - vf_max * vt_max
    upper_rhs = vf_max * vt_max * cosine * spread
    lower_rhs = -vf_min * vt_min * cosine * spread

    def lifted_upper_expression(m: Any, i: int) -> Any:
        return (
            sf[i] * st[i] * (np.cos(phi[i]) * m.wr[i] + np.sin(phi[i]) * m.wi[i])
            - vt_max[i] * cosine[i] * st[i] * m.w[int(f_bus[i])]
            - vf_max[i] * cosine[i] * sf[i] * m.w[int(t_bus[i])]
        )

    def lifted_lower_expression(m: Any, i: int) -> Any:
        return (
            sf[i] * st[i] * (np.cos(phi[i]) * m.wr[i] + np.sin(phi[i]) * m.wi[i])
            - vt_min[i] * cosine[i] * st[i] * m.w[int(f_bus[i])]
            - vf_min[i] * cosine[i] * sf[i] * m.w[int(t_bus[i])]
        )

    model.lifted_cut_upper_voltage = pyo.Constraint(
        model.ANGLE_PAIR,
        rule=lambda m, i: lifted_upper_expression(m, i) >= float(upper_rhs[i]),
    )
    model.lifted_cut_lower_voltage = pyo.Constraint(
        model.ANGLE_PAIR,
        rule=lambda m, i: lifted_lower_expression(m, i) >= float(lower_rhs[i]),
    )
    problem.register_constraints(
        "lifted_cut_upper_voltage",
        tuple(model.lifted_cut_upper_voltage[i] for i in model.ANGLE_PAIR),
    )
    problem.register_constraints(
        "lifted_cut_lower_voltage",
        tuple(model.lifted_cut_lower_voltage[i] for i in model.ANGLE_PAIR),
    )


__all__ = [
    "add_w_branch_voltage_expressions",
    "add_w_angle_constraints",
    "add_squared_voltage_variables",
    "add_w_voltage_variables",
    "map_w_solution",
    "reconstruct_voltage_angles",
    "voltage_product_bounds",
]
