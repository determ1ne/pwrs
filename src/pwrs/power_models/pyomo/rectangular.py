# Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause
"""Rectangular-voltage components shared by ACR and IVR formulations."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...core.idx_bus import VMAX, VMIN
from .context import PyomoPowerModel


def add_rectangular_voltage_variables(problem: PyomoPowerModel, pyo: Any) -> None:
    """Add bounded real and imaginary bus-voltage variables."""
    model, network = problem.model, problem.network
    model.vr = pyo.Var(
        model.BUS,
        bounds=lambda _, i: (-float(network.bus[i, VMAX - 1]), float(network.bus[i, VMAX - 1])),
        initialize=1.0,
    )
    model.vi = pyo.Var(
        model.BUS,
        bounds=lambda _, i: (-float(network.bus[i, VMAX - 1]), float(network.bus[i, VMAX - 1])),
        initialize=0.0,
    )
    problem.register_variables("vr", tuple(model.vr[i] for i in model.BUS))
    problem.register_variables("vi", tuple(model.vi[i] for i in model.BUS))
    model.voltage_magnitude_squared = pyo.Expression(
        model.BUS,
        rule=lambda m, i: m.vr[i] ** 2 + m.vi[i] ** 2,
    )
    problem.register_expression("voltage_magnitude_squared", model.voltage_magnitude_squared)


def add_rectangular_voltage_magnitude_constraints(problem: PyomoPowerModel, pyo: Any) -> None:
    """Constrain the magnitude of each rectangular bus voltage."""
    model, network = problem.model, problem.network
    model.voltage_lower = pyo.Constraint(
        model.BUS,
        rule=lambda m, i: m.voltage_magnitude_squared[i] >= float(network.bus[i, VMIN - 1] ** 2),
    )
    model.voltage_upper = pyo.Constraint(
        model.BUS,
        rule=lambda m, i: m.voltage_magnitude_squared[i] <= float(network.bus[i, VMAX - 1] ** 2),
    )
    problem.register_constraints("voltage_lower", tuple(model.voltage_lower[i] for i in model.BUS))
    problem.register_constraints("voltage_upper", tuple(model.voltage_upper[i] for i in model.BUS))


def add_rectangular_reference_constraints(problem: PyomoPowerModel, pyo: Any) -> None:
    """Fix the imaginary voltage at reference buses."""
    model, network = problem.model, problem.network
    reference_buses = tuple(int(i) for i in network.refs)
    model.REF_BUS = pyo.Set(initialize=reference_buses, ordered=True)
    model.reference_angle = pyo.Constraint(model.REF_BUS, rule=lambda m, i: m.vi[i] == 0.0)
    problem.register_constraints("reference_angle", tuple(model.reference_angle[i] for i in reference_buses))


def add_rectangular_branch_voltage_expressions(problem: PyomoPowerModel, pyo: Any) -> None:
    """Add real and imaginary branch voltage-product expressions."""
    model, network = problem.model, problem.network

    def product_real(m: Any, i: int) -> Any:
        f_bus, t_bus = int(network.f_bus[i]), int(network.t_bus[i])
        return m.vr[f_bus] * m.vr[t_bus] + m.vi[f_bus] * m.vi[t_bus]

    def product_imaginary(m: Any, i: int) -> Any:
        f_bus, t_bus = int(network.f_bus[i]), int(network.t_bus[i])
        return m.vi[f_bus] * m.vr[t_bus] - m.vr[f_bus] * m.vi[t_bus]

    model.branch_voltage_product_real = pyo.Expression(model.BRANCH, rule=product_real)
    model.branch_voltage_product_imaginary = pyo.Expression(model.BRANCH, rule=product_imaginary)
    problem.register_expression("branch_voltage_product_real", model.branch_voltage_product_real)
    problem.register_expression("branch_voltage_product_imaginary", model.branch_voltage_product_imaginary)


def add_rectangular_angle_constraints(
    problem: PyomoPowerModel,
    pyo: Any,
    *,
    per_branch: bool = False,
) -> None:
    """Add tangent angle bounds over bus pairs or individual branches."""
    model, network = problem.model, problem.network
    if per_branch:
        index = model.BRANCH
        real = model.branch_voltage_product_real
        imaginary = model.branch_voltage_product_imaginary
        angle_min = network.branch_angle_min
        angle_max = network.branch_angle_max
    else:
        pair_indices = tuple(range(len(network.angle_pairs)))
        model.ANGLE_PAIR = pyo.Set(initialize=pair_indices, ordered=True)

        def pair_real(m: Any, i: int) -> Any:
            f_bus, t_bus = (int(value) for value in network.angle_pairs[i])
            return m.vr[f_bus] * m.vr[t_bus] + m.vi[f_bus] * m.vi[t_bus]

        def pair_imaginary(m: Any, i: int) -> Any:
            f_bus, t_bus = (int(value) for value in network.angle_pairs[i])
            return m.vi[f_bus] * m.vr[t_bus] - m.vr[f_bus] * m.vi[t_bus]

        model.angle_product_real = pyo.Expression(model.ANGLE_PAIR, rule=pair_real)
        model.angle_product_imaginary = pyo.Expression(model.ANGLE_PAIR, rule=pair_imaginary)
        problem.register_expression("angle_product_real", model.angle_product_real)
        problem.register_expression("angle_product_imaginary", model.angle_product_imaginary)
        index = model.ANGLE_PAIR
        real = model.angle_product_real
        imaginary = model.angle_product_imaginary
        angle_min = network.angle_min
        angle_max = network.angle_max

    model.angle_upper = pyo.Constraint(
        index,
        rule=lambda m, i: imaginary[i] <= float(np.tan(angle_max[i])) * real[i],
    )
    model.angle_lower = pyo.Constraint(
        index,
        rule=lambda m, i: imaginary[i] >= float(np.tan(angle_min[i])) * real[i],
    )
    problem.register_constraints("angle_upper", tuple(model.angle_upper[i] for i in index))
    problem.register_constraints("angle_lower", tuple(model.angle_lower[i] for i in index))


__all__ = [
    "add_rectangular_angle_constraints",
    "add_rectangular_branch_voltage_expressions",
    "add_rectangular_reference_constraints",
    "add_rectangular_voltage_magnitude_constraints",
    "add_rectangular_voltage_variables",
]
