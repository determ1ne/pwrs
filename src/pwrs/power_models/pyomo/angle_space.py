# Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause
"""Shared polar-angle components for Pyomo power models."""

from __future__ import annotations

from typing import Any

from ..voltage import branch_pair_indices, fit_voltage_angles
from .context import PyomoPowerModel


def add_voltage_angle_variables(problem: PyomoPowerModel, pyo: Any) -> None:
    """Add bus voltage-angle variables with PowerModels' zero start."""
    model = problem.model
    model.va = pyo.Var(model.BUS, initialize=0.0)
    problem.register_variables("va", tuple(model.va[i] for i in model.BUS))


def add_reference_angle_constraints(problem: PyomoPowerModel, pyo: Any) -> None:
    """Fix the reference-bus angle in each connected component."""
    model, network = problem.model, problem.network
    reference_buses = tuple(int(i) for i in network.refs)
    model.REF_BUS = pyo.Set(initialize=reference_buses, ordered=True)
    model.reference_angle = pyo.Constraint(model.REF_BUS, rule=lambda m, i: m.va[i] == 0.0)
    problem.register_constraints("reference_angle", tuple(model.reference_angle[i] for i in reference_buses))


def add_linear_angle_difference_constraints(problem: PyomoPowerModel, pyo: Any) -> None:
    """Add aggregated bus-pair angle expressions and linear bounds."""
    model, network = problem.model, problem.network
    pair_indices = tuple(range(len(network.angle_pairs)))
    if not hasattr(model, "ANGLE_PAIR"):
        model.ANGLE_PAIR = pyo.Set(initialize=pair_indices, ordered=True)

    def angle_pair_rule(m: Any, i: int) -> Any:
        f_bus, t_bus = (int(value) for value in network.angle_pairs[i])
        return m.va[f_bus] - m.va[t_bus]

    model.angle_pair = pyo.Expression(model.ANGLE_PAIR, rule=angle_pair_rule)
    problem.register_expression("angle_pair", model.angle_pair)
    model.angle_upper = pyo.Constraint(
        model.ANGLE_PAIR,
        rule=lambda m, i: m.angle_pair[i] <= float(network.angle_max[i]),
    )
    model.angle_lower = pyo.Constraint(
        model.ANGLE_PAIR,
        rule=lambda m, i: m.angle_pair[i] >= float(network.angle_min[i]),
    )
    problem.register_constraints("angle_upper", tuple(model.angle_upper[i] for i in pair_indices))
    problem.register_constraints("angle_lower", tuple(model.angle_lower[i] for i in pair_indices))


__all__ = [
    "add_linear_angle_difference_constraints",
    "add_reference_angle_constraints",
    "add_voltage_angle_variables",
    "branch_pair_indices",
    "fit_voltage_angles",
]
