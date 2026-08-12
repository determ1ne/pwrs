# Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause
"""Shared active-power-only Pyomo network components."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...core.idx_brch import BR_R
from ...core.idx_bus import GS, PD
from .context import PyomoPowerModel


def add_lossless_branch_power_variables(problem: PyomoPowerModel, pyo: Any) -> None:
    """Add one bounded real-power variable per lossless branch."""
    model, network = problem.model, problem.network

    def flow_bounds(_: Any, i: int) -> tuple[float | None, float | None]:
        rating = float(network.rate[i])
        return (-rating, rating) if np.isfinite(rating) else (None, None)

    model.p = pyo.Var(model.BRANCH, initialize=0.0, bounds=flow_bounds)
    problem.register_variables("p", tuple(model.p[i] for i in model.BRANCH))


def add_active_power_balance_constraints(problem: PyomoPowerModel, pyo: Any) -> None:
    """Add fixed-unit-voltage active nodal balances for a lossless network."""
    model, network = problem.model, problem.network

    def active_balance_rule(m: Any, i: int) -> Any:
        return (
            float((network.bus[i, PD - 1] + network.bus[i, GS - 1]) / network.base_mva)
            + pyo.quicksum(m.p[j] for j in network.from_branches_at_bus[i])
            - pyo.quicksum(m.p[j] for j in network.to_branches_at_bus[i])
            + (pyo.quicksum(m.pdcf[j] for j in network.from_dclines_at_bus[i]) if len(network.dcline) else 0.0)
            + (pyo.quicksum(m.pdct[j] for j in network.to_dclines_at_bus[i]) if len(network.dcline) else 0.0)
            - pyo.quicksum(m.pg[j] for j in network.generators_at_bus[i])
            == 0.0
        )

    model.active_balance = pyo.Constraint(model.BUS, rule=active_balance_rule)
    problem.register_constraints("active_balance", tuple(model.active_balance[i] for i in model.BUS))


def add_directed_branch_power_variables(problem: PyomoPowerModel, pyo: Any) -> None:
    """Add bounded from- and to-end active branch power variables."""
    model, network = problem.model, problem.network

    def flow_bounds(_: Any, i: int) -> tuple[float | None, float | None]:
        rating = float(network.rate[i])
        return (-rating, rating) if np.isfinite(rating) else (None, None)

    model.pf = pyo.Var(model.BRANCH, initialize=0.0, bounds=flow_bounds)
    model.pt = pyo.Var(model.BRANCH, initialize=0.0, bounds=flow_bounds)
    problem.register_variables("pf", tuple(model.pf[i] for i in model.BRANCH))
    problem.register_variables("pt", tuple(model.pt[i] for i in model.BRANCH))


def add_directed_active_power_balance_constraints(problem: PyomoPowerModel, pyo: Any) -> None:
    """Add active balances using independent from- and to-end branch flows."""
    model, network = problem.model, problem.network

    def active_balance_rule(m: Any, i: int) -> Any:
        return (
            float((network.bus[i, PD - 1] + network.bus[i, GS - 1]) / network.base_mva)
            + pyo.quicksum(m.pf[j] for j in network.from_branches_at_bus[i])
            + pyo.quicksum(m.pt[j] for j in network.to_branches_at_bus[i])
            + (pyo.quicksum(m.pdcf[j] for j in network.from_dclines_at_bus[i]) if len(network.dcline) else 0.0)
            + (pyo.quicksum(m.pdct[j] for j in network.to_dclines_at_bus[i]) if len(network.dcline) else 0.0)
            - pyo.quicksum(m.pg[j] for j in network.generators_at_bus[i])
            == 0.0
        )

    model.active_balance = pyo.Constraint(model.BUS, rule=active_balance_rule)
    problem.register_constraints("active_balance", tuple(model.active_balance[i] for i in model.BUS))


def add_linear_branch_power_equations(
    problem: PyomoPowerModel,
    pyo: Any,
    coefficient: np.ndarray,
    angle_shift: np.ndarray,
) -> None:
    """Link lossless branch power to a scaled, shifted angle difference."""
    model, network = problem.model, problem.network
    coefficient = np.asarray(coefficient, dtype=float)
    angle_shift = np.asarray(angle_shift, dtype=float)
    model.branch_flow = pyo.Constraint(
        model.BRANCH,
        rule=lambda m, i: (
            m.p[i]
            == float(coefficient[i])
            * (m.va[int(network.f_bus[i])] - m.va[int(network.t_bus[i])] - float(angle_shift[i]))
        ),
    )
    problem.register_constraints("branch_flow", tuple(model.branch_flow[i] for i in model.BRANCH))


def add_directed_dc_branch_constraints(
    problem: PyomoPowerModel,
    pyo: Any,
    coefficient: np.ndarray,
) -> None:
    """Add DCP from-end flow equations and convex line-loss inequalities."""
    model, network = problem.model, problem.network
    coefficient = np.asarray(coefficient, dtype=float)
    resistance = np.asarray(network.branch[:, BR_R - 1], dtype=float)
    model.branch_flow = pyo.Constraint(
        model.BRANCH,
        rule=lambda m, i: (
            m.pf[i] == float(coefficient[i]) * (m.va[int(network.f_bus[i])] - m.va[int(network.t_bus[i])])
        ),
    )
    model.branch_loss = pyo.Constraint(
        model.BRANCH,
        rule=lambda m, i: m.pf[i] + m.pt[i] >= float(resistance[i]) * m.pf[i] ** 2,
    )
    problem.register_constraints("branch_flow", tuple(model.branch_flow[i] for i in model.BRANCH))
    problem.register_constraints("branch_loss", tuple(model.branch_loss[i] for i in model.BRANCH))


__all__ = [
    "add_active_power_balance_constraints",
    "add_directed_active_power_balance_constraints",
    "add_directed_branch_power_variables",
    "add_directed_dc_branch_constraints",
    "add_linear_branch_power_equations",
    "add_lossless_branch_power_variables",
]
