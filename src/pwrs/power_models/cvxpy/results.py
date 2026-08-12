# Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause
"""Result mapping shared by CVXPY W-space and branch-flow formulations."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..network import PowerNetwork
from ..results import PowerModelSolution, build_matpower_result
from ..voltage import (
    branch_voltage_product_values,
    fit_voltage_angles,
    map_w_solution,
    reconstruct_voltage_angles,
)


def build_w_result(
    network: PowerNetwork,
    solution: PowerModelSolution,
    objective: float,
    success: int,
    info: dict[str, Any],
    formulation: str,
):
    """Map a W-space conic solution to MATPOWER voltage and branch fields."""
    multipliers = dict(solution.constraint_multipliers)
    wr = solution.variables["wr"]
    if "angle_upper" in multipliers:
        multipliers["angle_upper"] = multipliers["angle_upper"] * wr / np.cos(network.angle_max) ** 2
    if "angle_lower" in multipliers:
        multipliers["angle_lower"] = -multipliers["angle_lower"] * wr / np.cos(network.angle_min) ** 2
    native = PowerModelSolution(
        vector=solution.vector,
        variables=solution.variables,
        constraint_multipliers=multipliers,
        lower_bound_multipliers=solution.lower_bound_multipliers,
        upper_bound_multipliers=solution.upper_bound_multipliers,
    )
    va, angle_residual = reconstruct_voltage_angles(network, wr, solution.variables["wi"])
    mapped = map_w_solution(native, va)
    result, raw = build_matpower_result(
        network,
        mapped,
        objective,
        success,
        info,
        formulation=formulation,
        implementation="CVXPY",
        branch_limit="conic_apparent_power",
    )
    raw["output"].update(
        {
            "voltage_angle_source": "least_squares_from_wr_wi",
            "voltage_angle_max_pair_residual": angle_residual,
        }
    )
    return result, raw


def build_branch_flow_result(
    network: PowerNetwork,
    solution: PowerModelSolution,
    objective: float,
    success: int,
    info: dict[str, Any],
    formulation: str,
):
    """Map a conic branch-flow solution to MATPOWER result fields."""
    real, imaginary = branch_voltage_product_values(
        network,
        solution.variables["w"],
        solution.variables["pf"],
        solution.variables["qf"],
    )
    branch_angles = np.arctan2(imaginary, real)
    edges = np.column_stack((network.f_bus, network.t_bus))
    va, angle_residual = fit_voltage_angles(network, edges, branch_angles)
    mapped = map_w_solution(solution, va)
    result, raw = build_matpower_result(
        network,
        mapped,
        objective,
        success,
        info,
        formulation=formulation,
        implementation="CVXPY",
        branch_limit="conic_apparent_power",
        angle_limit="branch",
    )
    raw["output"].update(
        {
            "voltage_angle_source": "least_squares_from_branch_flow",
            "voltage_angle_max_branch_residual": angle_residual,
        }
    )
    return result, raw


__all__ = ["build_branch_flow_result", "build_w_result"]
