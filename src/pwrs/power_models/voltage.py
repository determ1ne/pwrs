# Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause
"""Formulation-neutral voltage products and derived-angle utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsqr

from ..core.idx_brch import BR_R, BR_X
from ..core.idx_bus import VMAX, VMIN
from .network import PowerNetwork
from .results import PowerModelSolution


def branch_pair_indices(network: PowerNetwork) -> np.ndarray:
    """Map each physical branch to its aggregated directed bus-pair index."""
    pair_lookup = {(int(f), int(t)): i for i, (f, t) in enumerate(network.angle_pairs)}
    return np.asarray(
        [pair_lookup[(int(f), int(t))] for f, t in zip(network.f_bus, network.t_bus)],
        dtype=int,
    )


def fit_voltage_angles(
    network: PowerNetwork,
    edges: np.ndarray,
    angle_differences: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Fit bus angles to directed edge differences with references fixed."""
    nb = len(network.bus)
    refs = set(int(i) for i in network.refs)
    free_buses = [i for i in range(nb) if i not in refs]
    if not free_buses:
        return np.zeros(nb), 0.0
    columns = {bus: i for i, bus in enumerate(free_buses)}
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for row, (f_bus, t_bus) in enumerate(np.asarray(edges, dtype=int)):
        if int(f_bus) in columns:
            rows.append(row)
            cols.append(columns[int(f_bus)])
            data.append(1.0)
        if int(t_bus) in columns:
            rows.append(row)
            cols.append(columns[int(t_bus)])
            data.append(-1.0)
    incidence = sparse.coo_matrix(
        (data, (rows, cols)),
        shape=(len(edges), len(free_buses)),
    ).tocsr()
    differences = np.asarray(angle_differences, dtype=float)
    # SciPy currently exposes ``lsqr`` as a function at runtime, while its
    # package layout makes Pyright resolve the symbol as the backing module.
    fitted = lsqr(incidence, differences, atol=1e-12, btol=1e-12)[0]  # pyright: ignore[reportCallIssue]
    va = np.zeros(nb)
    va[free_buses] = fitted
    residual = incidence @ fitted - differences
    return va, float(np.max(np.abs(residual), initial=0.0))


def voltage_product_bounds(network: PowerNetwork) -> tuple[np.ndarray, ...]:
    """Port PowerModels' bus-pair rectangular voltage-product bounds."""
    count = len(network.angle_pairs)
    wr_min = np.empty(count)
    wr_max = np.empty(count)
    wi_min = np.empty(count)
    wi_max = np.empty(count)
    for i, (f_bus, t_bus) in enumerate(network.angle_pairs):
        f_bus, t_bus = int(f_bus), int(t_bus)
        vf_min = float(network.bus[f_bus, VMIN])
        vf_max = float(network.bus[f_bus, VMAX])
        vt_min = float(network.bus[t_bus, VMIN])
        vt_max = float(network.bus[t_bus, VMAX])
        angle_min = float(network.angle_min[i])
        angle_max = float(network.angle_max[i])
        if angle_min >= 0:
            wr_max[i] = vf_max * vt_max * np.cos(angle_min)
            wr_min[i] = vf_min * vt_min * np.cos(angle_max)
            wi_max[i] = vf_max * vt_max * np.sin(angle_max)
            wi_min[i] = vf_min * vt_min * np.sin(angle_min)
        elif angle_max <= 0:
            wr_max[i] = vf_max * vt_max * np.cos(angle_max)
            wr_min[i] = vf_min * vt_min * np.cos(angle_min)
            wi_max[i] = vf_min * vt_min * np.sin(angle_max)
            wi_min[i] = vf_max * vt_max * np.sin(angle_min)
        else:
            wr_max[i] = vf_max * vt_max
            wr_min[i] = vf_min * vt_min * min(np.cos(angle_min), np.cos(angle_max))
            wi_max[i] = vf_max * vt_max * np.sin(angle_max)
            wi_min[i] = vf_max * vt_max * np.sin(angle_min)
    return wr_min, wr_max, wi_min, wi_max


def map_w_solution(solution: PowerModelSolution, va: np.ndarray) -> PowerModelSolution:
    """Add polar voltage values and multipliers to a native W-space solution."""
    w = solution.variables["w"]
    vm = np.sqrt(w)
    variables = dict(solution.variables)
    variables.update({"va": np.asarray(va, dtype=float), "vm": vm})
    lower_bounds = dict(solution.lower_bound_multipliers)
    upper_bounds = dict(solution.upper_bound_multipliers)
    lower_bounds["vm"] = 2 * vm * solution.lower_bound_multipliers["w"]
    upper_bounds["vm"] = 2 * vm * solution.upper_bound_multipliers["w"]
    return PowerModelSolution(
        vector=solution.vector,
        variables=variables,
        constraint_multipliers=solution.constraint_multipliers,
        lower_bound_multipliers=lower_bounds,
        upper_bound_multipliers=upper_bounds,
    )


def reconstruct_voltage_angles(
    network: PowerNetwork,
    wr: np.ndarray,
    wi: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Fit bus angles to relaxed pair angles, fixing references at zero."""
    return fit_voltage_angles(network, network.angle_pairs, np.arctan2(wi, wr))


def branch_angle_coefficients(network: Any) -> tuple[np.ndarray, ...]:
    """Return the affine BF voltage-product coefficients for each branch."""
    resistance = network.branch[:, BR_R]
    reactance = network.branch[:, BR_X]
    tr = network.tap * np.cos(network.shift)
    ti = network.tap * np.sin(network.shift)
    tzr = resistance * tr + reactance * ti
    tzi = resistance * ti - reactance * tr
    tap_squared = network.tap**2
    real_w = (tr + tzi * network.b_fr) / tap_squared
    imag_w = (ti - tzr * network.b_fr) / tap_squared
    return real_w, -tzr, tzi, imag_w, -tzi, -tzr


def branch_voltage_product_values(
    network: Any,
    w: np.ndarray,
    pf: np.ndarray,
    qf: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate BF voltage-product components used by angle constraints."""
    real_w, real_p, real_q, imag_w, imag_p, imag_q = branch_angle_coefficients(network)
    w_from = np.asarray(w)[network.f_bus]
    real = real_w * w_from + real_p * pf + real_q * qf
    imaginary = imag_w * w_from + imag_p * pf + imag_q * qf
    return real, imaginary


__all__ = [
    "branch_angle_coefficients",
    "branch_pair_indices",
    "branch_voltage_product_values",
    "fit_voltage_angles",
    "map_w_solution",
    "reconstruct_voltage_angles",
    "voltage_product_bounds",
]
