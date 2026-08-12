# Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause
"""Generator and DC-line cost parsing shared by optimization backends."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.idx_cost import COST, MODEL, NCOST, POLYNOMIAL, PW_LINEAR
from ..core.idx_gen import PMAX, PMIN
from .network import PowerNetwork


@dataclass(frozen=True)
class PolynomialGeneratorCost:
    """Descending polynomial coefficients evaluated against generator MW."""

    coefficients: np.ndarray


@dataclass(frozen=True)
class PiecewiseLinearGeneratorCost:
    """Convex PWL points with power coordinates in per unit."""

    points: np.ndarray


type GeneratorCost = PolynomialGeneratorCost | PiecewiseLinearGeneratorCost


def _clean_pwl_points(
    points: np.ndarray,
    pmin: float,
    pmax: float,
    *,
    component: str,
) -> np.ndarray:
    """Port PowerModels' active-segment cleanup for one PWL cost curve."""
    if pmin > pmax:
        raise ValueError(f"{component} PWL cost requires pmin <= pmax")
    if len(points) < 2:
        raise ValueError(f"{component} PWL cost requires at least two points")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{component} PWL cost points must be finite")
    differences = np.diff(points[:, 0])
    if np.any(differences <= 0):
        raise ValueError(f"{component} PWL cost power points must be strictly increasing")

    first = len(points) - 2
    for i in range(len(points) - 1):
        if pmin <= points[i + 1, 0]:
            first = i
            break
    last = 1
    for i in range(len(points) - 1, 0, -1):
        if pmax >= points[i - 1, 0]:
            last = i
            break
    active = points[first : last + 1].copy()

    tolerance = 1e-2
    if active[0, 0] > pmin:
        x0 = pmin - tolerance
        slope = (active[1, 1] - active[0, 1]) / (active[1, 0] - active[0, 0])
        active[0] = (x0, active[1, 1] - slope * (active[1, 0] - x0))
    if active[-1, 0] < pmax:
        x3 = pmax + tolerance
        slope = (active[-1, 1] - active[-2, 1]) / (active[-1, 0] - active[-2, 0])
        active[-1] = (x3, slope * (x3 - active[-2, 0]) + active[-2, 1])

    slopes = np.diff(active[:, 1]) / np.diff(active[:, 0])
    if np.any(np.diff(slopes) < -1e-9):
        raise ValueError(f"{component} PWL cost must be convex over its active range")
    return active


def _parse_costs(
    rows: np.ndarray,
    pmin: np.ndarray,
    pmax: np.ndarray,
    base_mva: float,
    *,
    component: str,
) -> tuple[GeneratorCost, ...]:
    costs: list[GeneratorCost] = []
    for i, row in enumerate(rows):
        label = f"{component} {i}"
        model = int(row[MODEL - 1])
        count = int(row[NCOST - 1])
        if count < 1:
            raise ValueError(f"{label} cost NCOST must be positive")
        width = count if model == POLYNOMIAL else 2 * count
        if COST - 1 + width > len(row):
            raise ValueError(f"{label} cost row is shorter than NCOST requires")
        values = np.asarray(row[COST - 1 : COST - 1 + width], dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{label} cost values must be finite")
        if model == POLYNOMIAL:
            costs.append(PolynomialGeneratorCost(values))
        elif model == PW_LINEAR:
            points = values.reshape((-1, 2)).copy()
            points[:, 0] /= base_mva
            costs.append(
                PiecewiseLinearGeneratorCost(
                    _clean_pwl_points(
                        points,
                        float(pmin[i]),
                        float(pmax[i]),
                        component=label,
                    )
                )
            )
        else:
            raise ValueError(f"{label} has unsupported cost model {model}")
    return tuple(costs)


def generator_costs(network: PowerNetwork) -> tuple[GeneratorCost, ...]:
    """Parse active-generator cost rows into validated backend-neutral data."""
    return _parse_costs(
        network.gencost,
        network.gen[:, PMIN - 1] / network.base_mva,
        network.gen[:, PMAX - 1] / network.base_mva,
        network.base_mva,
        component="generator",
    )


def dcline_costs(network: PowerNetwork) -> tuple[GeneratorCost, ...]:
    """Parse active DC-line from-side costs into backend-neutral data."""
    return _parse_costs(
        network.dclinecost,
        network.dc_pmin_from,
        network.dc_pmax_from,
        network.base_mva,
        component="DC line",
    )


__all__ = [
    "GeneratorCost",
    "PiecewiseLinearGeneratorCost",
    "PolynomialGeneratorCost",
    "dcline_costs",
    "generator_costs",
]
