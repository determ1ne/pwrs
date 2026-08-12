# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from .idx_cost import COST, MODEL, NCOST, PW_LINEAR
from .totcost import totcost


def poly2pwl(polycost, Pmin, Pmax, npts):
    """Convert polynomial costs to piecewise-linear costs at evenly spaced points."""
    polynomial = np.atleast_2d(np.asarray(polycost, dtype=float))
    pmin = np.asarray(Pmin, dtype=float).reshape(-1)
    pmax = np.asarray(Pmax, dtype=float).reshape(-1)
    point_count = int(npts)
    if pmin.size != polynomial.shape[0] or pmax.size != polynomial.shape[0]:
        raise ValueError("poly2pwl: Pmin and Pmax must contain one value per cost row")
    if point_count < 2:
        raise ValueError("poly2pwl: npts must be at least 2")
    if point_count < 3 and np.any((pmin > 0) | (pmax < 0)):
        raise ValueError("poly2pwl: npts must be at least 3 when the range excludes zero")

    width = max(polynomial.shape[1], COST + 2 * point_count)
    piecewise = np.zeros((polynomial.shape[0], width), dtype=float)
    piecewise[:, : polynomial.shape[1]] = polynomial
    piecewise[:, MODEL] = PW_LINEAR
    piecewise[:, COST :] = 0
    piecewise[:, NCOST] = point_count

    for row in range(polynomial.shape[0]):
        if pmin[row] > 0:
            x = np.r_[0.0, np.linspace(pmin[row], pmax[row], point_count - 1)]
        elif pmax[row] < 0:
            x = np.r_[np.linspace(pmin[row], pmax[row], point_count - 1), 0.0]
        else:
            x = np.linspace(pmin[row], pmax[row], point_count)
        repeated_cost = np.repeat(polynomial[row : row + 1], point_count, axis=0)
        y = np.asarray(totcost(repeated_cost, x), dtype=float).reshape(-1)
        piecewise[row, COST : COST + 2 * point_count : 2] = x
        piecewise[row, COST + 1 : COST + 2 * point_count : 2] = y
    return piecewise
