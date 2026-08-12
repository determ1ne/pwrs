# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from .idx_cost import COST, MODEL, NCOST, POLYNOMIAL, PW_LINEAR
from .polycost import polycost


def _totcost_pwl(x: np.ndarray, y: np.ndarray, pg: np.ndarray) -> np.ndarray:
    values = np.zeros(pg.shape, dtype=float)
    if x.size < 2:
        return values

    slopes = np.diff(y) / np.diff(x)
    for j, p in enumerate(pg):
        if p <= x[0]:
            k = 0
        elif p >= x[-1]:
            k = len(slopes) - 1
        else:
            k = np.searchsorted(x, p, side="right") - 1
        values[j] = y[k] + slopes[k] * (p - x[k])
    return values


def totcost(gencost, Pg):
    """Evaluate total generator production costs.

    Mirrors MATPOWER's ``totcost`` helper by evaluating either piecewise
    linear or polynomial generator cost models at the supplied output levels.

    Parameters
    ----------
    gencost : array_like
        Generator cost matrix.
    Pg : array_like
        Generator output values at which to evaluate total cost.
    nargout : int, optional
        MATLAB compatibility flag.

    Returns
    -------
    numpy.ndarray or float
        Total production costs with the same shape as ``Pg``.
    """
    gencost = np.asarray(gencost, dtype=float)
    Pg = np.asarray(Pg, dtype=float)
    original_shape = Pg.shape
    if Pg.ndim == 0:
        Pg = Pg.reshape(1, 1)
    elif Pg.ndim == 1:
        Pg = Pg.reshape(-1, 1)

    ng, m = gencost.shape
    totalcost = np.zeros((ng, Pg.shape[1]), dtype=float)

    if gencost.size:
        ipwl = np.flatnonzero(gencost[:, MODEL - 1] == PW_LINEAR)
        ipol = np.flatnonzero(gencost[:, MODEL - 1] == POLYNOMIAL)
        if ipwl.size:
            x = gencost[:, COST - 1 : m - 1 : 2]
            y = gencost[:, COST:m:2]
            for i in ipwl:
                ncost = int(gencost[i, NCOST - 1])
                if ncost > 0:
                    totalcost[i, :] = _totcost_pwl(x[i, :ncost], y[i, :ncost], Pg[i, :])
        for i in range(totalcost.shape[1]):
            if ipol.size:
                totalcost[ipol, i] = np.asarray(polycost(gencost[ipol, :], Pg[ipol, i])).reshape(-1)

    if original_shape == ():
        return float(totalcost.reshape(-1)[0])
    if len(original_shape) == 1:
        return totalcost.reshape(original_shape)
    return totalcost.reshape(original_shape)
