# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from .idx_cost import COST, MODEL, NCOST, POLYNOMIAL, PW_LINEAR
from .polycost import polycost


def margcost(gencost, Pg):
    """Evaluate marginal generator production costs.

    Mirrors MATPOWER's ``margcost`` helper by evaluating the slope of either
    piecewise linear or polynomial generator cost models at the supplied
    output levels.

    Parameters
    ----------
    gencost : array_like
        Generator cost matrix.
    Pg : array_like
        Generator output values at which to evaluate marginal cost.
    nargout : int, optional
        MATLAB compatibility flag.

    Returns
    -------
    numpy.ndarray or float
        Marginal costs with the same shape as ``Pg``.
    """
    gencost = np.asarray(gencost, dtype=float)
    Pg = np.asarray(Pg, dtype=float)
    original_shape = Pg.shape
    if Pg.ndim == 0:
        Pg = Pg.reshape(1, 1)
    elif Pg.ndim == 1:
        Pg = Pg.reshape(-1, 1)

    ng, m = gencost.shape
    marginalcost = np.zeros((ng, Pg.shape[1]), dtype=float)

    if gencost.size:
        ipwl = np.flatnonzero(gencost[:, MODEL - 1] == PW_LINEAR)
        ipol = np.flatnonzero(gencost[:, MODEL - 1] == POLYNOMIAL)
        if ipwl.size:
            x = gencost[:, COST - 1 : m - 1 : 2]
            y = gencost[:, COST:m:2]
            for i in ipwl:
                ncost = int(gencost[i, NCOST - 1])
                if ncost > 0:
                    c = np.diff(y[i, :ncost]) / np.diff(x[i, :ncost])
                    for j, p in enumerate(Pg[i, :]):
                        k = np.flatnonzero(p <= x[i, :ncost])
                        if k.size == 0:
                            marginalcost[i, j] = c[-1]
                        elif k[0] == 0:
                            marginalcost[i, j] = c[0]
                        else:
                            marginalcost[i, j] = c[k[0] - 1]
        if ipol.size:
            for i in range(Pg.shape[1]):
                marginalcost[ipol, i] = np.asarray(polycost(gencost[ipol, :], Pg[ipol, i], 1)).reshape(-1)

    if original_shape == ():
        return float(marginalcost.reshape(-1)[0])
    if len(original_shape) == 1:
        return marginalcost.reshape(original_shape)
    return marginalcost.reshape(original_shape)
