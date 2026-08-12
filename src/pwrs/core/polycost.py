# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from .idx_cost import COST, MODEL, NCOST, PW_LINEAR


def polycost(gencost, Pg, der=0):
    """Evaluate polynomial generator cost functions or derivatives.

    Mirrors MATPOWER's ``polycost`` helper by evaluating polynomial cost
    curves, or their derivatives of order ``der``, for the supplied
    generator output levels.

    Parameters
    ----------
    gencost : array_like
        Polynomial generator cost matrix.
    Pg : array_like
        Generator output values at which to evaluate the cost.
    der : int, optional
        Derivative order. ``0`` evaluates the cost itself.
    nargout : int, optional
        MATLAB compatibility flag.

    Returns
    -------
    numpy.ndarray or float
        Evaluated polynomial cost values with the same shape as ``Pg``.
    """
    gencost = np.asarray(gencost, dtype=float)
    Pg = np.asarray(Pg, dtype=float)
    original_shape = Pg.shape
    if Pg.ndim == 0:
        Pg = Pg.reshape(1, 1)
    elif Pg.ndim == 1:
        Pg = Pg.reshape(-1, 1)

    if np.any(gencost[:, MODEL - 1] == PW_LINEAR):
        raise ValueError("polycost: all costs must be polynomial")

    ng = Pg.shape[0]
    max_n = int(np.max(gencost[:, NCOST - 1]))
    min_n = int(np.min(gencost[:, NCOST - 1]))

    c = np.zeros((ng, max_n), dtype=float)
    for n in range(min_n, max_n + 1):
        k = np.flatnonzero(gencost[:, NCOST - 1] == n)
        if k.size:
            c[np.ix_(k, np.arange(n))] = gencost[np.ix_(k, np.arange(COST + n - 2, COST - 2, -1))]

    for d in range(1, int(der) + 1):
        if c.shape[1] >= 2:
            c = c[:, 1 : max_n - d + 1]
        else:
            c = np.zeros((ng, 1), dtype=float)
            break
        for k in range(1, max_n - d):
            c[:, k] = (k + 1) * c[:, k]

    if c.size == 0:
        f = np.zeros_like(Pg, dtype=float)
    else:
        f = np.tile(c[:, [0]], (1, Pg.shape[1]))
        for k in range(1, c.shape[1]):
            f = f + c[:, [k]] * Pg**k

    if original_shape == ():
        return float(f.reshape(-1)[0])
    if len(original_shape) == 1:
        return f.reshape(original_shape)
    return f.reshape(original_shape)
