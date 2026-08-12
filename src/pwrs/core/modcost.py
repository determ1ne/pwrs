# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from scipy import sparse

from .idx_cost import COST, MODEL, NCOST, POLYNOMIAL, PW_LINEAR


def _polyshift(c: np.ndarray, a: float) -> np.ndarray:
    n = len(c)
    d = np.zeros_like(c, dtype=float)
    A = (-a * np.ones(n)) ** np.arange(n)
    b = np.ones(n)
    for k in range(1, n + 1):
        d[n - k] = b @ (c[n - k :: -1] * A[: n - k + 1])
        if n - k > 0:
            b = np.cumsum(b[: n - k])
    return d


def modcost(gencost, alpha, modtype="SCALE_F"):
    """Modify generator cost curves by scaling or shifting.

    Mirrors MATPOWER's ``modcost`` helper. It applies multiplicative or
    additive modifications to polynomial or piecewise-linear cost functions
    in either the cost axis or the quantity axis.

    Parameters
    ----------
    gencost : array_like
        Generator cost matrix.
    alpha : float or array_like
        Scalar or per-generator modification factor.
    modtype : str, optional
        Modification type, such as ``'SCALE_F'``, ``'SCALE_X'``,
        ``'SHIFT_F'``, or ``'SHIFT_X'``.
    nargout : int, optional
        MATLAB compatibility flag.

    Returns
    -------
    numpy.ndarray
        Modified generator cost matrix.
    """
    gencost = np.array(gencost, dtype=float, copy=True)
    modtype = str(modtype)
    ng, m = gencost.shape

    if ng == 0:
        return gencost

    alpha = np.asarray(alpha, dtype=float)
    if alpha.size != ng:
        if alpha.size == 1 and ng > 1:
            alpha = np.full((ng, 1), alpha)
        else:
            raise ValueError("modcost: ALPHA must be a scalar or col vector with NG rows")
    elif alpha.ndim > 1 and alpha.shape[1] != 1:
        alpha = alpha.T
    alpha = alpha.reshape(-1)

    ipwl = np.flatnonzero(gencost[:, MODEL - 1] == PW_LINEAR)
    ipol = np.flatnonzero(gencost[:, MODEL - 1] == POLYNOMIAL)
    npwl = len(ipwl)
    c = gencost[np.ix_(ipol, np.arange(COST - 1, m))]

    if modtype == "SCALE_F":
        if len(ipol):
            gencost[ipol, COST - 1 : m] = sparse.diags(alpha[ipol], offsets=0, shape=(len(ipol), len(ipol))) @ c
        if npwl:
            cols = np.arange(COST, m, 2)
            gencost[np.ix_(ipwl, cols)] = (
                sparse.diags(alpha[ipwl], offsets=0, shape=(npwl, npwl)) @ gencost[np.ix_(ipwl, cols)]
            )
    elif modtype == "SCALE_X":
        for k, row in enumerate(ipol):
            n = int(gencost[row, NCOST - 1])
            for i in range(1, n + 1):
                gencost[row, COST + i - 2] = c[k, i - 1] / alpha[row] ** (n - i)
        if npwl:
            cols = np.arange(COST - 1, m - 1, 2)
            gencost[np.ix_(ipwl, cols)] = (
                sparse.diags(alpha[ipwl], offsets=0, shape=(npwl, npwl)) @ gencost[np.ix_(ipwl, cols)]
            )
    elif modtype == "SHIFT_F":
        for k, row in enumerate(ipol):
            n = int(gencost[row, NCOST - 1])
            gencost[row, COST + n - 2] = alpha[row] + c[k, n - 1]
        if npwl:
            cols = np.arange(COST, m, 2)
            gencost[np.ix_(ipwl, cols)] = (
                sparse.diags(alpha[ipwl], offsets=0, shape=(npwl, npwl)) @ np.ones((npwl, len(cols)))
                + gencost[np.ix_(ipwl, cols)]
            )
    elif modtype == "SHIFT_X":
        for k, row in enumerate(ipol):
            n = int(gencost[row, NCOST - 1])
            gencost[row, COST - 1 : COST + n - 1] = _polyshift(c[k, :n], float(alpha[row]))
        if npwl:
            cols = np.arange(COST - 1, m - 1, 2)
            gencost[np.ix_(ipwl, cols)] = (
                sparse.diags(alpha[ipwl], offsets=0, shape=(npwl, npwl)) @ np.ones((npwl, len(cols)))
                + gencost[np.ix_(ipwl, cols)]
            )
    else:
        raise ValueError(f"modcost: '{modtype}' is not a valid modtype")

    return gencost
