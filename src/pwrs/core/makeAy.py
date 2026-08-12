# Copyright (c) 1996-2016, Power Systems Engineering Research Center (PSERC) by Ray Zimmerman, PSERC Cornell
# Modifications Copyright (c) 2026, Liangyu Zhang
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from scipy import sparse

from .idx_cost import COST, MODEL, NCOST, PW_LINEAR


def makeAy(baseMVA, ng, gencost, pgbas, qgbas, ybas, nargout=1):
    """Construct basin constraints for the CCV piecewise-linear formulation.

    Builds the linear constraint ``Ay * X <= by`` used by the
    constrained-cost-variable formulation for piecewise-linear generator
    costs.

    Parameters
    ----------
    baseMVA : float
        System power base.
    ng : int
        Number of generators.
    gencost : ndarray
        Generator cost matrix.
    pgbas, qgbas, ybas : int
        Starting indices within the optimization variable vector for
        active-power, reactive-power, and CCV ``y`` variables.
    nargout : int, optional
        Number of outputs to emulate from the MATLAB interface.

    Returns
    -------
    tuple or sparse matrix
        Returns ``(Ay, by)`` or the leading subset requested by
        ``nargout``.
    """
    iycost = np.flatnonzero(gencost[:, MODEL] == PW_LINEAR)
    ny = len(iycost)
    if ny == 0:
        out = (sparse.csc_matrix((0, ybas + ny - 1)), np.array([]))
        return out[:nargout] if nargout > 1 else out[0]
    total_cost_points = int(np.sum(gencost[iycost, NCOST]))
    Ay = sparse.lil_matrix((total_cost_points - ny, ybas + ny - 1))
    by: list[float] = []
    k = 0
    for i in iycost:
        ns = int(gencost[i, NCOST])
        p = gencost[i, COST : COST + 2 * ns : 2] / baseMVA
        c = gencost[i, COST + 1 : COST + 2 * ns : 2]
        m = np.diff(c) / np.diff(p)
        b = m * p[:-1] - c[:-1]
        by.extend(b.tolist())
        sidx = qgbas + (i + 1 - ng) - 2 if i + 1 > ng else pgbas + i - 1
        Ay[k : k + ns - 1, sidx] = m.reshape(-1, 1)
        k += ns - 1
    k = 0
    j = 0
    for i in iycost:
        ns = int(gencost[i, NCOST])
        Ay[k : k + ns - 1, ybas + j - 1] = -np.ones((ns - 1, 1))
        k += ns - 1
        j += 1
    out = (Ay.tocsr(), np.asarray(by, dtype=float))
    return out[:nargout] if nargout > 1 else out[0]


def makeAy_full(baseMVA, ng, gencost, pgbas, qgbas, ybas):
    """Return ``(Ay, by)`` for the CCV formulation."""
    return makeAy(baseMVA, ng, gencost, pgbas, qgbas, ybas, nargout=2)
